"""
TwoDThreeDNetV2 — most advanced transformer for monocular 3D localization.

Architecture improvements over TwoDThreeDFormer (model_v3.py):

  1. Multi-stream encoder
     - Appearance stream  : bbox features d[0:4]      → 2-layer RoPE encoder
     - Geometry stream    : ground projections d̃[4:14] → 2-layer RoPE encoder + CAPE PE
     - Camera stream      : camera params δ[14:18]    → single context token (mean-pooled)
     Streams fused with a 4-layer cross-attention fusion encoder.

  2. RoPE temporal encoding  (LLaMA / GPT-4 style)
     Rotary Position Embedding applied to Q and K within each attention layer.
     Better length-generalisation and relative-position sensitivity than additive PE.

  3. Iterative geometry-anchored decoder  (DAB-DETR / Sparse4D style)
     Decoder queries are initialized from per-class embeddings + a 2D ground-plane
     anchor estimated from the last geometry-stream token.
     Each decoder layer refines the anchor (Δx, Δy, z updated iteratively).

  4. Deep supervision
     An output head is applied after EVERY decoder layer.
     All intermediate predictions contribute to training loss.

  5. Class-conditioned queries
     Separate learned embeddings for car, bus, pedestrian.  The class token from
     the dataset (index in {0,1,2}) selects the right base query.

Input feature layout (enc_in_dim = 18):
  [0:4]   d      — bbox (x1, y1, x2, y2) normalised
  [4:14]  d̃      — 10 ground-plane projections normalised
  [14:18] δ      — camera (focal, pitch, roll, height) normalised

Output (per forward call):
  main_pred  : (B, S, dec_out_dim=8)   — final decoder layer output
  aux_preds  : list[(B, S, 8)]         — one tensor per intermediate layer

Drop-in for training:  set model_class = "TwoDThreeDNetV2" in config.
Requires TrainerDeepSupervision in train_v2.py for aux loss handling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ──────────────────────────────────────────────────────────────────
# 1.  RoPE utilities
# ──────────────────────────────────────────────────────────────────

def _rope_freqs(dim: int, base: float = 10000.0, device=None) -> Tensor:
    """Pre-compute cos/sin frequency table for RoPE."""
    assert dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return theta                                          # (dim//2,)


def _apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """
    Apply rotary position embedding to x.
    x     : (B, T, H, head_dim)   — split into pairs along last axis
    freqs : (T, head_dim//2)      — per-position frequencies
    """
    B, T, H, D = x.shape
    x1 = x[..., 0::2]                                    # (B, T, H, D//2)
    x2 = x[..., 1::2]
    cos = freqs[:T, :].unsqueeze(0).unsqueeze(2)          # (1, T, 1, D//2)
    sin = freqs[:T, :].unsqueeze(0).unsqueeze(2)

    # Pre-compute actual cos/sin from stored theta
    # freqs here is (T, D//2) already in cos/sin form — handled by RoPECache
    cos_val = freqs[:T, :, 0].unsqueeze(0).unsqueeze(2)   # (1,T,1,D//2)
    sin_val = freqs[:T, :, 1].unsqueeze(0).unsqueeze(2)

    r1 = x1 * cos_val - x2 * sin_val
    r2 = x1 * sin_val + x2 * cos_val

    out = torch.stack([r1, r2], dim=-1).flatten(-2)        # (B,T,H,D)
    return out


class RoPECache(nn.Module):
    """Cached (cos, sin) pairs for sequences up to max_len."""

    def __init__(self, head_dim: int, max_len: int = 256, base: float = 10000.0):
        super().__init__()
        theta = _rope_freqs(head_dim, base)                # (head_dim//2,)
        t     = torch.arange(max_len).float()
        freqs = torch.outer(t, theta)                      # (max_len, head_dim//2)
        # stack cos and sin on last dim → (max_len, head_dim//2, 2)
        rope  = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer('rope', rope)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """
        q, k : (B, T, H, head_dim)
        Returns rotated q and k.
        """
        freqs = self.rope                                  # (max_len, D//2, 2)
        return _apply_rope(q, freqs), _apply_rope(k, freqs)


# ──────────────────────────────────────────────────────────────────
# 2.  RoPE-augmented multi-head self-attention layer
# ──────────────────────────────────────────────────────────────────

class RoPETransformerEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer with RoPE applied to Q and K.
    Replaces sinusoidal / learnable additive PE for temporal sequences.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model   = d_model
        self.nhead     = nhead
        self.head_dim  = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out    = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)
        self.rope  = RoPECache(self.head_dim)

    def _reshape(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        return x.reshape(B, T, self.nhead, self.head_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Pre-norm self-attention
        r   = self.norm1(x)
        q   = self._reshape(self.q_proj(r))
        k   = self._reshape(self.k_proj(r))
        v   = self._reshape(self.v_proj(r))
        q, k = self.rope(q, k)                             # RoPE rotation

        B, T, H, D = q.shape
        q = q.transpose(1, 2)                              # (B,H,T,D)
        k = k.transpose(1, 2)
        v = v.reshape(B, T, H, D).transpose(1, 2)

        scale  = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)                        # (B,H,T,D)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        x   = x + self.drop(self.out(out))

        # Pre-norm FF
        x = x + self.ff(self.norm2(x))
        return x


class RoPETransformerEncoder(nn.Module):
    """Stack of RoPETransformerEncoderLayers."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────
# 3.  Cross-attention layer (for camera-context modulation)
# ──────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """Standard cross-attention: Q from query stream, KV from context."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        # Pre-norm cross-attn
        r = self.norm1(query)
        out, _ = self.attn(r, context, context)
        query  = query + self.drop(out)
        query  = query + self.ff(self.norm2(query))
        return query


# ──────────────────────────────────────────────────────────────────
# 4.  CAPE ray encoding  (reused from model_v3, standalone here)
# ──────────────────────────────────────────────────────────────────

class CAPERayEncoding(nn.Module):
    """
    Computes 3D ray direction for each temporal step and projects to d_model.
    Requires features [0:4] = (x1,y1,x2,y2) and [focal_idx] = focal length.
    """

    def __init__(self, d_model: int, cfg):
        super().__init__()
        self.register_buffer('bbox_mean_x', torch.tensor(cfg.bbox2d_mean[0]).float())
        self.register_buffer('bbox_mean_y', torch.tensor(cfg.bbox2d_mean[1]).float())
        self.register_buffer('bbox_std_x',  torch.tensor(cfg.bbox2d_std[0]).float())
        self.register_buffer('bbox_std_y',  torch.tensor(cfg.bbox2d_std[1]).float())
        self.register_buffer('img_cx', torch.tensor(cfg.image_size[0] / 2.0).float())
        self.register_buffer('img_cy', torch.tensor(cfg.image_size[1] / 2.0).float())

        focal_mean = sum(cfg.focal_length_range) / 2.0
        focal_std  = (cfg.focal_length_range[1] - cfg.focal_length_range[0]) / math.sqrt(12)
        self.register_buffer('focal_mean', torch.tensor(focal_mean).float())
        self.register_buffer('focal_std',  torch.tensor(max(focal_std, 1e-8)).float())

        self.focal_idx = getattr(cfg, 'focal_feature_idx', 14)
        self.ray_proj  = nn.Linear(3, d_model, bias=True)
        nn.init.normal_(self.ray_proj.weight, std=0.02)
        nn.init.zeros_(self.ray_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        x1_px = x[..., 0] * self.bbox_std_x + self.bbox_mean_x
        y1_px = x[..., 1] * self.bbox_std_y + self.bbox_mean_y
        x2_px = x[..., 2] * self.bbox_std_x + self.bbox_mean_x
        y2_px = x[..., 3] * self.bbox_std_y + self.bbox_mean_y
        cx_px = (x1_px + x2_px) * 0.5
        cy_px = (y1_px + y2_px) * 0.5
        f_px  = (x[..., self.focal_idx] * self.focal_std + self.focal_mean).clamp(min=1.0)
        dx = (cx_px - self.img_cx) / f_px
        dy = (cy_px - self.img_cy) / f_px
        dz = torch.ones_like(dx)
        ray = F.normalize(torch.stack([dx, dy, dz], dim=-1), dim=-1)
        return self.ray_proj(ray)                          # (B, T, d_model)


# ──────────────────────────────────────────────────────────────────
# 5.  Specialised stream encoders
# ──────────────────────────────────────────────────────────────────

class AppearanceEncoder(nn.Module):
    """
    Stream 1 — Appearance: bbox features d[0:4]
    Projects to D//2, then 2-layer RoPE encoder.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        half = d_model // 2
        self.proj    = nn.Linear(4, half)
        self.encoder = RoPETransformerEncoder(half, max(1, nhead // 2),
                                              dim_ff // 2, 2, dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(self.proj(x[..., 0:4]))        # (B, T, D//2)


class GeometryEncoder(nn.Module):
    """
    Stream 2 — Geometry: ground projections d̃[4:14]
    Projects to D//2, adds CAPE ray PE (from full 18-dim input), 2-layer RoPE encoder.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 dropout: float, cfg):
        super().__init__()
        half = d_model // 2
        self.proj    = nn.Linear(10, half)
        self.cape    = CAPERayEncoding(half, cfg)
        self.encoder = RoPETransformerEncoder(half, max(1, nhead // 2),
                                              dim_ff // 2, 2, dropout)

    def forward(self, full_x: Tensor) -> Tensor:
        """full_x : (B, T, 18)  — needs full input for CAPE"""
        geo_feat = self.proj(full_x[..., 4:14])            # (B, T, D//2)
        geo_feat = geo_feat + self.cape(full_x)            # add CAPE
        return self.encoder(geo_feat)                      # (B, T, D//2)


class CameraContextEncoder(nn.Module):
    """
    Stream 3 — Camera context: δ[14:18]
    Mean-pooled over time → single context token (B, 1, D).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        cam = x[..., 14:18].mean(dim=1, keepdim=True)      # (B, 1, 4)
        return self.proj(cam)                               # (B, 1, D)


# ──────────────────────────────────────────────────────────────────
# 6.  Fusion encoder
# ──────────────────────────────────────────────────────────────────

class FusionEncoder(nn.Module):
    """
    Fuses appearance + geometry streams (after concat to D) through a 4-layer
    RoPE encoder, then applies camera-context cross-attention.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.rope_enc  = RoPETransformerEncoder(d_model, nhead, dim_ff, 4, dropout)
        self.cam_cross = CrossAttentionLayer(d_model, nhead, dropout)

    def forward(self, fused: Tensor, camera_ctx: Tensor) -> Tensor:
        """
        fused      : (B, T, D)   — concat of appearance and geometry
        camera_ctx : (B, 1, D)   — single camera context token
        """
        memory = self.rope_enc(fused)
        memory = self.cam_cross(memory, camera_ctx)
        return memory                                      # (B, T, D)


# ──────────────────────────────────────────────────────────────────
# 7.  Iterative geometry-anchored decoder
# ──────────────────────────────────────────────────────────────────

class AnchorEmbedding(nn.Module):
    """Encodes a (Δx, Δy, z) anchor position into a d_model embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, anchor: Tensor) -> Tensor:
        return self.mlp(anchor)                            # (B, S, D)


class IterativeDecoderLayer(nn.Module):
    """
    One layer of the iterative decoder.
    - Self-attention over current queries
    - Cross-attention with encoder memory
    - Anchor refinement: predict Δ(x,y,z) and add to running anchor
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float,
                 dec_out_dim: int):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead,
                                                 dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead,
                                                 dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)
        # Small MLP to predict refined Δ(x,y,z) from current query
        self.anchor_refiner = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, dec_out_dim),
        )

    def forward(
        self,
        query:  Tensor,
        memory: Tensor,
        anchor: Tensor,
        anchor_embed_fn,
        tgt_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        query  : (B, S, D)
        memory : (B, T, D)
        anchor : (B, S, 3)   — current Δx, Δy, z estimate
        Returns updated (query, anchor, prediction).
        """
        # Pre-norm self-attention
        r = self.norm1(query)
        sa, _ = self.self_attn(r, r, r, attn_mask=tgt_mask)
        query = query + self.drop(sa)

        # Pre-norm cross-attention
        r = self.norm2(query)
        ca, _ = self.cross_attn(r, memory, memory)
        query = query + self.drop(ca)

        # FF
        query = query + self.ff(self.norm3(query))

        # Anchor refinement: update anchor and re-inject as positional bias
        delta_anchor = self.anchor_refiner(query)          # (B, S, 3)
        anchor       = anchor + delta_anchor               # iterative update
        query        = query + anchor_embed_fn(anchor)     # spatial grounding

        pred = self.output_head(query)                     # (B, S, dec_out_dim)
        return query, anchor, pred


class IterativeDecoder(nn.Module):
    """
    Geometry-anchored iterative decoder with deep supervision.
    Anchor initialized from last geometry-stream token projected to (Δx, Δy, z).
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 num_layers: int, dropout: float, dec_seq_len: int,
                 dec_out_dim: int, num_classes: int = 3):
        super().__init__()
        self.dec_seq_len = dec_seq_len

        # Class-conditioned base queries: separate learned init per class
        self.class_queries = nn.Embedding(num_classes, d_model)
        nn.init.normal_(self.class_queries.weight, std=0.02)

        # Per-step positional queries (T prediction steps)
        self.step_queries = nn.Embedding(dec_seq_len, d_model)

        # Project last geometry token → initial anchor (Δx, Δy, z)
        self.anchor_init = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )

        self.anchor_embed = AnchorEmbedding(d_model)

        self.layers = nn.ModuleList([
            IterativeDecoderLayer(d_model, nhead, dim_ff, dropout, dec_out_dim)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        memory:       Tensor,
        geo_last:     Tensor,
        class_tokens: Optional[Tensor] = None,
        tgt_mask:     Optional[Tensor] = None,
    ) -> tuple[Tensor, list]:
        """
        memory       : (B, T, D)     — fusion encoder output
        geo_last     : (B, D)        — last token of geometry stream
        class_tokens : (B,) int      — class index per sample (0=car,1=bus,2=ped)
        Returns (final_pred, aux_preds) where aux_preds is a list of intermediate preds.
        """
        B = memory.size(0)
        S = self.dec_seq_len

        # Class-conditioned base query
        if class_tokens is not None:
            cls_tok = class_tokens.to(memory.device).clamp(0, self.class_queries.num_embeddings - 1)
            cls_emb = self.class_queries(cls_tok)          # (B, D)
            cls_emb = cls_emb.unsqueeze(1).expand(B, S, -1)
        else:
            cls_emb = self.class_queries.weight[0].unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        # Per-step positional queries
        s_idx   = torch.arange(S, device=memory.device)
        step_pe = self.step_queries(s_idx).unsqueeze(0).expand(B, -1, -1)

        query = cls_emb + step_pe                          # (B, S, D)

        # Initial anchor from geometry stream last token
        anchor = self.anchor_init(geo_last).unsqueeze(1).expand(B, S, -1)  # (B,S,3)

        # Ground initial query with anchor embedding
        query = query + self.anchor_embed(anchor)

        aux_preds = []
        for layer in self.layers:
            query, anchor, pred = layer(
                query, memory, anchor, self.anchor_embed, tgt_mask
            )
            aux_preds.append(pred)

        final_pred = aux_preds[-1]
        return final_pred, aux_preds[:-1]


# ──────────────────────────────────────────────────────────────────
# 8.  Main model
# ──────────────────────────────────────────────────────────────────

class TwoDThreeDNetV2(nn.Module):
    """
    Most advanced monocular 3D localization transformer.

    Config requirements (add to config_outdoor.py):
        d_model, nhead, num_encoder_layers (fusion), num_decoder_layers,
        dim_feedforward, dropout, enc_in_dim (=18), dec_out_dim (=8),
        enc_seq_len, dec_seq_len, use_tgt_mask,
        bbox2d_mean, bbox2d_std, image_size, focal_length_range,
        focal_feature_idx (=14), num_classes (=3)
    """

    def __init__(self, cfg):
        super().__init__()

        d_model      = cfg.d_model
        nhead        = cfg.nhead
        dim_ff       = cfg.dim_feedforward
        dropout      = cfg.dropout
        dec_seq_len  = cfg.dec_seq_len
        dec_out_dim  = cfg.dec_out_dim
        num_classes  = getattr(cfg, 'num_classes', 3)

        # Number of decoder layers (default 4)
        num_dec_layers = cfg.num_decoder_layers

        # ── Stream encoders ─────────────────────────────────────────
        self.app_enc  = AppearanceEncoder(d_model, nhead, dim_ff, dropout)
        self.geo_enc  = GeometryEncoder(d_model, nhead, dim_ff, dropout, cfg)
        self.cam_enc  = CameraContextEncoder(d_model)

        # ── Fusion encoder ──────────────────────────────────────────
        self.fusion   = FusionEncoder(d_model, nhead, dim_ff, dropout)

        # ── Iterative decoder ────────────────────────────────────────
        self.decoder  = IterativeDecoder(
            d_model, nhead, dim_ff, num_dec_layers, dropout,
            dec_seq_len, dec_out_dim, num_classes,
        )

        # Causal mask (optional)
        self.register_buffer(
            'tgt_mask',
            nn.Transformer.generate_square_subsequent_mask(dec_seq_len)
        )
        self.use_tgt_mask = cfg.use_tgt_mask

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        src:          Tensor,
        class_tokens: Optional[Tensor] = None,
    ) -> tuple[Tensor, list]:
        """
        src          : (B, T, 18)
        class_tokens : (B,) int — optional; falls back to car (0) if None
        Returns:
          main_pred  : (B, S, 8)      — final decoder output
          aux_preds  : list[(B,S,8)]  — intermediate predictions (for deep supervision)
        """
        # Stream encoding
        app_out = self.app_enc(src)                        # (B, T, D//2)
        geo_out = self.geo_enc(src)                        # (B, T, D//2)
        cam_ctx = self.cam_enc(src)                        # (B, 1, D)

        # Fuse appearance + geometry → (B, T, D)
        fused  = torch.cat([app_out, geo_out], dim=-1)
        memory = self.fusion(fused, cam_ctx)               # (B, T, D)

        # Last geometry token for anchor initialisation
        geo_last = geo_out[:, -1, :]                       # (B, D//2)
        # Pad to d_model by repeating (simple, no extra params)
        geo_last = torch.cat([geo_last, geo_last], dim=-1) # (B, D)

        tgt_mask = self.tgt_mask if self.use_tgt_mask else None

        main_pred, aux_preds = self.decoder(
            memory, geo_last, class_tokens, tgt_mask
        )
        return main_pred, aux_preds


# ──────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class _MockCfg:
        d_model            = 128
        nhead              = 8
        num_encoder_layers = 4
        num_decoder_layers = 4
        dim_feedforward    = 512
        dropout            = 0.1
        enc_in_dim         = 18
        dec_out_dim        = 8
        enc_seq_len        = 12
        dec_seq_len        = 4
        use_tgt_mask       = False
        num_classes        = 3
        bbox2d_mean        = [960.0, 540.0]
        bbox2d_std         = [640.0, 360.0]
        image_size         = (1920, 1080)
        focal_length_range = [930.0, 1000.0]
        focal_feature_idx  = 14

    cfg    = _MockCfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TwoDThreeDNetV2(cfg).to(device)

    B   = 8
    T   = cfg.enc_seq_len
    src = torch.randn(B, T, cfg.enc_in_dim, device=device)
    src[:, :, 0] = torch.rand(B, T, device=device) * 2 - 1
    src[:, :, 1] = torch.rand(B, T, device=device) * 2 - 1
    src[:, :, 2] = src[:, :, 0] + 0.2
    src[:, :, 3] = src[:, :, 1] + 0.3
    src[:, :, cfg.focal_feature_idx] = torch.randn(B, T, device=device) * 0.1

    class_tokens = torch.randint(0, 3, (B,), device=device)
    main_pred, aux_preds = model(src, class_tokens)

    assert main_pred.shape == (B, cfg.dec_seq_len, cfg.dec_out_dim), \
        f"Bad main shape: {main_pred.shape}"
    for i, a in enumerate(aux_preds):
        assert a.shape == (B, cfg.dec_seq_len, cfg.dec_out_dim), \
            f"Bad aux[{i}] shape: {a.shape}"

    loss = main_pred.mean() + sum(a.mean() for a in aux_preds)
    loss.backward()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"TwoDThreeDNetV2 OK")
    print(f"  main output : {main_pred.shape}")
    print(f"  aux outputs : {len(aux_preds)} × {aux_preds[0].shape if aux_preds else 'none'}")
    print(f"  Total params: {total_params:,}")

    # Stream-wise breakdown
    streams = {
        'AppearanceEncoder': model.app_enc,
        'GeometryEncoder':   model.geo_enc,
        'CameraEncoder':     model.cam_enc,
        'FusionEncoder':     model.fusion,
        'IterativeDecoder':  model.decoder,
    }
    for name, mod in streams.items():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:22s}: {n:>8,}  ({100*n/total_params:.1f}%)")

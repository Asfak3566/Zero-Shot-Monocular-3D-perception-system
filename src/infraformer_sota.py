"""
InfraFormerSOTA — State-of-the-art monocular 3D localization for infrastructure cameras.

Architecture advances over TwoDThreeDNetV2 (model_v4.py):

  1. Real Waymo occlusion conditioning
     The raw per-frame occlusion score (0=visible, 1=fully occluded) is appended as
     a 17th input feature. This lets the model explicitly weight observations by quality.

  2. Social interaction stream
     Up to 5 neighboring agents contribute an 8D social feature vector per frame.
     A masked mean-pool aggregates them; a learned sigmoid gate fuses the result with
     the ego stream — so the model attends to context only when it helps.

  3. 4-class conditioning (car / bus / pedestrian / cyclist)
     IterativeDecoder uses separate learned base queries for all 4 classes.

  4. Compact single-stream ego encoder + fusion encoder
     Linear(17→D) + CAPE + 4-layer RoPE encoder for ego.
     2-layer RoPE fusion encoder after social gating.
     Camera context token cross-attends final memory.

Input layout (enc_in_dim = 17):
  [0:4]   d      — bbox (u, v, j, k) normalised  ← center + dims, paper Eq 2
  [4:12]  d̃      — 8D ground-plane projections   ← 4 corners
  [12:16] δ      — camera (focal, pitch, roll, h)
  [16]    occ    — Waymo occlusion score [0, 1]

Social inputs:
  social_src  : (B, T, 5, 8) — per-frame features of up to 5 neighbours
  social_mask : (B, T, 5)    — 1.0 where neighbour slot is occupied

Output:
  main_pred  : (B, S, 8)  — [Δx, Δy, z, l, w, h, cos θ, sin θ]
  aux_preds  : list of (B, S, 8) — one per intermediate decoder layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# Reuse RoPE, cross-attention, and iterative decoder from model_v4
from model_v4 import (
    RoPETransformerEncoder,
    CrossAttentionLayer,
    AnchorEmbedding,
    IterativeDecoderLayer,
    IterativeDecoder,
)


# ──────────────────────────────────────────────────────────────────
# 1.  CAPE for (u, v, j, k) bbox format
# ──────────────────────────────────────────────────────────────────

class InfraCAPERayEncoding(nn.Module):
    """
    Camera-Aware Positional Encoding adapted for the PaperAligned bbox format.

    In the 17D input d[0:4] = (u, v, j, k) where u and v are already the bbox
    *centre* coordinates — so cx = u directly (no corner averaging needed).

    Focal length is at d[focal_feature_idx] inside the δ block (default idx 12).
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

        self.focal_idx = getattr(cfg, 'focal_feature_idx', 12)
        self.ray_proj  = nn.Linear(3, d_model, bias=True)
        nn.init.normal_(self.ray_proj.weight, std=0.02)
        nn.init.zeros_(self.ray_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, T, D_in ≥ 17)  →  (B, T, d_model)"""
        cx_px = x[..., 0] * self.bbox_std_x + self.bbox_mean_x   # u → centre_x
        cy_px = x[..., 1] * self.bbox_std_y + self.bbox_mean_y   # v → centre_y
        f_px  = (x[..., self.focal_idx] * self.focal_std + self.focal_mean).clamp(min=1.0)
        dx = (cx_px - self.img_cx) / f_px
        dy = (cy_px - self.img_cy) / f_px
        dz = torch.ones_like(dx)
        ray = F.normalize(torch.stack([dx, dy, dz], dim=-1), dim=-1)
        return self.ray_proj(ray)                                  # (B, T, d_model)


# ──────────────────────────────────────────────────────────────────
# 2.  Social context encoder
# ──────────────────────────────────────────────────────────────────

class SocialContextEncoder(nn.Module):
    """
    Aggregates up to N neighbour feature vectors via masked mean-pooling.

    Input  : (B, T, N, social_dim)   neighbour features
    Mask   : (B, T, N)               1.0 = valid slot, 0.0 = empty
    Output : (B, T, d_social)        context vector per timestep
    """

    def __init__(self, social_dim: int, d_social: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(social_dim, d_social),
            nn.GELU(),
        )

    def forward(self, social_src: Tensor, social_mask: Optional[Tensor]) -> Tensor:
        feats = self.proj(social_src)                              # (B, T, N, D_s)
        if social_mask is not None:
            valid = social_mask.unsqueeze(-1).float()             # (B, T, N, 1)
            feats = feats * valid
            count = valid.sum(dim=-2).clamp(min=1.0)             # (B, T, 1)
            return feats.sum(dim=-2) / count                      # (B, T, D_s)
        return feats.mean(dim=-2)


# ──────────────────────────────────────────────────────────────────
# 3.  Gated social fusion
# ──────────────────────────────────────────────────────────────────

class GatedSocialFusion(nn.Module):
    """
    Fuses ego-encoded features with social context using a per-dim sigmoid gate.

      fused = ego + sigmoid(W_g · ego) ⊙ W_up(social_ctx)

    The gate is conditioned on the ego stream so the model suppresses social
    information when it is not useful (e.g. isolated pedestrians).
    """

    def __init__(self, d_model: int, d_social: int):
        super().__init__()
        self.social_up = nn.Linear(d_social, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)

    def forward(self, ego: Tensor, social_ctx: Tensor) -> Tensor:
        """
        ego        : (B, T, D)
        social_ctx : (B, T, D_s)
        """
        social_up = self.social_up(social_ctx)                    # (B, T, D)
        gate      = torch.sigmoid(self.gate_proj(ego))            # (B, T, D)
        return ego + gate * social_up


# ──────────────────────────────────────────────────────────────────
# 4.  Camera context projector
# ──────────────────────────────────────────────────────────────────

class CameraContextProjector(nn.Module):
    """Projects δ = (f, pitch, roll, h) to a single context token (B, 1, D)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor, cam_start: int = 12) -> Tensor:
        """x : (B, T, D_in)  →  (B, 1, D)"""
        cam = x[..., cam_start:cam_start + 4].mean(dim=1, keepdim=True)  # (B, 1, 4)
        return self.proj(cam)


# ──────────────────────────────────────────────────────────────────
# 5.  InfraFormerSOTA — main model
# ──────────────────────────────────────────────────────────────────

class InfraFormerSOTA(nn.Module):
    """
    SOTA monocular 3D localization for infrastructure cameras.

    Required cfg attributes:
        d_model, nhead, num_encoder_layers, num_fusion_layers,
        num_decoder_layers, dim_feedforward, dropout,
        enc_in_dim (=17), dec_out_dim (=8),
        enc_seq_len, dec_seq_len, use_tgt_mask,
        social_feat_dim (=8), n_social_neighbors (=5), d_social (=32),
        num_classes (=4),
        bbox2d_mean, bbox2d_std, image_size, focal_length_range,
        focal_feature_idx (=12)
    """

    def __init__(self, cfg):
        super().__init__()

        d_model     = cfg.d_model
        nhead       = cfg.nhead
        dim_ff      = cfg.dim_feedforward
        dropout     = cfg.dropout
        enc_in_dim  = getattr(cfg, 'enc_in_dim', 17)
        dec_seq_len = cfg.dec_seq_len
        dec_out_dim = cfg.dec_out_dim
        num_classes = getattr(cfg, 'num_classes', 4)
        d_social    = getattr(cfg, 'd_social', 32)
        social_dim  = getattr(cfg, 'social_feat_dim', 8)
        n_enc       = cfg.num_encoder_layers
        n_fuse      = getattr(cfg, 'num_fusion_layers', 2)
        n_dec       = cfg.num_decoder_layers

        self.cam_start = getattr(cfg, 'focal_feature_idx', 12)

        # ── Ego encoding ─────────────────────────────────────────────
        self.ego_proj = nn.Linear(enc_in_dim, d_model)
        self.cape     = InfraCAPERayEncoding(d_model, cfg)
        self.ego_enc  = RoPETransformerEncoder(d_model, nhead, dim_ff, n_enc, dropout)

        # ── Social stream ─────────────────────────────────────────────
        self.social_enc   = SocialContextEncoder(social_dim, d_social)
        self.social_gate  = GatedSocialFusion(d_model, d_social)

        # ── Fusion encoder ────────────────────────────────────────────
        self.fusion_enc   = RoPETransformerEncoder(d_model, nhead, dim_ff, n_fuse, dropout)

        # ── Camera context cross-attention ────────────────────────────
        self.cam_proj  = CameraContextProjector(d_model)
        self.cam_cross = CrossAttentionLayer(d_model, nhead, dropout)

        # ── Iterative decoder ─────────────────────────────────────────
        self.decoder = IterativeDecoder(
            d_model, nhead, dim_ff, n_dec, dropout,
            dec_seq_len, dec_out_dim, num_classes,
        )

        # Optional causal mask for decoder
        self.register_buffer(
            'tgt_mask',
            nn.Transformer.generate_square_subsequent_mask(dec_seq_len)
        )
        self.use_tgt_mask = getattr(cfg, 'use_tgt_mask', False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ego_src:      Tensor,
        social_src:   Optional[Tensor] = None,
        social_mask:  Optional[Tensor] = None,
        class_tokens: Optional[Tensor] = None,
    ) -> tuple[Tensor, list]:
        """
        ego_src      : (B, T, 17)      — [d | d̃ | δ | occ]
        social_src   : (B, T, 5, 8)    — neighbour social features (optional)
        social_mask  : (B, T, 5)       — 1=valid neighbour, 0=empty (optional)
        class_tokens : (B,) int        — 0=car,1=bus,2=pedestrian,3=cyclist

        Returns:
          main_pred  : (B, S, 8)
          aux_preds  : list[(B, S, 8)] — intermediate decoder outputs
        """
        # ── Ego encoding ─────────────────────────────────────────────
        ego_emb = self.ego_proj(ego_src) + self.cape(ego_src)     # (B, T, D)
        ego_enc = self.ego_enc(ego_emb)                           # (B, T, D)

        # ── Social gated fusion ───────────────────────────────────────
        if social_src is not None:
            social_ctx = self.social_enc(social_src, social_mask)  # (B, T, D_s)
            fused      = self.social_gate(ego_enc, social_ctx)     # (B, T, D)
        else:
            fused = ego_enc

        # ── Fusion encoder ────────────────────────────────────────────
        memory = self.fusion_enc(fused)                           # (B, T, D)

        # ── Camera context cross-attention ────────────────────────────
        cam_ctx = self.cam_proj(ego_src, self.cam_start)          # (B, 1, D)
        memory  = self.cam_cross(memory, cam_ctx)                 # (B, T, D)

        # ── Iterative decoder ─────────────────────────────────────────
        geo_last = memory[:, -1, :]                               # (B, D)
        tgt_mask = self.tgt_mask if self.use_tgt_mask else None
        main_pred, aux_preds = self.decoder(
            memory, geo_last, class_tokens, tgt_mask
        )
        return main_pred, aux_preds


# ──────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    class _Cfg:
        d_model            = 128
        nhead              = 8
        num_encoder_layers = 4
        num_fusion_layers  = 2
        num_decoder_layers = 4
        dim_feedforward    = 512
        dropout            = 0.1
        enc_in_dim         = 17
        dec_out_dim        = 8
        enc_seq_len        = 12
        dec_seq_len        = 4
        use_tgt_mask       = False
        num_classes        = 4
        social_feat_dim    = 8
        n_social_neighbors = 5
        d_social           = 32
        bbox2d_mean        = [960.0, 540.0]
        bbox2d_std         = [640.0, 360.0]
        image_size         = (1920, 1080)
        focal_length_range = [400.0, 1000.0]
        focal_feature_idx  = 12

    cfg    = _Cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = InfraFormerSOTA(cfg).to(device)

    B, T = 8, cfg.enc_seq_len
    ego  = torch.randn(B, T, 17, device=device)
    soc  = torch.randn(B, T, 5, 8, device=device)
    mask = torch.ones(B, T, 5, device=device)
    cls  = torch.randint(0, 4, (B,), device=device)

    main, aux = model(ego, soc, mask, cls)
    assert main.shape == (B, cfg.dec_seq_len, 8), f"Bad main shape: {main.shape}"
    loss = main.mean() + sum(a.mean() for a in aux)
    loss.backward()

    total = sum(p.numel() for p in model.parameters())
    print(f"InfraFormerSOTA — sanity check passed")
    print(f"  main : {main.shape}")
    print(f"  aux  : {len(aux)} × {aux[0].shape if aux else 'none'}")
    print(f"  params: {total:,}")
    for name, mod in [
        ("EgoEncoder", model.ego_enc),
        ("SocialEncoder", model.social_enc),
        ("SocialGate", model.social_gate),
        ("FusionEncoder", model.fusion_enc),
        ("Decoder", model.decoder),
    ]:
        n = sum(p.numel() for p in mod.parameters())
        print(f"    {name:20s}: {n:>8,}  ({100*n/total:.1f}%)")

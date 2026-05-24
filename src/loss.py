import torch
import torch.nn as nn
import torch.nn.functional as F

# Predictions are stored normalized (÷100). Convert to meters before applying
# real-world safe distances so config values stay human-readable.
_PRED_SCALE = 100.0

class MultiAgentTrajectoryLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_pos = config['loss']['lambda_pos']
        self.lambda_ori = config['loss']['lambda_ori']
        self.lambda_size = config['loss']['lambda_size']
        self.lambda_smooth = config['loss']['lambda_smooth']
        self.lambda_collision = config['loss']['lambda_collision']

        self.enable_smooth = config['loss']['enable_smoothness_loss']
        self.enable_collision = config['loss']['enable_collision_loss']

        self.safe_dist_ped = config['loss']['safe_distance_ped']
        self.safe_dist_veh = config['loss']['safe_distance_veh']

    def forward(self, pred, target, agent_mask, warmup_factor=1.0):
        """
        pred:       (B, N, P, 8)
        target:     (B, N, P, 8)
        agent_mask: (B, N)
        warmup_factor: scalar in [0, 1] to scale auxiliary losses during warmup
        """
        B, N, P, _ = pred.shape

        # Expand agent_mask to (B, N, P, 1) for broadcasting
        mask = agent_mask.unsqueeze(-1).unsqueeze(-1)
        num_valid = agent_mask.sum() * P + 1e-6

        # 1. Position Loss (L2)
        pos_loss = F.mse_loss(pred[..., 0:3] * mask, target[..., 0:3] * mask, reduction='sum')
        pos_loss = pos_loss / num_valid

        # 2. Size Loss (SmoothL1)
        size_loss = F.smooth_l1_loss(pred[..., 3:6] * mask, target[..., 3:6] * mask, reduction='sum')
        size_loss = size_loss / num_valid

        # 3. Orientation Loss (Cosine: 1 - cos·cos - sin·sin)
        cos_sim = (pred[..., 6:8] * target[..., 6:8]).sum(dim=-1)
        ori_loss = ((1.0 - cos_sim) * agent_mask.unsqueeze(-1)).sum() / num_valid

        total_loss = (self.lambda_pos * pos_loss +
                      self.lambda_ori * ori_loss +
                      self.lambda_size * size_loss)

        # 4. Smoothness Loss — optional, scaled by warmup_factor
        if self.enable_smooth and P >= 3 and warmup_factor > 0:
            # Apply agent mask before summing so padded agents don't contribute
            accel = pred[..., 2:, 0:3] - 2 * pred[..., 1:-1, 0:3] + pred[..., 0:-2, 0:3]
            accel_mask = agent_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            smooth_loss = ((accel ** 2) * accel_mask).sum() / (num_valid * 3)
            total_loss = total_loss + warmup_factor * self.lambda_smooth * smooth_loss

        # 5. Collision Loss — optional, vectorized over P, scaled by warmup_factor
        # Predictions are normalized (÷100); multiply by _PRED_SCALE to get meters
        # so that safe_dist_veh/ped config values (in meters) are applied correctly.
        if self.enable_collision and warmup_factor > 0:
            pred_pos_m = pred[..., 0:2] * _PRED_SCALE  # (B, N, P, 2) in meters

            # (B, P, N, 2) → cdist gives (B, P, N, N)
            pred_pos_t = pred_pos_m.permute(0, 2, 1, 3)
            dist = torch.cdist(pred_pos_t, pred_pos_t)  # (B, P, N, N)

            # Pairwise valid-agent mask, no self-pairs: (B, 1, N, N)
            m2 = agent_mask.unsqueeze(1) * agent_mask.unsqueeze(2)  # (B, N, N)
            diag = torch.eye(N, device=pred.device).unsqueeze(0)
            m2 = (m2 * (1.0 - diag)).unsqueeze(1)  # (B, 1, N, N)

            penalty = torch.clamp(self.safe_dist_veh - dist, min=0.0) ** 2  # (B, P, N, N)
            count = m2.sum() * P + 1e-6
            collision_loss = (penalty * m2).sum() / count
            total_loss = total_loss + warmup_factor * self.lambda_collision * collision_loss

        return {
            'total': total_loss,
            'pos': pos_loss,
            'ori': ori_loss,
            'size': size_loss,
        }

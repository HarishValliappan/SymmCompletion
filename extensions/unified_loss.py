"""
Unified Loss Function for Symm3dTriV9

Implements all 5 loss components:
1. Multi-stage Chamfer Distance (hybrid L1+L2 from V7)
2. Geometric symmetry loss
3. Feature symmetry loss (NEWLY IMPLEMENTED - fixes V8 issue #4)
4. Attention symmetry loss (from V7)
5. Diffusion noise prediction loss

Key fixes:
- All 3 stages (coarse, mid, fine) computed (V8 only had coarse)
- lambda_sym_feat properly implemented (V8 had it defined but unused)
- Adaptive weighting (early epoch → diffusion focus, late epoch → detail focus)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


class UnifiedLoss(nn.Module):
    """
    V9 Unified Loss with adaptive weighting
    """

    def __init__(self, config):
        super().__init__()

        # Loss weights from config
        loss_cfg = config.get('loss', {})
        self.lambda_cd = loss_cfg.get('lambda_cd', 1.0)
        self.lambda_sym_geo = loss_cfg.get('lambda_sym_geo', 0.1)
        self.lambda_sym_feat = loss_cfg.get('lambda_sym_feat', 0.05)  # NOW USED!
        self.lambda_sym_attn = loss_cfg.get('lambda_sym_attn', 0.02)
        self.lambda_diffusion = loss_cfg.get('lambda_diffusion', 0.5)

        # Stage weights for multi-stage Chamfer Distance
        self.stage_weights = loss_cfg.get('stage_weights', [0.4, 0.7, 1.0])

        # Hybrid Chamfer Distance (V7 feature)
        self.cd_l1 = ChamferDistanceL1()
        self.cd_l2 = ChamferDistanceL2()
        self.cd_alpha = loss_cfg.get('cd_alpha', 0.7)  # Mix: 0.7*L1 + 0.3*L2

        # Adaptive weighting warmup
        self.warmup_epochs = config.get('scheduler', {}).get('warmup_epoch', 20)

    def forward(self, outputs, ground_truth, epoch=0):
        """
        Compute all loss components

        Args:
            outputs: Dict from model forward with:
                - coarse: (B, N_coarse, 3)
                - mid: (B, N_mid, 3) or None
                - fine: (B, N_fine, 3)
                - symmetry_points: (B, N_sym, 3)
                - symmetry_features: (B, C, N_sym)
                - noise_pred: (B, 3, C, H, W)
                - noise_gt: (B, 3, C, H, W)
                - pred_triplanes: (B, 3, C, H, W)
                - target_triplanes: (B, 3, C, H, W)
            ground_truth: (B, N_gt, 3)
            epoch: Current epoch for adaptive weighting

        Returns:
            loss_total: Total loss scalar
            loss_dict: Dict with all loss components for logging
        """
        loss_dict = {}

        # Adaptive alpha: ramp up symmetry losses over training
        alpha = min(1.0, max(0.0, (epoch - self.warmup_epochs) / 50.0))

        # ================================================================
        # 1. Multi-Stage Chamfer Distance (HYBRID L1+L2 from V7)
        # ================================================================
        loss_cd_total = 0
        stage_names = ['coarse', 'mid', 'fine']

        for stage_name, stage_weight in zip(stage_names, self.stage_weights):
            if stage_name in outputs and outputs[stage_name] is not None:
                pred = outputs[stage_name]

                # Hybrid Chamfer: 0.7 * L1 + 0.3 * L2
                cd_l1 = self.cd_l1(pred, ground_truth)
                cd_l2 = self.cd_l2(pred, ground_truth)
                loss_cd = self.cd_alpha * cd_l1 + (1 - self.cd_alpha) * cd_l2

                loss_dict[f'loss_cd_{stage_name}'] = loss_cd.item()
                loss_cd_total += stage_weight * loss_cd

        loss_dict['loss_cd_total'] = loss_cd_total.item()

        # ================================================================
        # 2. Geometric Symmetry Loss (on points)
        # ================================================================
        loss_sym_geo = torch.tensor(0.0, device=ground_truth.device)

        if 'symmetry_points' in outputs and outputs['symmetry_points'] is not None:
            sym_points = outputs['symmetry_points']  # (B, N, 3)

            # Reflect along X-axis (assume X is the primary symmetry axis)
            sym_points_reflected = sym_points.clone()
            sym_points_reflected[:, :, 0] = -sym_points_reflected[:, :, 0]

            # MSE between points and reflected points
            loss_sym_geo = F.mse_loss(sym_points, sym_points_reflected)

            loss_dict['loss_sym_geo'] = loss_sym_geo.item()

        # ================================================================
        # 3. Feature Symmetry Loss (NEWLY IMPLEMENTED - fixes V8 issue #4)
        # ================================================================
        loss_sym_feat = torch.tensor(0.0, device=ground_truth.device)

        if 'symmetry_features' in outputs and outputs['symmetry_features'] is not None:
            sym_feat = outputs['symmetry_features']  # (B, C, N)

            # Reflect features spatially (flip along feature dimension)
            sym_feat_reflected = torch.flip(sym_feat, dims=[2])

            # MSE between features and reflected features
            loss_sym_feat = F.mse_loss(sym_feat, sym_feat_reflected)

            loss_dict['loss_sym_feat'] = loss_sym_feat.item()

        # ================================================================
        # 4. Attention Symmetry Loss (from V7)
        # ================================================================
        loss_sym_attn = torch.tensor(0.0, device=ground_truth.device)

        if 'attention_maps' in outputs and outputs['attention_maps'] is not None:
            attn_maps = outputs['attention_maps']  # (B, H, W) or list

            if isinstance(attn_maps, list):
                # Average over multiple attention layers
                loss_sym_attn_sum = 0
                for attn_map in attn_maps:
                    attn_reflected = torch.flip(attn_map, dims=[2])
                    loss_sym_attn_sum += F.mse_loss(attn_map, attn_reflected)
                loss_sym_attn = loss_sym_attn_sum / len(attn_maps)
            else:
                attn_reflected = torch.flip(attn_maps, dims=[2])
                loss_sym_attn = F.mse_loss(attn_maps, attn_reflected)

            loss_dict['loss_sym_attn'] = loss_sym_attn.item()

        # ================================================================
        # 5. Diffusion Loss (noise prediction MSE)
        # ================================================================
        loss_diffusion = torch.tensor(0.0, device=ground_truth.device)

        if 'noise_pred' in outputs and 'noise_gt' in outputs:
            noise_pred = outputs['noise_pred']
            noise_gt = outputs['noise_gt']

            # MSE on noise prediction
            loss_diffusion = F.mse_loss(noise_pred, noise_gt)

            loss_dict['loss_diffusion'] = loss_diffusion.item()

        # ================================================================
        # Total Loss with Adaptive Weighting
        # ================================================================
        # Early epochs: focus on diffusion and coarse structure
        # Later epochs: emphasize fine details and symmetry

        loss_total = (
            self.lambda_cd * loss_cd_total +                         # Main reconstruction
            self.lambda_sym_geo * loss_sym_geo +                     # Geometric symmetry
            self.lambda_sym_feat * alpha * loss_sym_feat +           # Feature symmetry (ramp up)
            self.lambda_sym_attn * alpha * loss_sym_attn +           # Attention symmetry (ramp up)
            self.lambda_diffusion * (1 - 0.3 * alpha) * loss_diffusion  # Diffusion (ramp down)
        )

        loss_dict['loss_total'] = loss_total.item()
        loss_dict['alpha'] = alpha

        return loss_total, loss_dict


# ================================================================
# Auxiliary Loss Components
# ================================================================

class SymmetryPlaneLoss(nn.Module):
    """
    Optional: Loss on explicit symmetry plane prediction
    Can be used if the model predicts symmetry plane parameters
    """

    def __init__(self):
        super().__init__()

    def forward(self, points, sym_normal, sym_distance):
        """
        Args:
            points: (B, N, 3)
            sym_normal: (B, 3) - symmetry plane normal
            sym_distance: (B, 1) - distance from origin

        Returns:
            loss: Scalar loss encouraging points to lie on symmetric sides
        """
        # Reflect points across plane
        normal = sym_normal.unsqueeze(1)  # (B, 1, 3)
        d = sym_distance.unsqueeze(1)  # (B, 1, 1)

        # Signed distance from point to plane
        dist = torch.sum(points * normal, dim=2, keepdim=True) + d  # (B, N, 1)

        # Reflect points
        points_reflected = points - 2 * dist * normal

        # Loss: MSE between points and reflected points
        loss = F.mse_loss(points, points_reflected)

        return loss


class TriPlaneLoss(nn.Module):
    """
    Optional: Direct supervision on triplane representations
    Can be used if we have ground truth triplanes
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_triplanes, target_triplanes):
        """
        Args:
            pred_triplanes: (B, 3, C, H, W)
            target_triplanes: (B, 3, C, H, W)

        Returns:
            loss: L1 loss on triplane features
        """
        loss = F.l1_loss(pred_triplanes, target_triplanes)
        return loss


# ================================================================
# Metrics for Validation
# ================================================================

class Metrics:
    """
    Evaluation metrics: F-Score, Chamfer Distance
    """

    @staticmethod
    def f_score(pred, gt, threshold=0.01):
        """
        F-Score @ threshold

        Args:
            pred: (B, N_pred, 3)
            gt: (B, N_gt, 3)
            threshold: Distance threshold (default 1% of bounding box diagonal)

        Returns:
            f_score: (B,) F-Score for each sample
        """
        B = pred.shape[0]
        f_scores = []

        for i in range(B):
            pred_i = pred[i]  # (N_pred, 3)
            gt_i = gt[i]  # (N_gt, 3)

            # Pairwise distances
            dist_pred_to_gt = torch.cdist(pred_i, gt_i)  # (N_pred, N_gt)
            dist_gt_to_pred = dist_pred_to_gt.t()  # (N_gt, N_pred)

            # Precision: % of pred points within threshold of any GT point
            min_dist_pred = dist_pred_to_gt.min(dim=1)[0]  # (N_pred,)
            precision = (min_dist_pred < threshold).float().mean()

            # Recall: % of GT points within threshold of any pred point
            min_dist_gt = dist_gt_to_pred.min(dim=1)[0]  # (N_gt,)
            recall = (min_dist_gt < threshold).float().mean()

            # F-Score
            if precision + recall > 0:
                f_score = 2 * precision * recall / (precision + recall)
            else:
                f_score = torch.tensor(0.0, device=pred.device)

            f_scores.append(f_score)

        return torch.stack(f_scores)

    @staticmethod
    def chamfer_distance_l1(pred, gt):
        """Chamfer Distance L1"""
        cd_l1 = ChamferDistanceL1()
        return cd_l1(pred, gt)

    @staticmethod
    def chamfer_distance_l2(pred, gt):
        """Chamfer Distance L2"""
        cd_l2 = ChamferDistanceL2()
        return cd_l2(pred, gt)

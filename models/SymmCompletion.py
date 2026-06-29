import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS


# ──────────────────────────────────────────────────────────────────────
# Shared modules (unchanged from original SymmCompletion)
# ──────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        _, NK, _ = y.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossFormer(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x


class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4,
                                         qkv_bias=False, qk_scale=None,
                                         attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4,
                                         qkv_bias=False, qk_scale=None,
                                         attn_drop=0.0, proj_drop=0.0)

    def forward(self, feat_x, feat_y):
        feat = self.corssformer_1(feat_x, feat_y)
        feat = self.corssformer_2(feat, feat)
        return feat


class SGFormer(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)

        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)

        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        # partial part awareness
        feat_p = self.fusion_1(feat, partial_feat)
        # symmetric part awareness
        feat_s = self.fusion_2(feat, symmetry_feat)
        # fusion feature
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        # self attention for upsampling
        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)
        pcd_up = (coarse.transpose(2, 1).contiguous()
                  .unsqueeze(dim=2)
                  .repeat(1, 1, self.up_factor, 1)
                  .view(b, -1, 3) + offset)
        return pcd_up


class local_encoder(nn.Module):
    def __init__(self, out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self, input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, input)
        return feat


# ──────────────────────────────────────────────────────────────────────
# Tri-Plane Symmetry Field  (replaces LSTNet)
# ──────────────────────────────────────────────────────────────────────

class _PlaneUpsampler(nn.Module):
    """
    CNN-based plane generator: projects a global feature to a small 4×4
    spatial grid, then progressively upsamples with ConvTranspose2d.
    This is *much* more parameter-efficient than a flat Linear projection.

    Parameter comparison for plane_dim=128, plane_res=32:
        Linear approach:  ~67 M params per plane  (512 → 131072)
        CNN approach:     ~1 M params per plane   (3 upsample stages)
    """
    def __init__(self, global_dim: int, plane_dim: int, plane_res: int):
        super().__init__()
        self.plane_dim = plane_dim
        self.init_res = 4  # start from 4×4

        # Linear projection to small spatial grid: global_dim → plane_dim*4*4
        self.fc = nn.Sequential(
            nn.Linear(global_dim, plane_dim * self.init_res * self.init_res),
            nn.LeakyReLU(0.2),
        )

        # Build progressive upsampling stages: 4→8→16→32 (or fewer)
        layers = []
        cur_res = self.init_res
        while cur_res < plane_res:
            layers.extend([
                nn.ConvTranspose2d(plane_dim, plane_dim, kernel_size=4,
                                   stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ])
            cur_res *= 2
        self.upsample = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, gf):
        """gf: (B, global_dim) → plane: (B, plane_dim, H, W)"""
        B = gf.shape[0]
        x = self.fc(gf).view(B, self.plane_dim, self.init_res, self.init_res)
        return self.upsample(x)


class TriPlaneGenerator(nn.Module):
    """
    Takes a global feature vector and produces three orthogonal feature
    planes  T_xy, T_yz, T_zx  each of shape  (B, C, H, W).

    Uses CNN-based upsampling instead of flat Linear layers to keep
    total parameter count manageable (~3 M vs ~200 M).
    """
    def __init__(self, global_dim: int = 512, plane_dim: int = 128,
                 plane_res: int = 32):
        super().__init__()
        self.gen_xy = _PlaneUpsampler(global_dim, plane_dim, plane_res)
        self.gen_yz = _PlaneUpsampler(global_dim, plane_dim, plane_res)
        self.gen_zx = _PlaneUpsampler(global_dim, plane_dim, plane_res)

    def forward(self, gf):
        """
        Args:
            gf: (B, global_dim)   – global feature vector
        Returns:
            T_xy, T_yz, T_zx  each (B, plane_dim, H, W)
        """
        return self.gen_xy(gf), self.gen_yz(gf), self.gen_zx(gf)


class TriPlaneSampler(nn.Module):
    """
    Given three feature planes and 3-D query points, bilinearly sample each
    plane and aggregate features by summation.

    Point coordinates are expected in [-1, 1] (normalised).
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _normalise_coords(pts: torch.Tensor) -> torch.Tensor:
        """
        Normalise point coordinates to [-1, 1] based on per-batch min/max
        so that grid_sample can be used.
        Args:
            pts: (B, N, 3)
        Returns:
            pts_norm: (B, N, 3) in [-1, 1]
        """
        mins = pts.min(dim=1, keepdim=True)[0]   # (B, 1, 3)
        maxs = pts.max(dim=1, keepdim=True)[0]   # (B, 1, 3)
        span = (maxs - mins).clamp(min=1e-6)
        return 2.0 * (pts - mins) / span - 1.0

    def forward(self, T_xy, T_yz, T_zx, pts):
        """
        Args:
            T_xy, T_yz, T_zx: (B, C, H, W)  feature planes
            pts:               (B, N, 3)      query points (raw coords)
        Returns:
            f_tri: (B, C, N)  aggregated tri-plane features
        """
        pts_n = self._normalise_coords(pts)  # (B, N, 3) in [-1,1]
        x, y, z = pts_n[..., 0], pts_n[..., 1], pts_n[..., 2]

        # build 2-D grid for each projection   (B, N, 1, 2)
        grid_xy = torch.stack([x, y], dim=-1).unsqueeze(2)   # project (x,y)
        grid_yz = torch.stack([y, z], dim=-1).unsqueeze(2)   # project (y,z)
        grid_zx = torch.stack([z, x], dim=-1).unsqueeze(2)   # project (z,x)

        # grid_sample wants (B, C, H_in, W_in) and grid (B, H_out, W_out, 2)
        # We treat N query pts as H_out=N, W_out=1
        f_xy = F.grid_sample(T_xy, grid_xy, align_corners=True, mode='bilinear',
                             padding_mode='border').squeeze(-1)  # (B, C, N)
        f_yz = F.grid_sample(T_yz, grid_yz, align_corners=True, mode='bilinear',
                             padding_mode='border').squeeze(-1)
        f_zx = F.grid_sample(T_zx, grid_zx, align_corners=True, mode='bilinear',
                             padding_mode='border').squeeze(-1)

        f_tri = f_xy + f_yz + f_zx  # (B, C, N)
        return f_tri


class SymmetryFlowField(nn.Module):
    """
    MLP that takes per-point tri-plane features and predicts a 3-D
    displacement  Δp  so that  p_sym = p + Δp.
    """
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 3, 1),
        )

    def forward(self, f_tri):
        """
        Args:
            f_tri: (B, C, N)
        Returns:
            delta: (B, 3, N)   displacement vectors
        """
        return self.mlp(f_tri)


class TriPlaneSymmetryNet(nn.Module):
    """
    Replacement for LSTNet.
    1. Encode partial cloud  →  key-points + per-point features + global feat
    2. Generate three orthogonal feature planes from global feat
    3. Sample tri-plane features at key-point locations
    4. Predict symmetry displacement field  →  symmetric points
    """
    def __init__(self, out_dim: int = 512, plane_dim: int = 128,
                 plane_res: int = 32):
        super().__init__()
        # ---- point feature encoder (same backbone as original LSTNet) ----
        self.sa_module_1 = PointNet_SA_Module_KNN(
            512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])

        # ---- tri-plane generation from global feature ----
        self.plane_generator = TriPlaneGenerator(
            global_dim=out_dim, plane_dim=plane_dim, plane_res=plane_res)

        # ---- tri-plane feature sampling ----
        self.sampler = TriPlaneSampler()

        # ---- symmetry displacement field ----
        self.flow_field = SymmetryFlowField(in_dim=plane_dim)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, 3, N)  – partial point cloud
        Returns:
            coarse:           (B, 3, 2*npoint)  – concatenation of partial key-pts + predicted symmetric pts
            symmetry_points:  (B, 3, npoint)
            keyfeatures:      (B, 128, npoint)  – encoder features (passed to SGFormer)
        """
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        # ── 1. encode  ──────────────────────────────────────────────
        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)
        # keypoints:   (B, 3, 512)
        # keyfeatures: (B, 128, 512)
        keyfeatures = self.transformer_1(keyfeatures, keypoints)  # (B,128,512)

        feat = self.expanding(keyfeatures)          # (B, out_dim, 512)
        gf = feat.max(dim=2)[0]                     # (B, out_dim)  global feat

        # ── 2. tri-plane generation ──────────────────────────────────
        T_xy, T_yz, T_zx = self.plane_generator(gf)

        # ── 3. sample tri-plane features at keypoint locations ───────
        kp_t = keypoints.transpose(2, 1).contiguous()  # (B, 512, 3)
        f_tri = self.sampler(T_xy, T_yz, T_zx, kp_t)  # (B, plane_dim, 512)

        # ── 4. predict displacement field ────────────────────────────
        delta = self.flow_field(f_tri)  # (B, 3, 512)

        symmetry_points = keypoints + delta             # (B, 3, 512)
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)  # (B, 3, 1024)

        return coarse, symmetry_points, keyfeatures


# ──────────────────────────────────────────────────────────────────────
# Main model  (TriPlane variant)
# ──────────────────────────────────────────────────────────────────────

@MODELS.register_module()
class SymmCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]

        # ── Tri-Plane Symmetry Net (replaces LSTNet) ────────────────
        plane_dim = getattr(config, 'plane_dim', 128)
        plane_res = getattr(config, 'plane_res', 32)
        self.triplane_net = TriPlaneSymmetryNet(
            out_dim=512, plane_dim=plane_dim, plane_res=plane_res)

        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.include_input
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, rets, gt):
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total, loss_list[0], loss_list[-1], loss_list[0], loss_list[-1]

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3)
        """
        # ── coarse stage: tri-plane symmetry ─────────────────────────
        coarse, symmetry_points, keyfeatures = self.triplane_net(
            point_cloud.transpose(2, 1).contiguous())

        # ── feature extraction for refinement ────────────────────────
        feat_symmetry = self.local_encoder(symmetry_points)   # (B,128,512)
        feat_partial  = keyfeatures                           # (B,128,512)

        # ── refinement stages (SGFormer, unchanged) ──────────────────
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2, 1).contiguous(),
                                feat_symmetry, feat_partial)

        if self.include_input:
            fine2 = torch.cat([fine2, point_cloud], dim=1).contiguous()

        rets = [coarse.transpose(2, 1).contiguous(), fine1, fine2]
        self.pred_dense_point = rets[-1]

        return rets

"""
Symm3dTri — Detailed Architecture & Diagnostic Report Generator
Produces a multi-page PDF using matplotlib (no reportlab needed).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_PATH = "/home/msai/harihara011/Symm3dTri/SymmCompletion/Symm3dTri_Architecture_Diagnostic_Report.pdf"

# ── colour palette ────────────────────────────────────────────────────────
C = dict(
    title   = '#1a237e',
    head    = '#283593',
    sub     = '#3949ab',
    block   = '#e8eaf6',
    block2  = '#fce4ec',
    block3  = '#e8f5e9',
    block4  = '#fff8e1',
    arrow   = '#5c6bc0',
    bad     = '#c62828',
    good    = '#2e7d32',
    warn    = '#e65100',
    text    = '#212121',
    muted   = '#546e7a',
    border  = '#9fa8da',
)

def new_fig(landscape=True):
    if landscape:
        fig = plt.figure(figsize=(17, 11))
    else:
        fig = plt.figure(figsize=(11, 17))
    fig.patch.set_facecolor('white')
    return fig

def title_bar(fig, title, subtitle=''):
    ax = fig.add_axes([0, 0.93, 1, 0.07])
    ax.set_facecolor(C['title'])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.62, title, ha='center', va='center',
            fontsize=18, fontweight='bold', color='white')
    if subtitle:
        ax.text(0.5, 0.18, subtitle, ha='center', va='center',
                fontsize=10, color='#c5cae9')

def section_box(ax, x, y, w, h, label, color=None, fontsize=9, labelcolor='white'):
    color = color or C['sub']
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.01",
                       facecolor=color, edgecolor=C['border'], linewidth=1.2)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=labelcolor,
            wrap=True)

def arrow(ax, x0, y0, x1, y1, color=None):
    color = color or C['arrow']
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

def hline(ax, y, color='#9e9e9e', lw=0.8):
    ax.axhline(y, color=color, lw=lw, linestyle='--')

def bullet(ax, x, y, text, fontsize=9, color=None, indent=0):
    color = color or C['text']
    ax.text(x + indent, y, u'\u2022 ' + text, fontsize=fontsize,
            color=color, va='top', wrap=True)

def math_text(ax, x, y, text, fontsize=10, color=None):
    color = color or C['text']
    ax.text(x, y, text, fontsize=fontsize, color=color, va='top',
            fontfamily='monospace')

def para(ax, x, y, text, fontsize=9, color=None, style='normal'):
    color = color or C['text']
    ax.text(x, y, text, fontsize=fontsize, color=color, va='top',
            style=style, wrap=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Cover / Overview
# ══════════════════════════════════════════════════════════════════════════
def page_cover(pdf):
    fig = new_fig()
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.88])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor('white')

    # Header
    ax.add_patch(FancyBboxPatch((0, 80), 100, 18,
                 boxstyle="round,pad=0.5", facecolor=C['title'],
                 edgecolor='none'))
    ax.text(50, 92, 'Symm3dTri', ha='center', va='center',
            fontsize=32, fontweight='bold', color='white')
    ax.text(50, 84, 'Detailed Architecture, Mathematical Formulation & Diagnostic Analysis',
            ha='center', va='center', fontsize=13, color='#c5cae9')

    # Summary boxes
    boxes = [
        ('Dataset',    'PCN (ShapeNet)',       C['block'],  10, 65, 20, 12),
        ('Metric',     'Chamfer Distance L1',  C['block'],  33, 65, 20, 12),
        ('Best CDL1',  '6.632',               C['block3'], 56, 65, 20, 12),
        ('F-Score',    '0.832',               C['block3'], 79, 65, 20, 12),
    ]
    for label, val, col, x, y, w, h in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.5", facecolor=col,
                     edgecolor=C['border'], linewidth=1.5))
        ax.text(x+w/2, y+h*0.68, label, ha='center', va='center',
                fontsize=9,  color=C['muted'], fontweight='bold')
        ax.text(x+w/2, y+h*0.28, val,   ha='center', va='center',
                fontsize=11, color=C['title'], fontweight='bold')

    # Architecture pipeline overview
    ax.text(2, 62, 'Architecture Pipeline Overview', fontsize=12,
            fontweight='bold', color=C['head'])
    hline(ax, 61)

    stages = [
        ('Partial\nInput\n(B,N,3)',    C['block4'],  3),
        ('PointNet\nSA-KNN\n(Encoder)',C['block'],  20),
        ('TriPlane\nGenerator',        C['block2'], 37),
        ('TriPlane\nSampler +\nFlow Field', C['block2'], 54),
        ('Coarse\nOutput\n(B,1024,3)',  C['block3'], 71),
        ('SGFormer\n×2\n(Refine)',     C['block'],  88),
    ]
    for label, col, x in stages:
        section_box(ax, x, 43, 14, 16, label, color=col,
                    fontsize=8, labelcolor=C['text'])

    for i in range(len(stages)-1):
        arrow(ax, stages[i][2]+14, 51, stages[i+1][2], 51)

    ax.text(50, 40, 'Final Dense Output  (B, 16384, 3)',
            ha='center', fontsize=10, color=C['good'], fontweight='bold')

    # Key numbers
    ax.text(2, 37, 'Key Hyperparameters', fontsize=12,
            fontweight='bold', color=C['head'])
    hline(ax, 36)

    params = [
        ('Keypoints (SA output)', '512'),
        ('Plane dim', '128'),
        ('Plane resolution', '32 x 32'),
        ('up_factors', '[2, 8]  →  1024 → 16384'),
        ('Global feature dim', '512'),
        ('Optimizer', 'AdamW  lr=2e-4'),
        ('Scheduler', 'WarmUpCosLR  (warm=20, max=120)'),
        ('Batch size', '32'),
    ]
    cols = 4
    for i, (k, v) in enumerate(params):
        col_x = 2 + (i % cols) * 24
        row_y = 32 - (i // cols) * 8
        ax.add_patch(FancyBboxPatch((col_x, row_y-6), 22, 7,
                     boxstyle="round,pad=0.3", facecolor=C['block'],
                     edgecolor=C['border'], linewidth=1))
        ax.text(col_x+11, row_y-1.5, k,  ha='center', fontsize=7.5,
                color=C['muted'], fontweight='bold')
        ax.text(col_x+11, row_y-4.5, v,  ha='center', fontsize=8.5,
                color=C['title'], fontweight='bold')

    ax.text(2, 14, 'Contents of this Report', fontsize=12,
            fontweight='bold', color=C['head'])
    hline(ax, 13)
    contents = [
        'Page 1  —  Cover & Overview',
        'Page 2  —  Full Architecture Diagram (component-level)',
        'Page 3  —  TriPlaneSymmetryNet: Mathematics & Detail',
        'Page 4  —  SGFormer (Refinement Stages): Mathematics & Detail',
        'Page 5  —  Training Objective & Loss Functions',
        'Page 6  —  Diagnostic Results: Stage-wise CDL1 Breakdown',
        'Page 7  —  Diagnostic Results: Symmetry Branch Analysis',
        'Page 8  —  Per-Category Analysis & Root Cause',
        'Page 9  —  Conclusions & Recommended Fixes',
    ]
    for i, c in enumerate(contents):
        ax.text(4, 11 - i*1.3, c, fontsize=9, color=C['text'], va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Full Architecture Diagram
# ══════════════════════════════════════════════════════════════════════════
def page_architecture(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 2 — Full Architecture Diagram', 'Symm3dTri Component-Level View')
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # ── Left column: TriPlaneSymmetryNet ──────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.5, 2), 44, 96,
                 boxstyle="round,pad=0.5", facecolor='#f3f4fe',
                 edgecolor=C['sub'], linewidth=2, linestyle='--'))
    ax.text(22.5, 96.5, 'TriPlaneSymmetryNet  (Coarse Stage)',
            ha='center', fontsize=10, fontweight='bold', color=C['sub'])

    # Partial input
    section_box(ax, 8, 88, 28, 6, 'Partial Input\n(B, N, 3)', C['block4'], labelcolor=C['text'])

    # SA Module
    section_box(ax, 2, 76, 40, 9,
                'PointNet_SA_Module_KNN\nnpoint=512, k=16, MLP[3→64→128]\nOutput: keypoints(B,3,512)  keyfeatures(B,128,512)',
                C['block'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 88, 22, 85)

    # Transformer1
    section_box(ax, 2, 67, 40, 7,
                'Positional Transformer\n(in=128, dim=64, k_knn=16)\nkeyfeatures: (B, 128, 512)',
                C['block'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 76, 22, 74)

    # Expanding MLP
    section_box(ax, 2, 58, 40, 7,
                'MLP_CONV  Expanding\n128 → 256 → 512\nOutput feat: (B, 512, 512)',
                C['block'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 67, 22, 65)

    # Global feature
    section_box(ax, 8, 49, 28, 7,
                'Global Max Pool\ngf = max(feat, dim=2)\ngf: (B, 512)',
                '#e8eaf6', fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 58, 22, 56)

    # TriPlane Generator
    section_box(ax, 1, 37, 42, 10,
                'TriPlaneGenerator\nshared FC: Linear(512 → 128×16)  →  seed(B,128,4,4)\n'
                'up_xy / up_yz / up_zx: ConvT2d(128,128)×3\n'
                'T_xy, T_yz, T_zx: each (B,128,32,32)',
                C['block2'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 49, 22, 47)

    # TriPlane Sampler
    section_box(ax, 1, 27, 42, 8,
                'TriPlaneSampler\nNormalise keypoints → [-1,1]\n'
                'grid_sample on T_xy, T_yz, T_zx\n'
                'f_tri = f_xy + f_yz + f_zx   (B,128,512)',
                C['block2'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 37, 22, 35)

    # Flow field
    section_box(ax, 2, 18, 40, 7,
                'SymmetryFlowField\nConv1d: 128→256→128→3\n'
                'delta: (B, 3, 512)',
                C['block2'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 27, 22, 25)

    # Sym points + coarse
    section_box(ax, 1, 8, 42, 8,
                'sym_pts = keypoints + delta     (B,3,512)\n'
                'coarse  = cat[sym_pts, keypoints]  (B,3,1024)',
                C['block3'], fontsize=8, labelcolor=C['text'])
    arrow(ax, 22, 18, 22, 16)

    # ── Right column: Refinement ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((46, 2), 53.5, 96,
                 boxstyle="round,pad=0.5", facecolor='#f1f8e9',
                 edgecolor=C['good'], linewidth=2, linestyle='--'))
    ax.text(72.5, 96.5, 'Refinement Stages  (SGFormer × 2)',
            ha='center', fontsize=10, fontweight='bold', color=C['good'])

    # local_encoder
    section_box(ax, 48, 83, 49, 9,
                'local_encoder(sym_pts)\nMLP_CONV 3→64→128  +  MaxPool  +  MLP_CONV 256→128\n'
                'Transformer(128, dim=64)\nfeat_symmetry: (B, 128, 512)',
                C['block'], fontsize=8, labelcolor=C['text'])
    ax.text(49.5, 80, '(feat_partial = keyfeatures from encoder)',
            fontsize=7.5, color=C['muted'], style='italic')

    # SGFormer-1
    ax.add_patch(FancyBboxPatch((47, 42), 51, 37,
                 boxstyle="round,pad=0.3", facecolor='#e8f5e9',
                 edgecolor=C['good'], linewidth=1.5))
    ax.text(72.5, 78, 'SGFormer-1  (up_factor=2)', ha='center',
            fontsize=9, fontweight='bold', color=C['good'])

    section_box(ax, 48, 68, 49, 8,
                'MLP_CONV(3→64→128) + MaxPool + MLP_CONV(256→128)\n'
                'Transformer(128, dim=64)  on coarse pts\n'
                'coarse_feat: (B,128,1024)',
                C['block'], fontsize=7.5, labelcolor=C['text'])

    section_box(ax, 48, 58, 49, 8,
                'expand_dim: feat,partial,symm → (B,256,N)\n'
                'Fusion-1: CrossFormer(coarse_feat ← partial_feat)\n'
                'Fusion-2: CrossFormer(coarse_feat ← symm_feat)\n'
                'feat = cat[feat_p, feat_s]  (B,512,N)',
                C['block'], fontsize=7.5, labelcolor=C['text'])
    arrow(ax, 72, 68, 72, 66)

    section_box(ax, 48, 48, 49, 8,
                'MLP_fusion: Linear(512→512)\n'
                'Fusion-3: CrossFormer(self-attn)\n'
                'FC: Linear 512→512→128→3×up_factor\n'
                'fine1: (B, 2048, 3)',
                C['block3'], fontsize=7.5, labelcolor=C['text'])
    arrow(ax, 72, 58, 72, 56)

    # SGFormer-2
    ax.add_patch(FancyBboxPatch((47, 5), 51, 37,
                 boxstyle="round,pad=0.3", facecolor='#e8f5e9',
                 edgecolor=C['good'], linewidth=1.5))
    ax.text(72.5, 41, 'SGFormer-2  (up_factor=8)', ha='center',
            fontsize=9, fontweight='bold', color=C['good'])

    section_box(ax, 48, 31, 49, 8,
                'Same architecture as SGFormer-1\n'
                'Input: fine1 (B, 3, 2048)\n'
                'Same feat_symmetry & feat_partial reused',
                C['block'], fontsize=7.5, labelcolor=C['text'])
    arrow(ax, 72, 48, 72, 39)

    section_box(ax, 48, 18, 49, 11,
                'FC: Linear 512→512→128→3×8\n'
                'fine2: (B, 16384, 3)\n\n'
                'include_input=False  →  final = fine2',
                C['block3'], fontsize=7.5, labelcolor=C['text'])
    arrow(ax, 72, 31, 72, 29)

    section_box(ax, 48, 8, 49, 8,
                'Output:  [coarse(B,1024,3),  fine1(B,2048,3),  fine2(B,16384,3)]',
                '#2e7d32', fontsize=9, labelcolor='white')
    arrow(ax, 72, 18, 72, 16)

    # Cross arrows from left to right
    arrow(ax, 44, 12, 48, 30,  color=C['arrow'])
    ax.text(45.5, 21, 'coarse', fontsize=7, color=C['arrow'], rotation=55)
    arrow(ax, 44, 67, 48, 76, color='#9c27b0')
    ax.text(44, 71, 'keyfeatures\n(feat_partial)', fontsize=7,
            color='#9c27b0', rotation=30)
    arrow(ax, 44, 12, 48, 83, color='#f44336')
    ax.text(42, 50, 'sym_pts\n→ local_enc', fontsize=7,
            color='#f44336', rotation=90)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — TriPlaneSymmetryNet Mathematics
# ══════════════════════════════════════════════════════════════════════════
def page_triplane_math(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 3 — TriPlaneSymmetryNet: Mathematics & Detail',
              'Coarse Stage: Encoder → TriPlane Generation → Symmetry Flow Field')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # ── Step 1: Encoder ───────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 83), 100, 14,
                 boxstyle="round,pad=0.3", facecolor=C['block'],
                 edgecolor=C['border'], linewidth=1.5))
    ax.text(2, 95.5, 'STEP 1 — Point Cloud Encoder (PointNet_SA_Module_KNN + Transformer)',
            fontsize=11, fontweight='bold', color=C['head'])

    lines = [
        r'Input: P_partial  in  R^(B x N x 3)',
        r'FPS: sample 512 keypoints from N input points using Furthest Point Sampling',
        r'  keypoints  in  R^(B x 3 x 512)',
        r'KNN grouping: for each keypoint, find k=16 nearest neighbors',
        r'  grouped_xyz  in  R^(B x 3 x 512 x 16)   (relative coordinates)',
        r'MLP_CONV per group: [3+3 -> 64 -> 128],  then MaxPool over k neighbors',
        r'  keyfeatures  in  R^(B x 128 x 512)',
        r'Positional Transformer (dim=64, k_knn=16): adds local geometry attention',
        r'  keyfeatures  in  R^(B x 128 x 512)   (refined)',
        r'Expanding MLP_CONV: 128 -> 256 -> 512',
        r'  feat  in  R^(B x 512 x 512)',
        r'Global feature: gf = max(feat, dim=2)   in  R^(B x 512)',
    ]
    for i, l in enumerate(lines):
        ax.text(3, 93 - i*1.1, l, fontsize=8.2, color=C['text'], va='top', fontfamily='monospace')

    # ── Step 2: TriPlane Generator ────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 56), 100, 25,
                 boxstyle="round,pad=0.3", facecolor=C['block2'],
                 edgecolor='#ef9a9a', linewidth=1.5))
    ax.text(2, 79.5, 'STEP 2 — TriPlane Generator',
            fontsize=11, fontweight='bold', color='#b71c1c')

    ax.text(2, 77.5, 'Motivation: Encode spatial symmetry structure in 3 orthogonal feature planes.',
            fontsize=9, color=C['text'], va='top')
    ax.text(2, 75.2, 'Three planes capture different 2-D projections of 3-D space:', fontsize=9,
            color=C['text'], va='top')

    planes = [
        ('T_xy', 'Projects X-Y plane', 'encodes bilateral left-right symmetry'),
        ('T_yz', 'Projects Y-Z plane', 'encodes front-back relationships'),
        ('T_zx', 'Projects Z-X plane', 'encodes up-down relationships'),
    ]
    for i, (name, proj, enc) in enumerate(planes):
        x = 2 + i*33
        ax.add_patch(FancyBboxPatch((x, 68), 30, 5.5,
                     boxstyle="round,pad=0.2", facecolor='white',
                     edgecolor='#ef9a9a', linewidth=1))
        ax.text(x+15, 71.5, name, ha='center', fontsize=10, fontweight='bold', color='#b71c1c')
        ax.text(x+15, 70,   proj, ha='center', fontsize=8,  color=C['muted'])
        ax.text(x+15, 68.5, enc,  ha='center', fontsize=7.5,color=C['text'])

    ax.text(2, 66.5, 'Implementation:', fontsize=9, fontweight='bold', color=C['text'], va='top')
    impl = [
        r'1. Shared FC:   seed = LeakyReLU(Linear(512, 128x16))   seed reshaped -> (B, 128, 4, 4)',
        r'2. Per-plane upsample (3 independent ConvTranspose2d chains):',
        r'   4x4 --(ConvT 128,128,k=4,s=2,p=1)--> 8x8 --(ConvT)--> 16x16 --(ConvT)--> 32x32',
        r'   T_xy, T_yz, T_zx  each in  R^(B x 128 x 32 x 32)',
        r'   Total TriPlaneGenerator params: ~3.4M  (vs 5.5M with 3 independent FCs)',
    ]
    for i, l in enumerate(impl):
        ax.text(3, 64.5 - i*1.4, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    # ── Step 3: TriPlane Sampler ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 34), 100, 20,
                 boxstyle="round,pad=0.3", facecolor='#fce4ec',
                 edgecolor='#e91e63', linewidth=1.5))
    ax.text(2, 52.5, 'STEP 3 — TriPlane Sampler  (Bilinear Grid Sampling)',
            fontsize=11, fontweight='bold', color='#880e4f')

    samp = [
        r'1. Normalise keypoint coordinates to [-1, 1]:',
        r'     p_norm = 2 * (p - min(p)) / (max(p) - min(p)) - 1      p_norm in R^(B x 512 x 3)',
        r'',
        r'2. Project onto each plane to get 2-D sampling grids:',
        r'     grid_xy = stack[x, y]  in R^(B x 512 x 1 x 2)',
        r'     grid_yz = stack[y, z]  in R^(B x 512 x 1 x 2)',
        r'     grid_zx = stack[z, x]  in R^(B x 512 x 1 x 2)',
        r'',
        r'3. Bilinear interpolation on each feature plane:',
        r'     f_xy = grid_sample(T_xy, grid_xy)  in R^(B x 128 x 512)',
        r'     f_yz = grid_sample(T_yz, grid_yz)  in R^(B x 128 x 512)',
        r'     f_zx = grid_sample(T_zx, grid_zx)  in R^(B x 128 x 512)',
        r'',
        r'4. Aggregate by summation:',
        r'     f_tri = f_xy + f_yz + f_zx        in R^(B x 128 x 512)',
    ]
    for i, l in enumerate(samp):
        ax.text(3, 50.5 - i*1.3, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    # ── Step 4: Flow Field ────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 8), 100, 24,
                 boxstyle="round,pad=0.3", facecolor=C['block3'],
                 edgecolor=C['good'], linewidth=1.5))
    ax.text(2, 30.5, 'STEP 4 — Symmetry Flow Field & Coarse Output',
            fontsize=11, fontweight='bold', color=C['good'])

    flow = [
        r'Input: f_tri  in  R^(B x 128 x 512)   (tri-plane features at keypoint locations)',
        r'',
        r'Flow field MLP (Conv1d):',
        r'   delta = Conv1d(128->256) -> LeakyReLU -> Conv1d(256->128) -> LeakyReLU -> Conv1d(128->3)',
        r'   delta  in  R^(B x 3 x 512)           (3-D displacement vectors)',
        r'',
        r'Symmetry prediction:',
        r'   sym_pts = keypoints + delta            (B, 3, 512)',
        r'   Each keypoint p_i is mapped to its predicted symmetric counterpart: q_i = p_i + delta_i',
        r'',
        r'Coarse output (concatenation of visible + predicted):',
        r'   coarse = cat[sym_pts, keypoints]       (B, 3, 1024)',
        r'           |-- predicted missing part --|  |-- visible part --|',
        r'',
        r'Coarse loss: L_coarse = ChamferDistanceL1(coarse, GT)',
        r'   NOTE: GT contains 16384 points but coarse has only 1024 -> loss is one-sided weighted',
    ]
    for i, l in enumerate(flow):
        ax.text(3, 28.5 - i*1.3, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — SGFormer Mathematics
# ══════════════════════════════════════════════════════════════════════════
def page_sgformer_math(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 4 — SGFormer: Refinement Stage Mathematics',
              'Symmetry-Guided Upsampling Transformer (×2 stages)')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(2, 98, 'SGFormer takes a coarse point cloud and upsamples it by up_factor using '
            'symmetry features and partial features as guidance.',
            fontsize=9.5, color=C['text'], va='top')

    # ── local_encoder ─────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 84), 100, 13,
                 boxstyle="round,pad=0.3", facecolor=C['block'],
                 edgecolor=C['border'], linewidth=1.5))
    ax.text(2, 95.5, 'local_encoder  —  Encodes symmetry_points for guidance',
            fontsize=11, fontweight='bold', color=C['head'])
    enc_lines = [
        r'Input: sym_pts  in  R^(B x 3 x 512)',
        r'feat = MLP_CONV(3 -> 64 -> 128)(sym_pts)                  (B, 128, 512)',
        r'feat = cat[feat, MaxPool(feat).repeat(512)]                (B, 256, 512)',
        r'feat = MLP_CONV(256 -> 128)(feat)                          (B, 128, 512)',
        r'feat_symmetry = Transformer(feat, sym_pts_pos, dim=64)     (B, 128, 512)',
        r'feat_partial  = keyfeatures  (reused from encoder)         (B, 128, 512)',
    ]
    for i, l in enumerate(enc_lines):
        ax.text(3, 93.5 - i*1.4, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    # ── SGFormer internal ─────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 42), 100, 40,
                 boxstyle="round,pad=0.3", facecolor='#e8f5e9',
                 edgecolor=C['good'], linewidth=1.5))
    ax.text(2, 80.5, 'SGFormer Internal — Point Feature Extraction & Dual Fusion',
            fontsize=11, fontweight='bold', color=C['good'])

    ax.text(2, 78.5, 'A. Point feature extraction from coarse points:', fontsize=9.5,
            fontweight='bold', color=C['text'], va='top')
    a_lines = [
        r'   feat = MLP_CONV(3->64->128)(coarse)                        (B, 128, N)',
        r'   feat_max = max(feat, dim=-1).repeat(N)                     (B, 128, N)',
        r'   feat = cat[feat, feat_max]                                 (B, 256, N)',
        r'   feat = MLP_CONV(256->256->128)(feat)                       (B, 128, N)',
        r'   feat = Transformer(feat, coarse_pos, dim=64)               (B, 128, N)  <- local geom. attention',
    ]
    for i, l in enumerate(a_lines):
        ax.text(3, 76 - i*1.4, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    ax.text(2, 68.5, 'B. Expand all features to 256-dim:', fontsize=9.5,
            fontweight='bold', color=C['text'], va='top')
    b_lines = [
        r'   feat         = MLP_CONV(128->256)(feat)          (B, 256, N)',
        r'   partial_feat  = MLP_CONV(128->256)(feat_partial)  (B, 256, N)',
        r'   symm_feat     = MLP_CONV(128->256)(feat_symmetry) (B, 256, N)',
    ]
    for i, l in enumerate(b_lines):
        ax.text(3, 66.5 - i*1.4, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    ax.text(2, 61.5, 'C. Dual Cross-Attention Fusion (CrossFormer):', fontsize=9.5,
            fontweight='bold', color=C['text'], va='top')
    ax.text(3, 59.5,
            r'CrossFormer(query=X, key=Y):  X = LayerNorm(X),  Y = LayerNorm(Y)'
            '\n'
            r'   attn = MultiheadAttention(Q=X, K=Y, V=Y)   (4 heads)'
            '\n'
            r'   X = X + attn                                (residual connection)'
            '\n'
            r'   X = X + FFN(LayerNorm(X))                  (feed-forward: Linear->LeakyReLU->Linear)',
            fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    c_lines = [
        r'   feat_p = Fusion(feat, partial_feat)             (B, 256, N)  <- partial awareness',
        r'   feat_s = Fusion(feat, symm_feat)                (B, 256, N)  <- symmetry awareness',
        r'   feat   = cat[feat_p, feat_s]                    (B, 512, N)',
    ]
    for i, l in enumerate(c_lines):
        ax.text(3, 54.5 - i*1.4, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    ax.text(2, 49.5, 'D. Self-attention + Upsampling:', fontsize=9.5,
            fontweight='bold', color=C['text'], va='top')
    d_lines = [
        r'   feat = Linear(512->512)(feat)                   (B, 512, N)',
        r'   feat = Fusion(feat, feat)                       (B, 512, N)  <- self-attention',
        r'   offset = FC(feat): Linear 512->512->128->(3*up_factor)',
        r'   offset reshaped to (B, N, up_factor, 3)',
        r'   coarse_expanded = coarse.repeat(up_factor)      (B, N*up_factor, 3)',
        r'   output = coarse_expanded + offset               (B, N*up_factor, 3)',
    ]
    for i, l in enumerate(d_lines):
        ax.text(3, 47.5 - i*1.4, l, fontsize=8.5, color=C['text'], va='top', fontfamily='monospace')

    # ── Two-stage pipeline ────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 8), 100, 32,
                 boxstyle="round,pad=0.3", facecolor=C['block4'],
                 edgecolor='#ffa000', linewidth=1.5))
    ax.text(2, 38.5, 'Two-Stage Upsampling Pipeline',
            fontsize=11, fontweight='bold', color='#e65100')

    # Flow diagram
    stages_up = [
        ('coarse\n(B,1024,3)', C['block'],  3,  26),
        ('SGFormer-1\nup×2',   '#a5d6a7',  25, 26),
        ('fine1\n(B,2048,3)',  C['block3'], 47, 26),
        ('SGFormer-2\nup×8',   '#a5d6a7',  69, 26),
        ('fine2\n(B,16384,3)', '#1b5e20',  91, 26),
    ]
    for label, col, x, y in stages_up:
        lc = 'white' if col == '#1b5e20' else C['text']
        ax.add_patch(FancyBboxPatch((x, y), 18, 10,
                     boxstyle="round,pad=0.4", facecolor=col,
                     edgecolor=C['border'], linewidth=1.2))
        ax.text(x+9, y+5, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=lc)

    for i in range(len(stages_up)-1):
        x0 = stages_up[i][2] + 18
        x1 = stages_up[i+1][2]
        y0 = stages_up[i][3] + 5
        arrow(ax, x0, y0, x1, y0)

    # Guidance arrows
    ax.text(50, 20, 'feat_symmetry (B,128,512)  +  feat_partial (B,128,512)  reused in BOTH stages',
            ha='center', fontsize=9, color='#5c6bc0', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#e8eaf6', edgecolor='#9fa8da'))
    ax.annotate('', xy=(34, 26), xytext=(50, 22),
                arrowprops=dict(arrowstyle='->', color='#5c6bc0', lw=1.5))
    ax.annotate('', xy=(78, 26), xytext=(50, 22),
                arrowprops=dict(arrowstyle='->', color='#5c6bc0', lw=1.5))

    ax.text(2, 14,
            'Loss: L_fine1 = CDL1(fine1, GT)    L_fine2 = CDL1(fine2, GT)\n'
            'Total Loss: L = L_coarse + L_fine1 + L_fine2   (equal weighting)',
            fontsize=9.5, color=C['text'], va='top', fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Loss Functions & Training
# ══════════════════════════════════════════════════════════════════════════
def page_loss(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 5 — Training Objective & Loss Functions',
              'Chamfer Distance L1, Training Schedule, Optimization')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # Chamfer Distance
    ax.add_patch(FancyBboxPatch((0, 75), 100, 22,
                 boxstyle="round,pad=0.3", facecolor=C['block'],
                 edgecolor=C['border'], linewidth=1.5))
    ax.text(2, 95.5, '1. Chamfer Distance L1  (CDL1)',
            fontsize=12, fontweight='bold', color=C['head'])

    ax.text(2, 93, 'Given predicted point cloud S and ground truth G:', fontsize=9.5, color=C['text'], va='top')
    ax.text(6, 90.5,
            r'CDL1(S, G) = (1/|S|) * sum_{s in S} min_{g in G} ||s - g||_1'
            '\n'
            r'           + (1/|G|) * sum_{g in G} min_{s in S} ||g - s||_1',
            fontsize=10, color=C['title'], va='top', fontfamily='monospace')
    ax.text(2, 85.5,
            'First term: for each predicted point, find nearest GT point  (precision — no hallucination)\n'
            'Second term: for each GT point, find nearest predicted point  (recall — no missing regions)\n'
            'L1 norm is more robust to outliers than L2 norm',
            fontsize=9, color=C['text'], va='top')

    ax.text(2, 81.5, 'Logged values are scaled ×1000 for readability.', fontsize=9,
            color=C['muted'], va='top', style='italic')

    # Total Loss
    ax.add_patch(FancyBboxPatch((0, 55), 100, 18,
                 boxstyle="round,pad=0.3", facecolor=C['block2'],
                 edgecolor='#ef9a9a', linewidth=1.5))
    ax.text(2, 71.5, '2. Total Training Loss', fontsize=12, fontweight='bold', color='#b71c1c')

    ax.text(2, 69, 'Three outputs are supervised simultaneously:', fontsize=9.5, color=C['text'], va='top')
    ax.text(6, 66.5,
            r'L_total = CDL1(coarse, GT)  +  CDL1(fine1, GT)  +  CDL1(fine2, GT)',
            fontsize=10.5, color=C['title'], va='top', fontfamily='monospace')
    ax.text(2, 63,
            'coarse: (B, 1024, 3)   — TriPlaneSymmetryNet output\n'
            'fine1:  (B, 2048, 3)   — SGFormer-1 output  (2x upsampled)\n'
            'fine2:  (B, 16384, 3)  — SGFormer-2 output  (final, 8x upsampled)\n\n'
            'Key observation: ALL three losses use the FULL GT (16384 pts).\n'
            'CDL1(coarse, GT) at 1024 pts is dominated by the "recall" term (GT->coarse)\n'
            'since 94% of GT points have no nearby coarse point.',
            fontsize=9, color=C['text'], va='top')

    # Critical flaw box
    ax.add_patch(FancyBboxPatch((0, 38), 100, 15,
                 boxstyle="round,pad=0.3", facecolor='#ffebee',
                 edgecolor=C['bad'], linewidth=2))
    ax.text(2, 51.5, '3. Critical Loss Design Issue  (Root Cause of Weak Symmetry Branch)',
            fontsize=12, fontweight='bold', color=C['bad'])
    ax.text(2, 49,
            'The coarse loss  CDL1(coarse, GT)  measures the CONCATENATION of [sym_pts, keypoints].\n\n'
            'keypoints are directly sampled from the partial input — they are ALWAYS accurate.\n'
            'sym_pts are the predicted missing part — these are often inaccurate.\n\n'
            'Since keypoints (~512 pts) already cover the visible region well, the model\n'
            'can minimize L_coarse without sym_pts being accurate.\n\n'
            'Result: the flow field learns small, conservative deltas (mean |delta| = 0.1354)\n'
            'rather than large, accurate symmetry displacements.',
            fontsize=9, color=C['bad'], va='top')

    # Scheduler
    ax.add_patch(FancyBboxPatch((0, 8), 100, 28,
                 boxstyle="round,pad=0.3", facecolor=C['block3'],
                 edgecolor=C['good'], linewidth=1.5))
    ax.text(2, 34.5, '4. Optimization & Learning Rate Schedule', fontsize=12, fontweight='bold', color=C['good'])

    sched_lines = [
        r'Optimizer:    AdamW  (lr=2e-4,  weight_decay=5e-4)',
        r'Scheduler:    WarmUpCosLR',
        r'  - Warm-up:  epochs 0-20   lr linearly increases from 0 to lr_max=2e-4',
        r'  - Cosine:   epochs 20-120 lr follows cosine annealing from lr_max to lr_min=1e-5',
        r'              lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*(t-warmup)/(max-warmup)))',
        r'Batch size:   32  (total_bs across GPUs)',
        r'Epochs:       120',
        r'Validation:   every 10 epochs on PCN test set (1200 samples)',
        r'Best model:   saved by CDL1 metric (lower is better)',
    ]
    for i, l in enumerate(sched_lines):
        ax.text(3, 32 - i*2.3, l, fontsize=9, color=C['text'], va='top', fontfamily='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6 — Diagnostic: Stage-wise Results
# ══════════════════════════════════════════════════════════════════════════
def page_diag_stages(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 6 — Diagnostic: Stage-wise CDL1 Breakdown',
              'From diagnose.py run on PCN test set (1200 samples, ckpt-best.pth)')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # ── Numbers table ─────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((5, 76), 90, 20,
                 boxstyle="round,pad=0.5", facecolor=C['block'],
                 edgecolor=C['border'], linewidth=1.5))
    ax.text(50, 94.5, 'Stage-wise Overall Results', ha='center',
            fontsize=13, fontweight='bold', color=C['head'])

    headers = ['Stage', 'CDL1', 'CDL2', 'Improvement (CDL1)', 'Cumulative %']
    col_x   = [8, 32, 48, 62, 82]
    ax.text(col_x[0], 92, headers[0], fontsize=9, fontweight='bold', color=C['head'])
    for i in range(1, len(headers)):
        ax.text(col_x[i], 92, headers[i], ha='center', fontsize=9,
                fontweight='bold', color=C['head'])

    rows = [
        ('Coarse (TriPlane)',   14.737, 0.95088, '—',         '—',     C['bad']),
        ('Fine-1 (SGFormer-1)', 10.288, 0.44671, '+4.449',    '30.2%', C['warn']),
        ('Fine-2 (SGFormer-2)',  6.633, 0.20553, '+3.655',    '55.0%', C['good']),
    ]
    for i, (stage, cd1, cd2, impr, cum, col) in enumerate(rows):
        y = 89 - i*4
        bg = '#f5f5f5' if i % 2 == 0 else 'white'
        ax.add_patch(FancyBboxPatch((6.5, y-3), 87, 3.8,
                     boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))
        ax.text(col_x[0], y, stage,          fontsize=9.5, color=col, fontweight='bold')
        ax.text(col_x[1], y, f'{cd1:.3f}',   fontsize=9.5, color=col, ha='center', fontweight='bold')
        ax.text(col_x[2], y, f'{cd2:.5f}',   fontsize=9,   color=C['text'], ha='center')
        ax.text(col_x[3], y, impr,           fontsize=9,   color=C['text'], ha='center')
        ax.text(col_x[4], y, cum,            fontsize=9,   color=C['text'], ha='center')

    # ── Bar chart ─────────────────────────────────────────────────────────
    ax_bar = fig.add_axes([0.05, 0.46, 0.42, 0.28])
    stages = ['Coarse\n(TriPlane)', 'Fine-1\n(SGFormer-1)', 'Fine-2\n(SGFormer-2)']
    cdl1   = [14.737, 10.288, 6.633]
    colors = ['#ef5350', '#ff9800', '#4caf50']
    bars = ax_bar.bar(stages, cdl1, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, cdl1):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_bar.set_ylabel('CDL1 (×1000)', fontsize=10)
    ax_bar.set_title('CDL1 per Stage', fontsize=11, fontweight='bold')
    ax_bar.set_ylim(0, 17)
    ax_bar.axhline(6.633, color='#2e7d32', linestyle='--', lw=1.5, label='Final target')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # ── Gain bar chart ────────────────────────────────────────────────────
    ax_gain = fig.add_axes([0.55, 0.46, 0.40, 0.28])
    gains  = [4.449, 3.655]
    glabs  = ['SGFormer-1\nGain', 'SGFormer-2\nGain']
    gcols  = ['#42a5f5', '#7e57c2']
    gbars = ax_gain.bar(glabs, gains, color=gcols, edgecolor='white', linewidth=1.5, width=0.4)
    for bar, val in zip(gbars, gains):
        ax_gain.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                     f'+{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax_gain.set_ylabel('CDL1 Reduction', fontsize=10)
    ax_gain.set_title('Refinement Gain per Stage', fontsize=11, fontweight='bold')
    ax_gain.set_ylim(0, 6)
    ax_gain.spines['top'].set_visible(False)
    ax_gain.spines['right'].set_visible(False)

    # ── Key observations ──────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0, 5), 100, 38,
                 boxstyle="round,pad=0.3", facecolor='#fff9c4',
                 edgecolor='#fbc02d', linewidth=1.5))
    ax.text(2, 41.5, 'Key Observations from Stage-wise Analysis',
            fontsize=12, fontweight='bold', color='#e65100')

    obs = [
        ('OBS-1', C['bad'],
         'The coarse stage CDL1 is 14.737 — more than DOUBLE the final output (6.633).',
         'This means the TriPlaneSymmetryNet is generating a very poor coarse reconstruction.'),
        ('OBS-2', C['warn'],
         'SGFormer-1 and SGFormer-2 together rescue 55% of the total quality (8.104 CDL1 gain).',
         'The architecture is over-relying on refiners to fix a weak coarse stage.'),
        ('OBS-3', C['good'],
         'Both refiners contribute almost equally: SF1=30.2%, SF2=35.5%.',
         'The refinement architecture itself is healthy and working as expected.'),
        ('OBS-4', C['head'],
         'If coarse improves from 14.737 to ~8.0, we expect final to drop below 6.0.',
         'Fixing the coarse stage is the highest-leverage improvement.'),
    ]
    for i, (tag, col, line1, line2) in enumerate(obs):
        y = 37 - i*8
        ax.add_patch(FancyBboxPatch((1.5, y-6), 97, 7,
                     boxstyle="round,pad=0.2", facecolor='white',
                     edgecolor=col, linewidth=1.5))
        ax.text(3, y-0.5, tag, fontsize=9, fontweight='bold', color=col, va='top')
        ax.text(10, y-0.5, line1, fontsize=9, color=col, fontweight='bold', va='top')
        ax.text(10, y-3.0, line2, fontsize=8.5, color=C['text'], va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7 — Diagnostic: Symmetry Branch
# ══════════════════════════════════════════════════════════════════════════
def page_diag_symmetry(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 7 — Diagnostic: Symmetry Branch Analysis',
              'symmetry_points quality, delta magnitude, and what they reveal')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # Numbers
    boxes_sym = [
        ('sym_pts CDL1 vs GT', '19.322', C['bad'],   4,  82, 28, 12),
        ('coarse CDL1 vs GT',  '14.737', '#ef6c00',  36, 82, 28, 12),
        ('final CDL1 vs GT',   '6.633',  C['good'],  68, 82, 28, 12),
    ]
    for label, val, col, x, y, w, h in boxes_sym:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.5", facecolor='white',
                     edgecolor=col, linewidth=2))
        ax.text(x+w/2, y+h*0.7, label, ha='center', fontsize=8.5,
                color=C['muted'], fontweight='bold', va='center')
        ax.text(x+w/2, y+h*0.25, val, ha='center', fontsize=16,
                color=col, fontweight='bold', va='center')

    ax.add_patch(FancyBboxPatch((4, 70), 92, 10,
                 boxstyle="round,pad=0.3", facecolor='#ffebee',
                 edgecolor=C['bad'], linewidth=2))
    ax.text(50, 78, 'sym_pts CDL1 (19.322)  >  coarse CDL1 (14.737)',
            ha='center', fontsize=11, fontweight='bold', color=C['bad'])
    ax.text(50, 74.5,
            'The symmetry_points ALONE are WORSE than the coarse output (which includes keypoints).\n'
            'Keypoints from the partial input are what is pulling the coarse CDL1 down to 14.737.',
            ha='center', fontsize=9, color=C['bad'], va='top')

    ax.text(2, 68, 'Delta Magnitude Analysis', fontsize=12, fontweight='bold', color=C['head'])
    hline(ax, 67, color=C['border'])

    ax.add_patch(FancyBboxPatch((2, 50), 96, 16,
                 boxstyle="round,pad=0.3", facecolor=C['block'],
                 edgecolor=C['border'], linewidth=1.5))
    ax.text(4, 64,
            r'mean |delta| = 0.1354   (Euclidean norm of displacement vector)',
            fontsize=10, color=C['text'], va='top', fontfamily='monospace')
    ax.text(4, 61.5,
            'Point clouds are roughly normalized to unit sphere (radius ~1.0).\n'
            'A delta magnitude of 0.1354 means the model displaces keypoints by only 13.5% of the '
            'object radius.\n'
            'For genuine bilateral symmetry, the displacement should often be close to the FULL '
            'width of the object\n'
            '(e.g., left wing → right wing on an airplane is ~2x the half-width ≈ displacement of ~1.0).\n\n'
            'Interpretation: the flow field has COLLAPSED to predicting near-zero displacements.\n'
            'The model learned to keep sym_pts close to keypoints (lazy symmetry).',
            fontsize=9, color=C['text'], va='top')

    # Visual illustration
    ax.text(2, 48, 'Why the Flow Field Collapses', fontsize=12,
            fontweight='bold', color=C['head'])
    hline(ax, 47, color=C['border'])

    ax.add_patch(FancyBboxPatch((2, 8), 96, 38,
                 boxstyle="round,pad=0.3", facecolor='#fce4ec',
                 edgecolor=C['bad'], linewidth=1.5))

    ax.text(4, 44,
            'The coarse loss is:   L_coarse = CDL1(cat[sym_pts, keypoints], GT)',
            fontsize=10, color=C['bad'], va='top', fontfamily='monospace', fontweight='bold')

    ax.text(4, 40.5,
            'Scenario A — sym_pts are accurate (large, correct delta):',
            fontsize=9.5, fontweight='bold', color=C['good'], va='top')
    ax.text(4, 38,
            '   CDL1(cat[good_sym, keypoints], GT)  ≈  small  ✓',
            fontsize=9, color=C['good'], va='top', fontfamily='monospace')

    ax.text(4, 35,
            'Scenario B — sym_pts are near-zero delta (lazy, wrong):',
            fontsize=9.5, fontweight='bold', color=C['bad'], va='top')
    ax.text(4, 32.5,
            '   keypoints already cover visible half of object well\n'
            '   CDL1(cat[lazy_sym, keypoints], GT)  ≈  medium  (still acceptable!)',
            fontsize=9, color=C['bad'], va='top', fontfamily='monospace')

    ax.text(4, 28,
            'The gradient through L_coarse w.r.t. sym_pts is WEAK because keypoints provide\n'
            'a strong baseline. The flow field never gets a strong gradient signal to learn\n'
            'large, accurate displacements.',
            fontsize=9.5, color=C['bad'], va='top')

    ax.text(4, 22,
            'FIX: Add a DEDICATED symmetry loss:',
            fontsize=10, fontweight='bold', color=C['head'], va='top')
    ax.text(4, 19.5,
            r'L_sym = CDL1(sym_pts, GT)    or    L_sym = CDL1(sym_pts, GT_missing_part)',
            fontsize=10, color=C['title'], va='top', fontfamily='monospace')
    ax.text(4, 16.5,
            'This directly penalizes bad symmetry predictions independent of keypoints.\n'
            'The flow field will then receive a strong gradient to learn large, accurate deltas.',
            fontsize=9.5, color=C['text'], va='top')

    ax.text(4, 12,
            r'New total loss:  L = L_coarse + lambda_sym * L_sym + L_fine1 + L_fine2',
            fontsize=10, color=C['title'], va='top', fontfamily='monospace')
    ax.text(4, 9.5,
            'Recommended: lambda_sym = 0.5 initially, tune based on coarse CDL1 improvement.',
            fontsize=9, color=C['muted'], va='top', style='italic')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 8 — Per-Category Analysis
# ══════════════════════════════════════════════════════════════════════════
def page_per_category(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 8 — Per-Category Stage-wise Analysis',
              'Which categories suffer most and what it reveals')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    categories = ['cabinet', 'sofa', 'car', 'chair', 'table', 'watercraft', 'lamp', 'airplane']
    coarse  = [18.782, 19.693, 16.929, 15.860, 13.803, 11.923, 11.770,  9.137]
    fine1   = [13.565, 13.697, 11.661, 11.108,  9.645,  8.761,  7.942,  5.928]
    fine2   = [ 8.943,  8.761,  7.687,  6.974,  6.033,  5.743,  5.162,  3.764]
    sf1gain = [ 5.216,  5.996,  5.268,  4.752,  4.157,  3.162,  3.828,  3.209]
    sf2gain = [ 4.622,  4.936,  3.973,  4.134,  3.612,  3.018,  2.780,  2.164]

    # Grouped bar chart
    ax_cat = fig.add_axes([0.04, 0.52, 0.92, 0.38])
    x = np.arange(len(categories))
    w = 0.25
    ax_cat.bar(x - w, coarse, w, label='Coarse', color='#ef5350', alpha=0.85)
    ax_cat.bar(x,     fine1,  w, label='Fine-1', color='#ff9800', alpha=0.85)
    ax_cat.bar(x + w, fine2,  w, label='Fine-2', color='#4caf50', alpha=0.85)
    ax_cat.set_xticks(x)
    ax_cat.set_xticklabels(categories, rotation=15, fontsize=9)
    ax_cat.set_ylabel('CDL1 (×1000)', fontsize=10)
    ax_cat.set_title('Per-Category CDL1 at Each Stage  (sorted worst→best final CDL1)',
                     fontsize=11, fontweight='bold')
    ax_cat.legend(fontsize=9)
    ax_cat.spines['top'].set_visible(False)
    ax_cat.spines['right'].set_visible(False)
    ax_cat.set_ylim(0, 23)
    ax_cat.grid(axis='y', alpha=0.3)

    # Table
    ax.text(2, 49, 'Detailed Per-Category Table', fontsize=11,
            fontweight='bold', color=C['head'])
    hline(ax, 48, color=C['border'])

    cols_x = [2, 22, 35, 48, 61, 74, 87]
    headers = ['Category', 'N', 'Coarse', 'Fine-1', 'Fine-2', 'SF1 gain', 'SF2 gain']
    for hx, hd in zip(cols_x, headers):
        ax.text(hx, 46.5, hd, fontsize=8.5, fontweight='bold', color=C['head'], va='top')

    data = list(zip(categories, [150]*8, coarse, fine1, fine2, sf1gain, sf2gain))
    for i, (cat, n, c0, c1, c2, g1, g2) in enumerate(data):
        y = 43.5 - i*4.5
        bg = '#f5f5f5' if i % 2 == 0 else 'white'
        ax.add_patch(FancyBboxPatch((1.5, y-3.5), 97, 4,
                     boxstyle='square,pad=0', facecolor=bg, edgecolor='none'))
        col = C['bad'] if c2 > 7.5 else (C['warn'] if c2 > 6.0 else C['good'])
        ax.text(cols_x[0], y, cat,          fontsize=9, color=col, fontweight='bold', va='top')
        ax.text(cols_x[1], y, str(n),       fontsize=9, color=C['text'], va='top')
        ax.text(cols_x[2], y, f'{c0:.3f}',  fontsize=9, color=C['bad'],  va='top')
        ax.text(cols_x[3], y, f'{c1:.3f}',  fontsize=9, color=C['warn'], va='top')
        ax.text(cols_x[4], y, f'{c2:.3f}',  fontsize=9, color=col,       va='top', fontweight='bold')
        ax.text(cols_x[5], y, f'+{g1:.3f}', fontsize=9, color=C['text'], va='top')
        ax.text(cols_x[6], y, f'+{g2:.3f}', fontsize=9, color=C['text'], va='top')

    # Analysis
    ax.add_patch(FancyBboxPatch((0, 2), 100, 8,
                 boxstyle="round,pad=0.3", facecolor=C['block4'],
                 edgecolor='#ffa000', linewidth=1.5))
    insights = [
        ('airplane (3.764)', C['good'],
         'Strongest bilateral symmetry. TriPlane can learn this pattern well.'),
        ('sofa/cabinet (8.7-8.9)', C['bad'],
         'Complex shapes, weak symmetry. Coarse ~19 → refiners struggle more.'),
        ('All categories', C['head'],
         'Coarse is consistently 2× worse than final. The symmetry net bottleneck is universal.'),
    ]
    for i, (cat, col, text) in enumerate(insights):
        ax.text(2 + i*33, 9.5, cat,  fontsize=8.5, color=col, fontweight='bold', va='top')
        ax.text(2 + i*33, 7.5, text, fontsize=7.5, color=C['text'], va='top')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 9 — Conclusions & Recommendations
# ══════════════════════════════════════════════════════════════════════════
def page_conclusions(pdf):
    fig = new_fig()
    title_bar(fig, 'Page 9 — Conclusions & Recommended Fixes',
              'Root cause summary and actionable next steps')
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.90])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # Summary
    ax.add_patch(FancyBboxPatch((0, 84), 100, 13,
                 boxstyle="round,pad=0.3", facecolor=C['title'],
                 edgecolor='none'))
    ax.text(50, 92, 'Root Cause Summary', ha='center', fontsize=14,
            fontweight='bold', color='white')
    ax.text(50, 88,
            'The TriPlane symmetry flow field produces near-zero displacements (mean |delta|=0.1354) '
            'because the coarse\n'
            'loss signal is dominated by accurate keypoints. sym_pts CDL1=19.322 vs coarse CDL1=14.737 '
            'confirms the\n'
            'symmetry branch is essentially inactive. SGFormers compensate, but this limits the ceiling.',
            ha='center', fontsize=9.5, color='#c5cae9', va='top')

    # Fixes
    fixes = [
        ('FIX 1 — Add Dedicated Symmetry Loss', C['bad'], True,
         [
             'Add  L_sym = CDL1(sym_pts.T, GT)  to training loss',
             'New total:  L = L_coarse + 0.5*L_sym + L_fine1 + L_fine2',
             'This gives flow field a direct gradient to learn large, accurate displacements',
             'Expected impact: coarse CDL1 drops from 14.7 to ~10-12,  final from 6.6 to ~5.5-6.0',
             'Code change: 2 lines in get_loss() in SymmCompletion.py',
         ]),
        ('FIX 2 — Asymmetric Coarse Loss (optional)', C['warn'], False,
         [
             'Replace CDL1(coarse, GT) with CDL1(sym_pts.T, GT) only (drop keypoints from coarse loss)',
             'Forces the entire coarse loss gradient to flow through sym_pts',
             'Risk: may destabilize training since keypoints no longer supervised',
             'Safer alternative to FIX 1 if 0.5 weight is not enough',
         ]),
        ('FIX 3 — Increase plane_dim back to 128 (already done in V11)', C['head'], False,
         [
             'Already reverted in Symm3dTriV11 (plane_dim=64→128)',
             'But V11 shows shared-FC is the bottleneck, not plane_dim',
             'This fix is necessary but not sufficient alone',
         ]),
        ('DO NOT CHANGE — SGFormer Architecture', C['good'], False,
         [
             'SGFormer-1 and SGFormer-2 are working correctly',
             'Both contribute ~30-35% improvement per stage',
             'CrossFormer dual fusion (partial + symmetry) is effective',
             'No changes needed here',
         ]),
    ]

    y_start = 82
    for i, (title, col, priority, bullets) in enumerate(fixes):
        h = 5 + len(bullets)*3.5
        y = y_start - h - 2
        y_start = y - 1
        face = '#ffebee' if col == C['bad'] else ('#fff8e1' if col == C['warn'] else
               ('#e8f5e9' if col == C['good'] else '#e8eaf6'))
        ax.add_patch(FancyBboxPatch((0, y), 100, h+1,
                     boxstyle="round,pad=0.3", facecolor=face,
                     edgecolor=col, linewidth=2 if priority else 1))
        badge = ' ★ HIGH PRIORITY' if priority else ''
        ax.text(2, y+h, title+badge, fontsize=10, fontweight='bold',
                color=col, va='top')
        for j, b in enumerate(bullets):
            ax.text(4, y+h-3.5 - j*3.2, u'\u2022 '+b, fontsize=8.5,
                    color=C['text'], va='top')

    # Final summary table
    ax.add_patch(FancyBboxPatch((0, 2), 100, 7,
                 boxstyle="round,pad=0.3", facecolor='#e8eaf6',
                 edgecolor=C['sub'], linewidth=1.5))
    ax.text(50, 8.5, 'Expected Improvement After FIX 1', ha='center',
            fontsize=10, fontweight='bold', color=C['head'])
    cols = ['Metric', 'Current', 'Expected After Fix']
    xs   = [5, 35, 65]
    for x, h in zip(xs, cols):
        ax.text(x, 7.5, h, fontsize=8.5, fontweight='bold', color=C['head'], va='top')
    rows_f = [
        ('sym_pts CDL1', '19.322', '~12-14'),
        ('Coarse CDL1',  '14.737', '~10-12'),
        ('Final CDL1',   '6.633',  '~5.5-6.0'),
    ]
    for j, (m, cur, exp) in enumerate(rows_f):
        x_off = j * 30 + 5
        ax.text(x_off,    5.5, m,   fontsize=8,  color=C['text'], va='top')
        ax.text(x_off+15, 5.5, cur, fontsize=8,  color=C['bad'],  va='top', fontweight='bold')
        ax.text(x_off+23, 5.5, exp, fontsize=8,  color=C['good'], va='top', fontweight='bold')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    print(f'Generating report → {OUT_PATH}')
    with PdfPages(OUT_PATH) as pdf:
        page_cover(pdf)
        page_architecture(pdf)
        page_triplane_math(pdf)
        page_sgformer_math(pdf)
        page_loss(pdf)
        page_diag_stages(pdf)
        page_diag_symmetry(pdf)
        page_per_category(pdf)
        page_conclusions(pdf)

        d = pdf.infodict()
        d['Title']   = 'Symm3dTri Architecture & Diagnostic Report'
        d['Author']  = 'Auto-generated by generate_diagnostic_report.py'
        d['Subject'] = 'Point Cloud Completion — Architecture Analysis'

    print('Done.')

if __name__ == '__main__':
    main()

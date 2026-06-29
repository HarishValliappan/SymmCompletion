"""
Diagnostic pipeline for Symm3dTri.

Runs the full model on the test set and reports per-stage losses so we can
see exactly where the architecture degrades:

  Stage 0  – TriPlaneSymmetryNet coarse output  (keypoints + sym_points)
  Stage 1  – SGFormer-1 fine output
  Stage 2  – SGFormer-2 fine output  (final prediction)

Extra diagnostics captured via forward hooks:
  • symmetry_points CDL1 vs GT   – quality of the symmetry branch alone
  • delta magnitude               – how large the symmetry displacement is

Per-category breakdown is printed for every stage so the bottleneck
(coarse vs refinement, category-wise) is immediately visible.

Usage (see diagnose.sh):
  python diagnose.py \
      --config cfgs/ShapeNet55_models/SymmCompletion.yaml \
      --ckpts  weights/shapenet55/ckpt-best.pth \
      --mode   median

NOTE: imports are done INSIDE main() to avoid the knn_cuda CUDA-at-import-time
check that fires when tools/__init__.py → runner.py → models/__init__.py →
AnchorFormer → knn_cuda is loaded on a node before CUDA is confirmed available.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

CROP_RATIO = {'easy': 1 / 4, 'median': 1 / 2, 'hard': 3 / 4}


# ──────────────────────────────────────────────────────────────────────────
# Argument parsing  (no heavy imports here)
# ──────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Symm3dTri diagnostic pipeline')
    p.add_argument('--config',   type=str,
                   default='cfgs/ShapeNet55_models/SymmCompletion.yaml')
    p.add_argument('--ckpts',    type=str, required=True)
    p.add_argument('--mode',     type=str, default='median',
                   choices=['easy', 'median', 'hard'])
    p.add_argument('--exp_name', type=str, default='diagnose')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    args.local_rank    = 0
    args.distributed   = False
    args.use_gpu       = True       # checked below after torch import
    args.deterministic = True
    args.launcher      = 'none'
    args.experiment_path = os.path.join(
        './experiments',
        Path(args.config).stem,
        Path(args.config).parent.stem,
        'test_' + args.exp_name + '_' + args.mode,
    )
    args.tfboard_path = args.experiment_path
    args.log_name     = Path(args.config).stem
    args.resume       = False
    args.start_ckpts  = None
    args.test         = True
    args.mode         = args.mode
    os.makedirs(args.experiment_path, exist_ok=True)
    return args


# ──────────────────────────────────────────────────────────────────────────
# Forward hooks
# ──────────────────────────────────────────────────────────────────────────

class IntermediateCapture:
    def __init__(self, model):
        self.net = model.module if hasattr(model, 'module') else model
        self._buf = {}
        self._handles = []
        self._register()

    def _register(self):
        h = self.net.triplane_net.register_forward_hook(self._hook_triplane)
        self._handles.append(h)
        h = self.net.triplane_net.flow_field.register_forward_hook(self._hook_delta)
        self._handles.append(h)

    def _hook_triplane(self, module, inp, out):
        self._buf['symmetry_points'] = out[1].detach()   # (B, 3, 512)

    def _hook_delta(self, module, inp, out):
        self._buf['delta'] = out.detach()                # (B, 3, 512)

    def get(self, key, default=None):
        return self._buf.get(key, default)

    def clear(self):
        self._buf.clear()

    def remove(self):
        for h in self._handles:
            h.remove()


# ──────────────────────────────────────────────────────────────────────────
# Diagnostic loop
# ──────────────────────────────────────────────────────────────────────────

def diagnose(model, dataloader, cap, cd1, cd2, config, args, logger):
    import torch
    from utils import misc
    from utils.AverageMeter import AverageMeter
    from utils.logger import print_log

    model.eval()

    dataset_name = config.dataset.test._base_.NAME
    npoints      = config.dataset.test._base_.N_POINTS
    is_shapenet  = dataset_name == 'ShapeNet'
    num_crop     = int(npoints * CROP_RATIO[args.mode]) if is_shapenet else None

    STAGE_NAMES = ['Coarse (TriPlane)', 'Fine-1 (SGFormer-1)', 'Fine-2 (SGFormer-2)']
    N_STAGES    = 3

    def make_meters(n):
        return [AverageMeter(['CDL1', 'CDL2']) for _ in range(n)]

    overall           = make_meters(N_STAGES)
    sym_cd1_meter     = AverageMeter(['SymCDL1'])
    delta_mag_meter   = AverageMeter(['DeltaMag'])
    category_metrics  = {}

    shapenet_choice = [
        torch.Tensor([1,  1,  1]), torch.Tensor([1,  1, -1]),
        torch.Tensor([1, -1,  1]), torch.Tensor([-1, 1,  1]),
        torch.Tensor([-1,-1,  1]), torch.Tensor([-1, 1, -1]),
        torch.Tensor([1, -1, -1]), torch.Tensor([-1,-1, -1]),
    ] if is_shapenet else None

    n_samples = len(dataloader)
    print_log(
        f'[DIAGNOSE] {n_samples} samples  |  mode={args.mode}  |  dataset={dataset_name}',
        logger=logger,
    )

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            taxonomy_id = (taxonomy_ids[0]
                           if isinstance(taxonomy_ids[0], str)
                           else taxonomy_ids[0].item())

            if is_shapenet:
                gt       = data.cuda()
                partials = []
                for item in shapenet_choice:
                    p, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    partials.append(misc.fps(p, 2048))
            elif dataset_name in ('PCN', 'MVP'):
                gt       = data[1].cuda()
                partials = [data[0].cuda()]
            else:
                raise NotImplementedError(dataset_name)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = make_meters(N_STAGES)

            for partial in partials:
                cap.clear()
                ret = model(partial)

                stage_preds = [ret[0], ret[1], ret[-1]]
                for s, pred in enumerate(stage_preds):
                    v1 = cd1(pred, gt).item() * 1000
                    v2 = cd2(pred, gt).item() * 1000
                    overall[s].update([v1, v2])
                    category_metrics[taxonomy_id][s].update([v1, v2])

                sym_pts = cap.get('symmetry_points')
                if sym_pts is not None:
                    sym_pts_t = sym_pts.transpose(2, 1).contiguous()
                    sym_cd1_meter.update([cd1(sym_pts_t, gt).item() * 1000])

                delta = cap.get('delta')
                if delta is not None:
                    delta_mag_meter.update([delta.norm(dim=1).mean().item()])

            if (idx + 1) % 200 == 0:
                print_log(
                    f'  [{idx+1}/{n_samples}]  '
                    f'Coarse={overall[0].avg(0):.3f}  '
                    f'Fine1={overall[1].avg(0):.3f}  '
                    f'Fine2={overall[2].avg(0):.3f}',
                    logger=logger,
                )

    # ── Print results ──────────────────────────────────────────────────────
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    sep = '=' * 80

    print_log(sep, logger=logger)
    print_log('STAGE-WISE OVERALL RESULTS', logger=logger)
    print_log(sep, logger=logger)
    print_log(f"{'Stage':<26}  {'CDL1':>8}  {'CDL2':>10}", logger=logger)
    print_log('-' * 50, logger=logger)
    for s, name in enumerate(STAGE_NAMES):
        print_log(
            f'{name:<26}  {overall[s].avg(0):>8.3f}  {overall[s].avg(1):>10.5f}',
            logger=logger,
        )

    c0 = overall[0].avg(0)
    c1 = overall[1].avg(0)
    c2 = overall[2].avg(0)
    print_log('-' * 50, logger=logger)
    print_log(f'SGFormer-1 gain : {c0-c1:+.3f}  ({(c0-c1)/c0*100:.1f}%)', logger=logger)
    print_log(f'SGFormer-2 gain : {c1-c2:+.3f}  ({(c1-c2)/c1*100:.1f}%)', logger=logger)
    print_log(f'Total gain      : {c0-c2:+.3f}  ({(c0-c2)/c0*100:.1f}%)', logger=logger)

    print_log(sep, logger=logger)
    print_log('SYMMETRY BRANCH DIAGNOSTICS', logger=logger)
    print_log(sep, logger=logger)
    sym_c1 = sym_cd1_meter.avg(0)
    print_log(f'  symmetry_points CDL1 vs GT : {sym_c1:.3f}', logger=logger)
    print_log(f'  mean delta magnitude        : {delta_mag_meter.avg(0):.4f}', logger=logger)
    print_log(
        f'  sym_pts CDL1 vs coarse CDL1 : {sym_c1 - c0:+.3f}  '
        f'(+ve = sym branch alone is worse than coarse which includes keypoints too)',
        logger=logger,
    )

    print_log(sep, logger=logger)
    print_log('PER-CATEGORY  STAGE-WISE CDL1  (sorted worst→best final CDL1)', logger=logger)
    print_log(sep, logger=logger)
    print_log(
        f"{'Category':<16}  {'N':>5}  "
        f"{'Coarse':>8}  {'Fine-1':>8}  {'Fine-2':>8}  "
        f"{'SF1 gain':>9}  {'SF2 gain':>9}",
        logger=logger,
    )
    print_log('-' * 80, logger=logger)

    rows = []
    for tax_id, meters in category_metrics.items():
        cat  = shapenet_dict.get(str(tax_id), str(tax_id))
        n    = meters[0].count(0)
        v0, v1, v2 = meters[0].avg(0), meters[1].avg(0), meters[2].avg(0)
        rows.append((cat, n, v0, v1, v2, v0-v1, v1-v2))

    rows.sort(key=lambda r: r[4], reverse=True)   # worst final CDL1 first
    for cat, n, v0, v1, v2, g1, g2 in rows:
        print_log(
            f'{cat:<16}  {n:>5}  '
            f'{v0:>8.3f}  {v1:>8.3f}  {v2:>8.3f}  '
            f'{g1:>+9.3f}  {g2:>+9.3f}',
            logger=logger,
        )

    print_log(sep, logger=logger)
    print_log('[DIAGNOSE] Done.', logger=logger)


# ──────────────────────────────────────────────────────────────────────────
# Entry point — all heavy imports happen here, AFTER CUDA is available
# ──────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # ── CUDA check BEFORE any CUDA-dependent import ───────────────────────
    import torch
    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — no GPU on this node.")
        print("       Re-submit the job or check SLURM GPU allocation.")
        sys.exit(1)

    # ── Now safe to import GPU-dependent packages ─────────────────────────
    # Import ONLY what diagnose.py needs, bypassing tools/__init__.py and
    # models/__init__.py which would pull in AnchorFormer → knn_cuda.
    from utils.logger import get_root_logger, print_log
    from utils.config import get_config
    from utils import misc
    from utils.AverageMeter import AverageMeter

    # Import builder directly (avoids tools/__init__ → runner → all models)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        'tools.builder',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'builder.py'),
    )
    builder = _ilu.module_from_spec(_spec)
    sys.modules['tools.builder'] = builder
    _spec.loader.exec_module(builder)

    # Import only the model and dataset we actually need (skip AnchorFormer etc.)
    from models.build import build_model_from_cfg
    import models.SymmCompletion          # registers SymmCompletion in the registry
    import datasets.ShapeNet55Dataset     # registers ShapeNet55
    import datasets.PCNDataset            # registers PCN
    import datasets.MVPDataset            # registers MVP

    from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

    misc.set_random_seed(args.seed, deterministic=args.deterministic)

    log_file = os.path.join(args.experiment_path, 'diagnose.log')
    logger   = get_root_logger(log_file=log_file, name=args.log_name)

    config = get_config(args, logger=logger)
    config.dataset.test.others.bs = 1

    print_log(f'Loading checkpoint: {args.ckpts}', logger=logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    model = build_model_from_cfg(config.model)

    state_dict = torch.load(args.ckpts, map_location='cpu')['base_model']
    weights    = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(weights)
    model = model.cuda()
    model.eval()

    cap = IntermediateCapture(model)
    cd1 = ChamferDistanceL1()
    cd2 = ChamferDistanceL2()

    diagnose(model, test_dataloader, cap, cd1, cd2, config, args, logger)
    cap.remove()


if __name__ == '__main__':
    main()

"""Generate and optimize new designs, or verify released reference samples."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from runtime import ROOT, load_models
from validation.design_generation_utils import (
    run_single_posture_sampler, run_multiposture_sampler, remap,
    DIFFUSION_GEOM_BOUNDS, DEFAULT_GEOM_BOUNDS, is_geom_feasible,
)
from validation.opt_utils import run_de_temp, run_de_temp_trajectory, par2sig


def read_json(path):
    return json.loads(Path(path).read_text())


def summarize(results):
    """Success requires every posture/frame of a design to pass the threshold."""
    maxima = [np.array([max(d['mse']) for d in r['designs']]) for r in results]
    flat = np.concatenate(maxima)
    return {'targets': len(results), 'designs': len(flat),
            'success_rate': float(np.mean([np.mean(x < 1e-3) for x in maxima])),
            'median_max_mse': float(np.median(flat)),
            'p90_max_mse': float(np.quantile(flat, .9)),
            'max_mse_quantiles': {str(q): float(np.quantile(flat, q)) for q in [0, .1, .25, .5, .75, .9, .99, 1]},
            'success_by_target': {str(r['target_id']): float(np.mean(x < 1e-3))
                                  for r, x in zip(results, maxima)}}


def verify(results, targets, surrogate, device):
    differences = []
    feasible = []
    for result in results:
        postures = np.asarray(targets[result['target_id']]['postures'], dtype=np.float32)
        for design in result['designs']:
            geometry = np.asarray(design['geometry'], dtype=np.float32)
            temps = np.asarray(design['temperatures'], dtype=np.float32)
            saved = np.asarray(design['mse'], dtype=np.float64)
            if geometry.shape != (6,) or not all(np.isfinite(x).all() for x in [geometry, temps, saved]):
                raise ValueError('Nonfinite or invalid reference values')
            if np.any(temps < 0) or np.any(temps > 1) or np.any(temps[:, 2] < temps[:, 1] - 1e-6):
                raise ValueError('Temperatures violate bounds or ordering')
            if temps.shape != (len(postures), 3) or saved.shape != (len(postures),):
                raise ValueError('Reference posture/temperature dimensions do not match')
            params = np.column_stack([np.tile(geometry, (len(temps), 1)), temps])
            with torch.no_grad():
                prediction = surrogate(torch.from_numpy(par2sig(params)).to(device))
                mse = ((prediction - torch.from_numpy(postures).to(device)) ** 2).mean((1, 2)).cpu().numpy()
            differences.extend(abs(mse - saved).tolist())
            feasible.append(bool(is_geom_feasible(geometry, tol=1e-5)))
    return {'max_absolute_mse_difference': max(differences),
            'mse_values_close': bool(max(differences) < 1e-5),
            'geometry_feasible_fraction': float(np.mean(feasible))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['generate', 'verify', 'compare'])
    parser.add_argument('--task', choices=['single', 'three', 'trajectory'], required=True)
    parser.add_argument('--input', type=Path, help='Result file; defaults to released reference')
    parser.add_argument('--output', type=Path, default=Path('results/generated.json'))
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--limit', type=int, help='First N targets for a smoke check')
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--seed', type=int, help='Optional seed for your own new run')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.samples < 1 or args.maxiter < 1 or (args.limit is not None and args.limit < 1):
        parser.error('samples, maxiter and limit must be positive')
    reference = read_json(ROOT / 'reference' / f'{args.task}.json')
    targets = read_json(ROOT / 'targets' / f'{args.task}.json')
    if args.action == 'compare':
        if args.input is None:
            parser.error('compare requires --input')
        candidate = read_json(args.input)
        if candidate['task'] != args.task:
            parser.error('input task mismatch')
        ids = [r['target_id'] for r in candidate['results']]
        if not ids or len(set(ids)) != len(ids) or any(i not in range(len(targets)) for i in ids):
            parser.error('input must contain unique valid target IDs')
        baseline = [r for r in reference['results'] if r['target_id'] in ids]
        print(json.dumps({'reference': summarize(baseline), 'candidate': summarize(candidate['results']),
                          'note': 'Descriptive comparison, not a statistical equivalence test.'}, indent=2))
        return
    torch.set_num_threads(min(4, torch.get_num_threads()))
    surrogate, model, schedule = load_models(args.device)
    if args.action == 'verify':
        data = read_json(args.input) if args.input else reference
        if data['task'] != args.task:
            parser.error('input task mismatch')
        results = data['results'][:args.limit]
        ids = [r['target_id'] for r in results]
        if not ids or len(set(ids)) != len(ids) or any(i not in range(len(targets)) for i in ids):
            parser.error('input must contain unique valid target IDs')
        checks = verify(results, targets, surrogate, args.device)
        print(json.dumps({'summary': summarize(results), **checks}, indent=2))
        if not checks['mse_values_close'] or checks['geometry_feasible_fraction'] < 1:
            raise SystemExit(1)
        return
    rng = np.random.default_rng(args.seed)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve().is_relative_to(ROOT / 'reference') or args.output.resolve().is_relative_to(ROOT / 'targets'):
        parser.error('output must not overwrite released targets or reference samples')
    if args.output.exists():
        parser.error('output already exists; choose a new output path')
    for item in targets[:args.limit]:
        postures = np.asarray(item['postures'], dtype=np.float32)
        sampling_seed = int(rng.integers(0, 2**32))
        if args.task == 'single':
            designs, _ = run_single_posture_sampler(model, schedule, torch.from_numpy(postures[0]).to(args.device),
                args.samples, guidance_scale=args.guidance, seed=sampling_seed, device=args.device,
                record_flow=False, use_constraint_projection=True)
        else:
            designs = run_multiposture_sampler(model, schedule, torch.from_numpy(postures).to(args.device),
                args.samples, guidance_scale=args.guidance, n_mcmc_steps=10, gamma=.5,
                seed=sampling_seed, device=args.device, use_constraint_projection=True)
        geometries = remap(designs.cpu(), DIFFUSION_GEOM_BOUNDS, DEFAULT_GEOM_BOUNDS).numpy()
        records = []
        for geometry in geometries:
            optimization_seed = int(rng.integers(0, 2**31))
            if args.task == 'trajectory':
                temperatures, losses, _ = run_de_temp_trajectory(postures, geometry, surrogate, args.device,
                    num_points=30, maxiter=args.maxiter, lam=5e-4, seed=optimization_seed)
            else:
                optimized = [run_de_temp(p, geometry, surrogate, args.device,
                    num_points=30, maxiter=args.maxiter, seed=int(rng.integers(0, 2**32))) for p in postures]
                temperatures = np.array([o[0] for o in optimized])
                losses = [o[1] for o in optimized]
            records.append({'geometry': geometry.tolist(), 'temperatures': temperatures.tolist(), 'mse': losses})
        results.append({'target_id': item['target_id'], 'designs': records})
        payload = {'task': args.task, 'guidance': args.guidance, 'maxiter': args.maxiter,
                   'samples': args.samples, 'results': results}
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
        temporary.replace(args.output)
        print(json.dumps(summarize(results)), flush=True)


if __name__ == '__main__':
    main()

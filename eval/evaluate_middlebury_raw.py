"""Cross-dataset disparity upsampling on official Middlebury ThirdSize data.
No fine-tuning or parameter selection on these test scenes. See README.
"""
import argparse
from data.middlebury_raw import MiddleburyRawDataset
from eval.evaluate import evaluate_methods, save_results, summary_markdown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='datasets/Middlebury_raw')
    p.add_argument('--scale', type=int, choices=[4,8,16], default=4)
    p.add_argument('--views', type=int, nargs='+', choices=[1,5], default=[1])
    p.add_argument('--max-samples', type=int)
    p.add_argument('--methods', nargs='+', default=['bicubic','deep'])
    p.add_argument('--out', default='results/middlebury_raw_x4')
    p.add_argument('--final', action='store_true', required=True,
                   help='Acknowledge test-only cross-dataset evaluation; do not tune on it.')
    a = p.parse_args()
    if a.max_samples is not None and a.max_samples <= 0:
        p.error('--max-samples must be positive')
    if len(set(a.views)) != len(a.views):
        p.error('Do not repeat views.')
    ds = MiddleburyRawDataset(a.root, a.scale, tuple(a.views))
    print(f'{len(ds)} RGB/disparity pairs. Units: third-size disparity pixels, NOT metres.')
    res = evaluate_methods(ds, a.methods, a.max_samples)
    res['meta'].update(protocol='Official 2006 ThirdSize, encoded disparity / 3',
                       views=a.views, target='disparity, not metric depth',
                       invalid='original disparity PNG == 0',
                       comparison_note='Separate protocol from prepared Middlebury/SGNet paper tables.')
    print(summary_markdown(res))
    print('saved:', save_results(res, a.out, f'middlebury2006_third_x{a.scale}'))
    failures = res['not_evaluated'] or any(v['n_failed'] for v in res['summary'].values())
    if failures:
        raise SystemExit('Some methods/images failed. Inspect JSON n_ok/n_failed before reporting.')


if __name__ == '__main__':
    main()

from classical.evaluation.ablate_jbu import METRICS, summarize, synthetic_samples
from classical.evaluation.synthetic_data import KINDS, SPLIT_SEEDS


def test_synthetic_splits_use_disjoint_scene_ids():
    seen = set()
    for split, expected in SPLIT_SEEDS.items():
        cases = {(kind, seed) for kind, seed, *_ in synthetic_samples(split, 2)}
        assert cases == {(kind, seed) for kind in KINDS for seed in expected}
        assert cases.isdisjoint(seen)
        seen.update(cases)
    assert set(SPLIT_SEEDS["test"]) == {400, 401, 402}


def test_paired_rmse_summary_resamples_seeds_as_units():
    rows = [
        dict.fromkeys(METRICS) | dict(
            split="test_extension", scale=2, kind=kind, scene=seed,
            method=method, rmse=offset + seed,
        )
        for kind in KINDS
        for seed in range(500, 520)
        for method, offset in (("jbu", 1), ("tuned_jbu", 0))
    ]
    overall = next(row for row in summarize(rows)
                   if row["kind"] == "all" and row["method"] == "tuned_jbu_minus_jbu")
    assert overall["n"] == 20
    assert overall["mean"] == overall["ci95_low"] == overall["ci95_high"] == -1
    assert overall["sd"] == 0

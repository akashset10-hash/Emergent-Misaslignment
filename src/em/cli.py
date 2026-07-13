"""Command-line entry point — the one-click surface.

    em run-stage --config configs/stage2_intervention.yaml
    em run-all   --config configs/base.yaml     # full gated pipeline
    em smoke                                     # end-to-end on the mock backend
    em figures   --store results/measurements.jsonl
    em paper
    em ablations
    em human-rate --generations results/generations.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from em.config import load_config, Config
from em.logging_utils import RunLogger, run_dir_for


def _console_approver(info: dict) -> str:
    print("\n" + "=" * 70)
    print(f"GATE: {info['stage']} — human sign-off required")
    print(f"Recommendation: {info['recommendation']}")
    print(f"Decision packet: {info['packet_dir']}/decision_packet_{info['stage']}.md")
    print("=" * 70)
    try:
        resp = input("approve / iterate / abort > ").strip().lower()
    except EOFError:
        return "approve"
    return resp if resp in {"approve", "iterate", "abort"} else "approve"


def cmd_run_stage(args):
    cfg = load_config(args.config)
    stage = args.stage or cfg.stage
    from em.loop import Pipeline
    p = Pipeline(cfg, interactive=not args.yes, mock=args.mock,
                 approver=_console_approver)
    result = p.run_stage(stage)
    print(f"\n[{stage}] passed={result.passed}\n{result.recommendation}")
    return 0 if result.passed else 1


def cmd_run_all(args):
    cfg = load_config(args.config)
    from em.loop import Pipeline
    p = Pipeline(cfg, interactive=not args.yes, mock=args.mock, approver=_console_approver)
    results = p.run_all()
    print("\n=== pipeline summary ===")
    for s, r in results.items():
        print(f"  {s}: {'PASS' if r.passed else 'HALT'} — {r.recommendation[:90]}")
    return 0


def cmd_smoke(args):
    """End-to-end on the mock backend, no GPU, no heavy deps. Proves the wiring."""
    cfg = load_config(args.config or "configs/base.yaml")
    cfg.run_name = "smoke"
    cfg.backend.kind = "mock"
    cfg.judge.backend = "disabled"
    cfg.seeds.values = [0, 1]
    cfg.gates.require_human_signoff = False
    # The mock backend's toy signal scale differs from a real model's; use
    # mock-scaled gate thresholds so smoke exercises the full pass path.
    cfg.gates.h0_min_logprob_divergence = 0.1
    cfg.gates.direction_min_separation_z = 0.1
    from em.loop import Pipeline
    p = Pipeline(cfg, interactive=False, mock=True)
    results = p.run_all()
    print("\n=== SMOKE summary (mock backend) ===")
    for s, r in results.items():
        print(f"  {s}: passed={r.passed}")
    from em.report import build_paper
    out = build_paper(str(Path(cfg.output_dir) / "measurements.jsonl"))
    print(f"paper draft: {out}")
    return 0


def cmd_figures(args):
    from em.analysis.results_store import ResultsStore
    from em.figures import plots
    from em.provenance import stamp
    figs = plots.save_all_figures(ResultsStore(args.store), args.out,
                                  provenance=stamp("na", 0).as_dict())
    print(f"generated {len(figs)} figures in {args.out}:")
    for f in figs:
        print(f"  {f}")
    return 0


def cmd_paper(args):
    from em.report import build_paper
    out = build_paper(args.store)
    print(f"paper draft written: {out}")
    return 0


def cmd_ablations(args):
    """Run every ablation config (mock unless --config-backed)."""
    from em.loop import Pipeline
    abls = sorted(Path("configs/ablations").glob("*.yaml"))
    for a in abls:
        cfg = load_config(str(a))
        cfg.backend.kind = "mock" if args.mock else cfg.backend.kind
        cfg.gates.require_human_signoff = False
        print(f"\n--- ablation: {a.name} ({cfg.stage}) ---")
        Pipeline(cfg, interactive=False, mock=args.mock).run_stage(cfg.stage)
    return 0


def cmd_human_rate(args):
    import subprocess
    return subprocess.call([sys.executable, "scripts/human_rate.py",
                            "--generations", args.generations, "--out", args.out])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="em", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run-stage"); p.add_argument("--config", required=True)
    p.add_argument("--stage"); p.add_argument("--mock", action="store_true")
    p.add_argument("--yes", action="store_true", help="skip human sign-off"); p.set_defaults(fn=cmd_run_stage)

    p = sub.add_parser("run-all"); p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--mock", action="store_true"); p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_run_all)

    p = sub.add_parser("smoke"); p.add_argument("--config", default=None); p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("figures"); p.add_argument("--store", default="results/measurements.jsonl")
    p.add_argument("--out", default="paper/figures"); p.set_defaults(fn=cmd_figures)

    p = sub.add_parser("paper"); p.add_argument("--store", default="results/measurements.jsonl")
    p.set_defaults(fn=cmd_paper)

    p = sub.add_parser("ablations"); p.add_argument("--mock", action="store_true", default=True)
    p.set_defaults(fn=cmd_ablations)

    p = sub.add_parser("human-rate")
    p.add_argument("--generations", default="results/generations.jsonl")
    p.add_argument("--out", default="results/human_ratings.jsonl"); p.set_defaults(fn=cmd_human_rate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

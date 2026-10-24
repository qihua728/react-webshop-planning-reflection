from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .env import WebShopError
from .evaluate import read_jsonl, summarize, write_summary
from .runner import check_environment, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ReAct ablation experiments for WebShop")
    parser.add_argument(
        "--config",
        default="configs/baseline_frozen.yaml",
        help="Path to YAML config",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run configured experiment episodes")
    run.add_argument("--start", type=int, help="First numeric task index")
    run.add_argument("--count", type=int, help="Number of tasks")
    run.add_argument("--no-resume", action="store_true", help="Run even if a result already exists")

    check = subparsers.add_parser("check-env", help="Fetch one task without calling the LLM")
    check.add_argument("--task-id", default="fixed_0")

    evaluate = subparsers.add_parser("evaluate", help="Recompute metrics from JSONL")
    evaluate.add_argument("--results", help="Override results JSONL path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "run":
        try:
            summary = run_experiment(
                config,
                start_index=args.start,
                count=args.count,
                resume=not args.no_resume if args.no_resume else None,
            )
        except WebShopError as exc:
            print(f"Experiment did not start: {exc}", file=sys.stderr)
            print(
                "Start WebShop, run check-env, and then retry. "
                "No environment-error rows were added.",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "check-env":
        try:
            print(check_environment(config, args.task_id))
        except WebShopError as exc:
            print(f"WebShop environment check failed: {exc}", file=sys.stderr)
            print("Start a local WebShop server and verify webshop.base_url in the config.", file=sys.stderr)
            raise SystemExit(2) from exc
    elif args.command == "evaluate":
        results_path = config.resolve(args.results or config.experiment.results_file)
        summary = summarize(read_jsonl(results_path), method=config.method)
        write_summary(summary, config.resolve(config.experiment.summary_file))
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

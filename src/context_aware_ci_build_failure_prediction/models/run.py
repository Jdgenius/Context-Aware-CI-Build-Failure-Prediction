from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from context_aware_ci_build_failure_prediction.models.baseline import fit as baseline_fit
from context_aware_ci_build_failure_prediction.models.baseline import inference as baseline_inference
from context_aware_ci_build_failure_prediction.models.baseline import sample_sweep as baseline_sample_sweep


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands: dict[str, Callable[[list[str] | None], int]] = {
        "baseline-train": baseline_fit.main,
        "baseline-infer": baseline_inference.main,
        "baseline-sample-sweep": baseline_sample_sweep.main,
    }
    parser = argparse.ArgumentParser(description="Model command runner.")
    parser.add_argument(
        "command",
        choices=sorted(commands),
        help="Command to run.",
    )

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = argv[0]
    if command not in commands:
        parser.error(f"invalid command: {command!r}")

    return commands[command](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

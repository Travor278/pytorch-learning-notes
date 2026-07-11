from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO trainer boundary for mini_llm.")
    parser.add_argument("--dry-run", action="store_true", help="Print the GRPO execution boundary.")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    print("GRPO is intentionally a boundary for now.")
    print("Required next pieces: grouped rollouts, reward scoring, group-normalized advantages, and KL control.")
    print("It should share rollout/eval infrastructure with PPO instead of duplicating a separate pipeline.")


if __name__ == "__main__":
    main()

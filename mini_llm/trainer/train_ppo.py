from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO trainer boundary for mini_llm.")
    parser.add_argument("--dry-run", action="store_true", help="Print the PPO execution boundary.")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    print("PPO is intentionally a boundary for now.")
    print("Required next pieces: rollout sampler, reward model/API, KL reference, value head, and minibatch PPO updates.")
    print("Use DPO first for offline preference data; PPO should come after reward instrumentation exists.")


if __name__ == "__main__":
    main()

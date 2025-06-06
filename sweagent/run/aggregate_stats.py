import argparse
import asyncio
from pathlib import Path

from sweagent.utils.groupings import (
    aggregate_all_stats_and_difficulty,
)


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory containing predictions")
    return parser


def run_from_cli(args: list[str] | None = None) -> None:
    cli_parser = get_cli_parser()
    cli_args = cli_parser.parse_args(args)
    asyncio.run(aggregate_all_stats_and_difficulty(cli_args.output))


if __name__ == "__main__":
    run_from_cli()

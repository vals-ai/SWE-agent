import asyncio
import logging
from pathlib import Path
import json

logger = logging.getLogger("sweagent.utils.groupings")


async def aggregate_all_stats(path: Path) -> None:
    """
    Goes through each valid instance directory and combines the metrics from base metric file into a single dict.

    Finalized dict is saved to a file in the parent directory at the same level that the instance directories exist in

    Args:
        path: path to the directory that has all the instance directories inside
    """
    batch_size = 20
    aggregated_stats: dict[str, float | int | dict[str, float]] = {}

    valid_directories = [
        directory for directory in path.iterdir() if directory.is_dir()
    ]

    if len(valid_directories) == 0:
        raise ValueError(f"No valid directories found at {path.parent}")

    async def process_directory(
        directory: Path,
    ) -> dict[str, float | int | dict[str, float]] | None:
        stats_path = directory / "stats.json"
        if not stats_path.exists():
            logger.warning(f"Stats file not found at {stats_path}")
            return None

        stats: dict[str, float | int | dict[str, float]] = json.load(
            stats_path.read_text()
        )
        return stats

    async def process_batch(directories: list[Path]) -> None:
        tasks = [process_directory(directory) for directory in directories]
        batch_results = await asyncio.gather(*tasks)

        for stats in batch_results:
            if stats is None:
                continue

            for key, value in stats.items():
                if key == "tool_call_definitions":
                    if key not in aggregated_stats:
                        aggregated_stats[key] = {}
                    for tool, count in value.items():
                        aggregated_stats[key][tool] = (
                            aggregated_stats[key].get(tool, 0) + count
                        )
                else:
                    if key not in aggregated_stats:
                        aggregated_stats[key] = 0
                    aggregated_stats[key] += value

    for i in range(0, len(valid_directories), batch_size):
        batch = valid_directories[i : i + batch_size]
        await process_batch(batch)

    stats_path = path.parent / "aggregated_stats.json"
    stats_path.write_text(json.dumps(aggregated_stats, indent=2))

    logger.info(f"Aggregated {len(valid_directories)} stats and saved to {stats_path}")


EASY = "<15 min fix"
MEDIUM = "15 min - 1 hour"
HARD = "1-4 hours"


async def aggregate_based_off_difficulty(path: Path) -> None:
    """
    Goes through each valid instance directory and combines the metrics from base metric file into a single dict.
    """
    batch_size = 20
    difficulty_mapping_path = Path("./difficulty_mappings.json")
    count = {
        EASY: 0,
        MEDIUM: 0,
        HARD: 0,
    }

    assert difficulty_mapping_path.exists(), "Difficulty mapping file not found"

    difficulty_mapping = json.load(difficulty_mapping_path.read_text())

    aggregated_stats: dict[str, float | int | dict[str, float]] = {
        EASY: {},
        MEDIUM: {},
        HARD: {},
    }

    valid_directories = [
        directory for directory in path.iterdir() if directory.is_dir()
    ]

    if len(valid_directories) == 0:
        raise ValueError(f"No valid directories found at {path.parent}")

    async def process_directory(
        directory: Path,
    ) -> tuple[dict[str, float | int | dict[str, float]] | None, str]:
        stats_path = directory / "stats.json"
        if not stats_path.exists():
            logger.warning(f"Stats file not found at {stats_path}")
            return None, difficulty_mapping[directory.name]

        stats: dict[str, float | int | dict[str, float]] = json.load(
            stats_path.read_text()
        )
        return stats, difficulty_mapping[directory.name]

    async def process_batch(directories: list[Path]) -> None:
        tasks = [process_directory(directory) for directory in directories]
        batch_results = await asyncio.gather(*tasks)

        for stats, difficulty in batch_results:
            if stats is None:
                continue

            count[difficulty] += 1

            for key, value in stats.items():
                if key == "tool_call_definitions":
                    if difficulty not in aggregated_stats:
                        aggregated_stats[difficulty] = {}
                    for tool, count in value.items():
                        aggregated_stats[difficulty][tool] = (
                            aggregated_stats[difficulty].get(tool, 0) + count
                        )
                else:
                    if difficulty not in aggregated_stats:
                        aggregated_stats[difficulty] = 0
                    aggregated_stats[difficulty] += value

    for i in range(0, len(valid_directories), batch_size):
        batch = valid_directories[i : i + batch_size]
        await process_batch(batch)

    aggregated_stats["difficulty_count"] = count

    stats_path = path.parent / "difficulty_aggregated_stats.json"
    stats_path.write_text(json.dumps(aggregated_stats, indent=2))

    logger.info(
        f"Aggregated {len(valid_directories)} stats based off difficulty and saved to {stats_path}"
    )

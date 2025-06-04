import asyncio
import logging
from pathlib import Path
import json

from sweagent.utils.log import get_logger

logger = get_logger("merge", emoji="➕")


async def aggregate_all_stats(directories: list[Path], output_path: Path) -> None:
    """
    Goes through each valid instance directory and combines the metrics from base metric file into a single dict.

    Finalized dict is saved to a file in the parent directory at the same level that the instance directories exist in

    Args:
        path: path to the directory that has all the instance directories inside
    """
    batch_size = 20
    aggregated_stats: dict[str, float | int | dict[str, float]] = {}

    async def process_directory(
        directory: Path,
    ) -> dict[str, float | int | dict[str, float]] | None:
        stats_path = directory / "stats.json"
        if not stats_path.exists():
            logger.warning(f"Stats file not found at {stats_path}")
            return None

        try:
            with open(stats_path, "r") as f:
                stats: dict[str, float | int | dict[str, float]] = json.load(f)
            return stats
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {stats_path}: {e}")
            return None

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

    for i in range(0, len(directories), batch_size):
        batch = directories[i : i + batch_size]
        await process_batch(batch)

    output_path.write_text(json.dumps(aggregated_stats, indent=2))

    logger.info(f"Aggregated {len(directories)} stats and saved to {output_path}")


EASY = "<15 min fix"
MEDIUM = "15 min - 1 hour"
HARD = "1-4 hours"
VERY_HARD = ">4 hours"


async def aggregate_based_off_difficulty(
    directories: list[Path], output_path: Path
) -> None:
    """
    Goes through each valid instance directory and combines the metrics from base metric file into a single dict.
    """
    batch_size = 20

    difficulty_mapping_path = Path("./difficulty_mappings.json")

    difficulty_count = {
        EASY: 0,
        MEDIUM: 0,
        HARD: 0,
        VERY_HARD: 0,
    }

    assert (
        difficulty_mapping_path.exists()
    ), f"Difficulty mapping file not found at {difficulty_mapping_path}"

    try:
        mapping_list: list[dict[str, str]] = json.loads(
            difficulty_mapping_path.read_text()
        )
        difficulty_mapping: dict[str, str] = {
            item["instance_id"]: item["difficulty"] for item in mapping_list
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {difficulty_mapping_path}: {e}")
        return

    aggregated_stats: dict[str, float | int | dict[str, float]] = {
        EASY: {},
        MEDIUM: {},
        HARD: {},
    }

    async def process_directory(
        directory: Path,
    ) -> tuple[dict[str, float | int | dict[str, float]] | None, str]:
        stats_path = directory / "stats.json"
        if not stats_path.exists():
            logger.warning(f"Stats file not found at {stats_path}")
            return None, difficulty_mapping[directory.name]

        try:
            with open(stats_path, "r") as f:
                stats: dict[str, float | int | dict[str, float]] = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {stats_path}: {e}")
            return None, difficulty_mapping[directory.name]

        instance_id = directory.name

        difficulty = difficulty_mapping.get(instance_id, None)
        if difficulty is None:
            logger.warning(f"Difficulty not found for {instance_id}")

        return stats, difficulty

    async def process_batch(directories: list[Path]) -> None:
        tasks = [process_directory(directory) for directory in directories]
        batch_results = await asyncio.gather(*tasks)

        for stats, difficulty in batch_results:
            if stats is None:
                continue

            difficulty_count[difficulty] += 1

            aggregated_stats_difficulty = aggregated_stats[difficulty]

            for key, value in stats.items():
                if key == "tool_call_definitions":
                    if key not in aggregated_stats_difficulty:
                        aggregated_stats_difficulty[key] = {}
                    for tool, count in value.items():
                        aggregated_stats_difficulty[key][tool] = (
                            aggregated_stats_difficulty[key].get(tool, 0) + count
                        )

                else:
                    if key not in aggregated_stats_difficulty:
                        aggregated_stats_difficulty[key] = 0
                    aggregated_stats_difficulty[key] += value

    for i in range(0, len(directories), batch_size):
        batch = directories[i : i + batch_size]
        await process_batch(batch)

    aggregated_stats["difficulty_count"] = difficulty_count

    output_path.write_text(json.dumps(aggregated_stats, indent=2))

    logger.info(
        f"Aggregated {len(directories)} stats based off difficulty and saved to {output_path}"
    )

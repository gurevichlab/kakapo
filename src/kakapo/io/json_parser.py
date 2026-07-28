"""JSON parser for full Parsed_rBAN_Record structures.

This module loads atom-level rBAN records serialized by nerpa-ms in JSON format
and converts them to kakapo's Molecule model while retaining the source Parsed_rBAN_Record.
"""

import hashlib
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, NoReturn

import orjson
from pydantic import BaseModel, Field, RootModel

from kakapo.io.yaml_parser import (
    _build_molecule_from_dict,
    _deserialize_molecule,
    _serialize_molecule,
)
from kakapo.models import Molecule
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _log_and_raise(
    msg: str,
    exc_type: type[Exception] = ValueError,
    cause: Exception | None = None,
    level: str = "error",
) -> NoReturn:
    """Log a message at the specified level and raise an exception.

    Args:
        msg: The message to log and use in the exception.
        exc_type: The exception class to raise.
        cause: The optional causing exception to raise from.
        level: Logging level ('error' or 'warning').
    """
    if level == "warning":
        logger.warning(msg)
    else:
        logger.error(msg)

    if cause is not None:
        raise exc_type(msg) from cause
    raise exc_type(msg)


def _load_json_file(json_path: str | Path) -> list | dict:
    """Load and parse a JSON file using orjson.

    Args:
        json_path: Path to the JSON file.

    Returns:
        The parsed JSON data (usually a list or dict).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    json_path = Path(json_path)
    try:
        with open(json_path, "rb") as f:
            try:
                return orjson.loads(f.read())
            except orjson.JSONDecodeError as e:
                _log_and_raise(f"Invalid JSON in {json_path}: {e}", ValueError, e)
    except FileNotFoundError as e:
        _log_and_raise(f"JSON file not found: {json_path}", FileNotFoundError, e)


class NewVariant(BaseModel):
    """Pydantic model representing a single altered variant of a compound."""

    score: float | None = None
    rank: int | None = None
    original_id: str | None = None
    new_record: dict[str, Any]


class BGCVariantItem(BaseModel):
    """Pydantic model representing a BGC compound and its variants."""

    bgc_variant_id: dict[str, Any] = Field(default_factory=dict)
    compound_id: str | None = None
    original_score: float | None = None
    new_variants: dict[str, NewVariant] = Field(default_factory=dict)


class BGCVariantsFile(RootModel[list[BGCVariantItem]]):
    """Root model to parse a list of BGC variant items from a JSON file."""

    pass


def _extract_and_clean_records(json_data: list | dict) -> tuple[list[dict], bool]:
    """Extract new_record dictionaries from BGC variant items.

    Uses Pydantic schemas to validate the structure of the JSON data and
    safely extract the nested new_record dicts.

    Args:
        json_data: Parsed JSON data, either a single BGC dict or, a list of BGC dicts,
    or a dictionary of BGC dicts indexed by IDs.

    Returns:
        A tuple containing:
            - A list of record dictionaries.
            - A boolean indicating if the input root was a list (for return type selection).
    """
    if isinstance(json_data, list):
        generated_candidates = BGCVariantsFile.model_validate(json_data).root
    elif isinstance(json_data, dict) == True:
        generated_candidates = [
            BGCVariantItem.model_validate(item)
            for item in json_data.values()
        ]
    else:
        raise TypeError(f"Expected list or dict, got {type(json_data).__name__}")

    records = []
    for item in generated_candidates:
        for _new_variant_id, variant in item.new_variants.items():
            records.append(variant.new_record)

    is_list = len(generated_candidates) > 1  # I hope, that was the intention
    return records, is_list


def _process_single_record(args: tuple[dict, str]) -> Molecule:
    """Helper function to parse a single record in parallel."""
    item, unique_id = args
    return _build_molecule_from_dict(item, structure_id=unique_id)


def parse_rban_json(  # noqa: C901
    json_path: str | Path, id_counts: dict[str, int] | None = None
) -> Molecule | dict[str, Molecule]:
    """Parse a Parsed_rBAN_Record JSON file and return a kakapo Molecule or a dict of Molecules.

    Args:
        json_path: Path to a JSON file containing BGC variant items with new_record structures.
        id_counts: Optional tracker dictionary mapping compound IDs to their occurrence count
            for suffix-based deduplication.

    Returns:
        A kakapo Molecule instance representing the structure, or a dictionary mapping
        structure IDs to Molecule instances if the file contains a list of BGC items or multiple
        variants.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the file is not valid JSON or fails to parse.
    """
    json_path = Path(json_path)
    logger.info("Parsing BGC variants JSON from %s", json_path)
    if id_counts is None:
        id_counts = {}

    json_data = _load_json_file(json_path)

    if json_data is None:
        _log_and_raise(f"JSON file {json_path} is empty", ValueError, level="warning")

    records, is_list = _extract_and_clean_records(json_data)

    if not records:
        _log_and_raise(
            f"No valid new_variants records found in {json_path}",
            ValueError,
            level="warning" if is_list else "error",
        )

    molecules = {}
    errors = []

    # Pre-calculate unique IDs sequentially to keep it correct and fast
    tasks = []
    for i, item in enumerate(records):
        base_id = item.get("compound_id")
        if not base_id:
            # If compound_id is missing, use a fallback
            metadata = item.get("metadata") or {}
            base_id = metadata.get("name") or f"unknown_{i}"

        count = id_counts.get(base_id, 0)
        if count > 0:
            unique_id = f"{base_id}_{count + 1}"
        else:
            unique_id = base_id
        id_counts[base_id] = count + 1
        tasks.append((item, unique_id))

    if tasks:
        # Determine whether to parse in parallel.
        # Parallel execution is beneficial for >= 5 records.
        num_workers = min(4, os.cpu_count() or 1)
        if len(tasks) > 5 and num_workers > 1:
            logger.info("Parsing JSON records in parallel using %d workers", num_workers)
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_process_single_record, task) for task in tasks]
                for idx, future in enumerate(futures):
                    try:
                        mol = future.result()
                        molecules[mol.structure_id] = mol
                    except Exception as e:
                        logger.warning(
                            "Index %d failed to parse in list from %s: %s",
                            idx,
                            json_path,
                            e,
                        )
                        errors.append(f"Index {idx}: {e}")
        else:
            # Sequential fallback for small lists or single cores
            for idx, task in enumerate(tasks):
                try:
                    mol = _process_single_record(task)
                    molecules[mol.structure_id] = mol
                except Exception as e:
                    if is_list:
                        logger.warning(
                            "Index %d failed to parse in list from %s: %s",
                            idx,
                            json_path,
                            e,
                        )
                        errors.append(f"Index {idx}: {e}")
                    else:
                        _log_and_raise(
                            f"Failed to parse Parsed_rBAN_Record from {json_path}. Error: {e}",
                            ValueError,
                            e,
                        )

    if errors:
        _log_and_raise(
            f"Failed to parse Parsed_rBAN_Record list from {json_path}. Errors:\n"
            + "\n".join(errors),
            ValueError,
        )

    if is_list:
        logger.info("Successfully parsed %d molecules from %s", len(molecules), json_path)
        return molecules

    molecule = list(molecules.values())[0]
    logger.info("Successfully parsed molecule %s from %s", molecule.structure_id, json_path)
    return molecule


def parse_rban_json_batch(  # noqa: C901
    json_paths: Iterable[str | Path],
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Molecule]:
    """Parse multiple rBAN JSON files.

    Args:
        json_paths: Iterable of paths to JSON files.
        cache_dir: Optional path to the cache directory.
        use_cache: If True, check and use the JSON cache for parsed molecules.

    Returns:
        Dictionary mapping structure IDs to Molecule instances.

    Raises:
        ValueError: If any file fails to parse (includes filename in error message).
    """
    molecules = {}
    id_counts = {}
    errors = []
    logger.info("Parsing batch of JSON file(s)...")

    for path in json_paths:
        json_path = Path(path)

        # 1. Check Cache
        cached_mols = None
        cache_path = None
        if use_cache and cache_dir:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                stat_info = json_path.stat()
                mtime = stat_info.st_mtime
                size = stat_info.st_size
                hash_input = f"{json_path.resolve()}_{mtime}_{size}"
                file_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
                cache_path = cache_dir / f"{json_path.name}_{file_hash}.json"

                if cache_path.exists():
                    logger.info("Cache hit. Loading structures from cache %s", cache_path)
                    cached_data = _load_json_file(cache_path)
                    if not isinstance(cached_data, dict):
                        _log_and_raise(
                            f"Cache file {cache_path} does not contain a dictionary structure",
                            TypeError,
                        )

                    cached_mols = {k: _deserialize_molecule(v) for k, v in cached_data.items()}
                    molecules.update(cached_mols)

                    # Update id_counts with base compound_ids
                    for mol_data in cached_data.values():
                        bid = mol_data.get("compound_id", mol_data["structure_id"])
                        id_counts[bid] = id_counts.get(bid, 0) + 1
                    continue
            except Exception as e:
                logger.warning("Failed to load structures from cache: %s", e)

        # 2. Parse JSON from scratch
        try:
            result = parse_rban_json(json_path, id_counts=id_counts)
            if isinstance(result, dict):
                parsed_dict = result
                molecules.update(result)
            else:
                parsed_dict = {result.structure_id: result}
                molecules[result.structure_id] = result

            # 3. Save to Cache
            if use_cache and cache_path:
                try:
                    logger.debug("Saving parsed structures to cache file: %s", cache_path)
                    serialized_data = {k: _serialize_molecule(v) for k, v in parsed_dict.items()}
                    temp_path = cache_path.with_suffix(".tmp")
                    with open(temp_path, "wb") as f:
                        f.write(orjson.dumps(serialized_data, option=orjson.OPT_INDENT_2))
                    temp_path.replace(cache_path)
                except Exception as e:
                    logger.warning("Failed to save structures to cache: %s", e)

        except Exception as e:
            logger.warning("Failed to parse JSON file %s in batch: %s", json_path, e)
            errors.append(f"{json_path}: {e}")

    if errors:
        _log_and_raise(
            "Failed to parse some JSON files:\n" + "\n".join(errors),
            ValueError,
        )

    logger.info("Completed batch load. Loaded %d total molecules.", len(molecules))
    return molecules

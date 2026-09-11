"""Output formatting and writing module for spectrum match results.

Supports exporting results as standard CSV strings or pretty-printed ASCII tables
to files or standard output.
"""

import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional, Union

from tabulate import tabulate

from kakapo.models import MatchResult, Molecule, Spectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _format_modifications_str(modifications: tuple) -> str:
    """Format a tuple of MonomerModification instances into a human-readable string."""
    if not modifications:
        return ""
    parts = []
    for mod in modifications:
        new_name_str = f"->{mod.new_monomer_name}" if mod.new_monomer_name else ""
        formula_str = f", {mod.formula_difference}" if mod.formula_difference else ""
        parts.append(
            f"{mod.initial_monomer_name}[{mod.monomer_idx}]{new_name_str} "
            f"({mod.mass_difference:+.4f}{formula_str})"
        )
    return "; ".join(parts)


def _filter_duplicates(results: list[MatchResult]) -> list[MatchResult]:
    """Filter out duplicate same-score results that are not representatives.

    Args:
        results: A list of MatchResult instances.

    Returns:
        Deduplicated list of MatchResult objects.
    """
    return [
        r
        for r in results
        if r.duplicate_representative is None or r.duplicate_representative == r.result_id
    ]


def filter_top_hits_per_spectrum(results: list[MatchResult], max_hits: int) -> list[MatchResult]:
    """Filter match results to keep at most max_hits top hits per spectrum by score.

    Args:
        results: List of MatchResult objects.
        max_hits: Maximum number of top hits to output per spectrum.

    Returns:
        Filtered list of MatchResult objects.
    """
    if max_hits <= 0:
        return results

    by_spectrum: dict[str, list[MatchResult]] = {}
    for res in results:
        by_spectrum.setdefault(res.spectrum_id, []).append(res)

    filtered: list[MatchResult] = []
    for spec_results in by_spectrum.values():
        sorted_hits = sorted(
            spec_results, key=lambda r: (-r.score, r.structure_id, r.result_id)
        )
        filtered.extend(sorted_hits[:max_hits])

    filtered.sort(key=lambda r: (-r.score, r.spectrum_id, r.structure_id, r.result_id))
    return filtered


def filter_top_n(results: list[MatchResult], top_n: int) -> list[MatchResult]:
    """Filter match results to keep at most overall top_n results by score.

    Args:
        results: List of MatchResult objects.
        top_n: Maximum total top match results to output.

    Returns:
        Filtered list of MatchResult objects.
    """
    if top_n <= 0:
        return results
    return results[:top_n]


def _group_and_sort_by_spectrum(results: list[MatchResult]) -> list[MatchResult]:
    """Group match results by spectrum_id and sort groups by their highest score descending.

    Within each spectrum group, results are sorted by score descending, structure_id, and result_id.
    Groups are sorted by the highest score in the group descending, then spectrum_id.

    Args:
        results: A list of MatchResult instances.

    Returns:
        A list of MatchResult instances grouped by spectrum and sorted.
    """
    if not results:
        return []

    by_spectrum: dict[str, list[MatchResult]] = {}
    for res in results:
        by_spectrum.setdefault(res.spectrum_id, []).append(res)

    sorted_groups: list[tuple[float, str, list[MatchResult]]] = []
    for spec_id, spec_results in by_spectrum.items():
        sorted_hits = sorted(
            spec_results, key=lambda r: (-r.score, r.structure_id, r.result_id)
        )
        highest_score = sorted_hits[0].score if sorted_hits else 0.0
        sorted_groups.append((highest_score, spec_id, sorted_hits))

    sorted_groups.sort(key=lambda item: (-item[0], item[1]))

    grouped_results: list[MatchResult] = []
    for _, _, group in sorted_groups:
        grouped_results.extend(group)

    return grouped_results


def _prepare_results_for_writing(
    results: list[MatchResult],
    max_hits_per_spectrum: Optional[int] = None,
    top_n: Optional[int] = None,
    filter_duplicates: bool = True,
    group_by_spectrum: bool = False,
) -> list[MatchResult]:
    """Prepare results for writing by applying duplicate filtering, max hits, and top N.

    Args:
        results: List of MatchResult objects.
        max_hits_per_spectrum: Optional maximum top hits per spectrum.
        top_n: Optional maximum total top match results.
        filter_duplicates: Whether to filter out non-representative duplicate results first.
        group_by_spectrum: Whether to group results by spectrum and sort groups by max score.

    Returns:
        Filtered list of MatchResult objects.
    """
    filtered = results
    if filter_duplicates:
        filtered = _filter_duplicates(filtered)
    if max_hits_per_spectrum is not None and max_hits_per_spectrum > 0:
        filtered = filter_top_hits_per_spectrum(filtered, max_hits_per_spectrum)
    if top_n is not None and top_n > 0:
        filtered = filter_top_n(filtered, top_n)
    if group_by_spectrum:
        filtered = _group_and_sort_by_spectrum(filtered)
    return filtered


def format_as_csv(
    results: list[MatchResult],
    molecules: Optional[dict[str, Molecule]] = None,
    max_hits_per_spectrum: Optional[int] = None,
    top_n: Optional[int] = None,
) -> str:
    """Format a list of MatchResult objects into a standard CSV string.

    Args:
        results: A list of MatchResult instances to format.
        molecules: Optional dictionary mapping structure IDs to Molecule objects.
        max_hits_per_spectrum: Optional maximum number of top hits to output per spectrum.
        top_n: Optional maximum total top match results to output.

    Returns:
        A standard RFC 4180 compliant CSV string representation.
    """
    results = _prepare_results_for_writing(
        results,
        max_hits_per_spectrum=max_hits_per_spectrum,
        top_n=top_n,
        filter_duplicates=True,
        group_by_spectrum=True,
    )
    logger.debug("Formatting %d match results as CSV", len(results))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    headers = [
        "structure_id",
        "spectrum_id",
        "structure_mass",
        "spectrum_mass",
        "tolerance",
        "score",
        "matched_peaks_count",
        "theoretical_peaks_count",
        "experimental_peaks_count",
        "monomer_count",
        "modifications",
        "nerpa_score",
        "nerpa_rank",
        "nerpa_modifications_count",
    ]
    writer.writerow(headers)

    for result in results:
        mol = molecules.get(result.structure_id) if molecules else None
        nerpa_data = mol.nerpa_ms_data if mol else None
        nerpa_score_str = (
            f"{nerpa_data.score:.3f}" if nerpa_data and nerpa_data.score is not None else ""
        )
        nerpa_rank_str = str(nerpa_data.rank) if nerpa_data and nerpa_data.rank is not None else ""
        nerpa_mods_str = str(len(nerpa_data.old_to_new_mon_map)) if nerpa_data else ""
        writer.writerow(
            [
                result.structure_id,
                result.spectrum_id,
                f"{result.structure_mass:.5f}",
                result.spectrum_mass,
                str(result.tolerance),
                f"{result.score:.3f}",
                result.matched_peaks_count,
                result.theoretical_peaks_count,
                result.experimental_peaks_count,
                result.monomer_count,
                _format_modifications_str(result.modifications),
                nerpa_score_str,
                nerpa_rank_str,
                nerpa_mods_str,
            ]
        )

    return output.getvalue()


def format_as_table(
    results: list[MatchResult],
    molecules: Optional[dict[str, Molecule]] = None,
    max_hits_per_spectrum: Optional[int] = None,
    top_n: Optional[int] = None,
) -> str:
    """Format a list of MatchResult objects into a pretty-printed ASCII table.

    Args:
        results: A list of MatchResult instances to format.
        molecules: Optional dictionary mapping structure IDs to Molecule objects.
        max_hits_per_spectrum: Optional maximum number of top hits to output per spectrum.
        top_n: Optional maximum total top match results to output.

    Returns:
        A string containing the pretty-printed table.
    """
    results = _prepare_results_for_writing(
        results,
        max_hits_per_spectrum=max_hits_per_spectrum,
        top_n=top_n,
        filter_duplicates=True,
        group_by_spectrum=True,
    )
    logger.debug("Formatting %d match results as ASCII table", len(results))

    headers = [
        "Structure ID",
        "Spectrum ID",
        "Structure Mass",
        "Spectrum Mass",
        "Tolerance",
        "Score",
        "Matched Peaks",
        "Theoretical Peaks",
        "Experimental Peaks",
        "Monomer Count",
        "Modifications",
        "NERPA Score",
        "NERPA Rank",
        "NERPA Modifications",
    ]

    data = []
    for result in results:
        mol = molecules.get(result.structure_id) if molecules else None
        nerpa_data = mol.nerpa_ms_data if mol else None
        nerpa_score_str = (
            f"{nerpa_data.score:.3f}" if nerpa_data and nerpa_data.score is not None else ""
        )
        nerpa_rank_str = str(nerpa_data.rank) if nerpa_data and nerpa_data.rank is not None else ""
        nerpa_mods_str = str(len(nerpa_data.old_to_new_mon_map)) if nerpa_data else ""
        data.append(
            [
                result.structure_id,
                result.spectrum_id,
                f"{result.structure_mass:.5f}",
                result.spectrum_mass,
                str(result.tolerance),
                f"{result.score:.3f}",
                result.matched_peaks_count,
                result.theoretical_peaks_count,
                result.experimental_peaks_count,
                result.monomer_count,
                _format_modifications_str(result.modifications),
                nerpa_score_str,
                nerpa_rank_str,
                nerpa_mods_str,
            ]
        )

    return tabulate(
        data,
        headers=headers,
        tablefmt="simple",
        disable_numparse=[2] if data else False,
    )


def mask_to_reversed_binary_string(mask: int, length: int) -> str:
    """Convert a bitmask to a reversed binary string matching the mass vector order.

    Example: mask = 5 (binary 101, monomers 0 and 2 active), length = 4
    returns: "1010"
    """
    chars = []
    for i in range(length):
        if mask & (1 << i):
            chars.append("1")
        else:
            chars.append("0")
    return "".join(chars)


def format_as_json(
    results: list[MatchResult],
    molecules: Optional[dict[str, Molecule]] = None,
    max_hits_per_spectrum: Optional[int] = None,
    top_n: Optional[int] = None,
) -> str:
    """Format a list of MatchResult objects into a standard JSON string.

    Args:
        results: A list of MatchResult instances to format.
        molecules: Optional dictionary mapping structure IDs to Molecule objects.
        max_hits_per_spectrum: Optional maximum number of top hits to output per spectrum.
        top_n: Optional maximum total top match results to output.

    Returns:
        A standard JSON string representation with 2-space indentation.
    """
    filter_duplicates = max_hits_per_spectrum is not None or top_n is not None
    results = _prepare_results_for_writing(
        results,
        max_hits_per_spectrum=max_hits_per_spectrum,
        top_n=top_n,
        filter_duplicates=filter_duplicates,
    )
    logger.debug("Formatting %d match results as JSON", len(results))
    serialized = []
    for r in results:
        matched_peaks_data = []
        for mp in r.matched_peaks:
            mask_str = mask_to_reversed_binary_string(mp.theoretical_fragment_mask, r.monomer_count)
            matched_peaks_data.append(
                {
                    "theoretical_mz": mp.theoretical_mz,
                    "experimental_mz": mp.experimental_mz,
                    "experimental_peak_intensity": mp.experimental_peak_intensity,
                    "experimental_peak_idx": mp.experimental_peak_idx,
                    "charge": mp.charge,
                    "theoretical_fragment_mask": mask_str,
                }
            )

        modifications_data = [
            {
                "monomer_idx": mod.monomer_idx,
                "initial_monomer_name": mod.initial_monomer_name,
                "new_monomer_name": mod.new_monomer_name,
                "mass_difference": mod.mass_difference,
                "formula_difference": mod.formula_difference,
                "initial_monomer_mass": mod.initial_monomer_mass,
            }
            for mod in r.modifications
        ]

        mol = molecules.get(r.structure_id) if molecules else None
        nerpa_data = mol.nerpa_ms_data if mol else None
        nerpa_score_val = (
            round(nerpa_data.score, 3) if nerpa_data and nerpa_data.score is not None else None
        )
        nerpa_rank_val = nerpa_data.rank if nerpa_data else None
        nerpa_mods_val = len(nerpa_data.old_to_new_mon_map) if nerpa_data else None

        serialized.append(
            {
                "result_id": r.result_id,
                "structure_id": r.structure_id,
                "spectrum_id": r.spectrum_id,
                "structure_mass": round(r.structure_mass, 5),
                "spectrum_mass": r.spectrum_mass,
                "tolerance": str(r.tolerance),
                "score": round(r.score, 3),
                "matched_peaks_count": r.matched_peaks_count,
                "theoretical_peaks_count": r.theoretical_peaks_count,
                "experimental_peaks_count": r.experimental_peaks_count,
                "monomer_count": r.monomer_count,
                "unmodified_monomer_mass_vector": list(r.unmodified_monomer_mass_vector),
                "matched_peaks": matched_peaks_data,
                "modifications": modifications_data,
                "nerpa_score": nerpa_score_val,
                "nerpa_rank": nerpa_rank_val,
                "nerpa_modifications_count": nerpa_mods_val,
                "duplicate_representative": r.duplicate_representative,
                "post_modification_ion_masses": r.post_modification_ion_masses,
            }
        )

    return json.dumps(serialized, indent=2)


def write_results(
    results: list[MatchResult],
    output_path: Optional[Union[str, Path]] = None,
    format_type: str = "table",
    molecules: Optional[list[Molecule]] = None,
    max_hits_per_spectrum: Optional[int] = None,
    top_n: Optional[int] = None,
) -> None:
    """Write formatted match results to a file or stdout.

    If output_path is None or "-", outputs to stdout. Otherwise, writes to the
    given file path. If the parent directories do not exist, they are created automatically.

    Args:
        results: A list of MatchResult instances.
        output_path: Path to write results. If None or "-", prints to standard output.
        format_type: Output format ("table", "txt", "csv", or "json"). Case-insensitive.
        molecules: Optional list of Molecule objects to look up metadata.
        max_hits_per_spectrum: Optional maximum number of top hits to output per spectrum.
        top_n: Optional maximum total top match results to output.

    Raises:
        ValueError: If an unsupported format_type is provided.
        OSError: If writing to the file fails.
    """
    logger.debug(
        "Writing %d results to %s with format '%s'",
        len(results),
        output_path if output_path is not None else "stdout",
        format_type,
    )

    molecule_map = {mol.structure_id: mol for mol in molecules} if molecules else {}

    fmt = format_type.lower()
    if fmt in ("table", "txt"):
        formatted_content = format_as_table(
            results,
            molecules=molecule_map,
            max_hits_per_spectrum=max_hits_per_spectrum,
            top_n=top_n,
        )
    elif fmt == "csv":
        formatted_content = format_as_csv(
            results,
            molecules=molecule_map,
            max_hits_per_spectrum=max_hits_per_spectrum,
            top_n=top_n,
        )
    elif fmt == "json":
        formatted_content = format_as_json(
            results,
            molecules=molecule_map,
            max_hits_per_spectrum=max_hits_per_spectrum,
            top_n=top_n,
        )
    else:
        logger.error("Unsupported output format type: %s", format_type)
        raise ValueError(
            f"Unsupported format type: {format_type}. Supported formats are: 'table', 'txt', 'csv',"
            " 'json'."
        )

    if output_path is None or str(output_path) == "-":
        logger.debug("Writing to stdout")
        sys.stdout.write(formatted_content)
        if not formatted_content.endswith("\n"):
            sys.stdout.write("\n")
    else:
        file_path = Path(output_path)
        logger.debug("Writing to file path: %s", file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(formatted_content, encoding="utf-8")
            logger.info("Successfully wrote %d results to %s", len(results), file_path)
        except OSError as e:
            logger.error("Failed to write results to %s: %s", file_path, e)
            raise


def _serialize_molecule(mol: Molecule) -> dict:
    """Serialize a Molecule object into a JSON-serializable dictionary for structure output writing.

    Args:
        mol: The Molecule instance to serialize.

    Returns:
        A dictionary representation of the Molecule.
    """
    return {
        "input_compound_id": mol.structure_id,
        "monomers": {
            str(k): {
                "norine_name": v.norine_name,
                "residue": v.residue,
                "is_methylated": v.is_methylated,
                "chirality": v.chirality.name,
                "atoms": list(v.atoms),
                "mass": v.mass,
            }
            for k, v in mol.monomers.items()
        },
        "bonds": [
            {
                "source_idx": b.source_idx,
                "target_idx": b.target_idx,
                "bond_type": b.bond_type,
                "subtypes": [{"name": s.name, "reversed": s.reversed} for s in b.subtypes],
            }
            for b in mol.bonds
        ],
        "smiles": mol.smiles,
        "atoms": {
            str(k): {
                "name": v.name,
                "hydrogens": v.hydrogens,
            }
            for k, v in mol.atoms.items()
        },
        "atomic_bonds": [
            [
                [edge.atom1, edge.atom2],
                {
                    "arity": ab.arity,
                    "bond_type": ab.bond_type,
                },
            ]
            for edge, ab in mol.atomic_bonds.items()
        ],
        "metadata": {
            "name": mol.metadata.name if mol.metadata else None,
            "smiles": mol.metadata.smiles if mol.metadata else None,
            "origin": mol.metadata.origin if mol.metadata else None,
            "inchikey": mol.metadata.inchikey if mol.metadata else None,
            "source": mol.metadata.source if mol.metadata else None,
        }
        if mol.metadata
        else None,
        "nerpa_ms_data": mol.nerpa_ms_data.to_dict() if mol.nerpa_ms_data else None,
        "duplicated_structure_ids": list(mol.duplicated_structure_ids),
    }


def write_structures(
    molecules: list[Molecule],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write serialized molecules to a file in JSON format.

    If output_path is None, or stdout/stderr/'-', this does not write anything
    to avoid polluting the results stream. If it points to a file, writes to that file.
    If it points to a directory (or has no suffix), writes to 'structures.json' in
    that directory. If output_path is a specific file, writes to 'structures.json' in
    the same parent directory.

    Args:
        molecules: A list of Molecule instances to serialize.
        output_path: Target file path or directory.
    """
    if output_path is None or str(output_path) == "-":
        logger.debug("Stdout output requested, skipping writing structures JSON")
        return

    path = Path(output_path)
    if path.is_dir() or not path.suffix:
        structures_path = path / "structures.json"
    else:
        structures_path = path.parent / "structures.json"

    logger.debug("Writing %d structures to %s", len(molecules), structures_path)

    serialized_data = {mol.structure_id: _serialize_molecule(mol) for mol in molecules}

    try:
        structures_path.parent.mkdir(parents=True, exist_ok=True)
        structures_path.write_text(json.dumps(serialized_data, indent=2), encoding="utf-8")
        logger.info("Successfully wrote %d structures to %s", len(molecules), structures_path)
    except OSError as e:
        logger.error("Failed to write structures to %s: %s", structures_path, e)
        raise


def serialize_spectrum(spec: Spectrum) -> dict:
    """Serialize a Spectrum object into a JSON-serializable dictionary."""
    return {
        "source_file_path": spec.source_file_path,
        "peaks": [
            {"mass": float(m), "intensity": float(i)} for m, i in zip(spec.mzs, spec.intensities)
        ],
        "precursor_mass": spec.precursor_mass,
        "charge": spec.charge,
    }


def write_spectra(
    spectra: list[Spectrum],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write serialized spectra to a file in JSON format.

    If output_path is None, or stdout/stderr/'-', this does not write anything.
    If it points to a file, writes to 'spectra.json' in the same parent directory.
    If it points to a directory (or has no suffix), writes to 'spectra.json' in
    that directory.

    Args:
        spectra: A list of Spectrum instances to serialize.
        output_path: Target file path or directory.
    """
    if output_path is None or str(output_path) == "-":
        logger.debug("Stdout output requested, skipping writing spectra JSON")
        return

    path = Path(output_path)
    if path.is_dir() or not path.suffix:
        spectra_path = path / "spectra.json"
    else:
        spectra_path = path.parent / "spectra.json"

    logger.debug("Writing %d spectra to %s", len(spectra), spectra_path)

    serialized_data = {spec.spectrum_id: serialize_spectrum(spec) for spec in spectra}

    try:
        spectra_path.parent.mkdir(parents=True, exist_ok=True)
        spectra_path.write_text(json.dumps(serialized_data, indent=2), encoding="utf-8")
        logger.info("Successfully wrote %d spectra to %s", len(spectra), spectra_path)
    except OSError as e:
        logger.error("Failed to write spectra to %s: %s", spectra_path, e)
        raise

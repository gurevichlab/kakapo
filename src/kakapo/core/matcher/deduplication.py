"""Deduplication and grouping utilities for spectrum matching results."""

from collections import defaultdict
from dataclasses import replace

from kakapo.models import MatchResult, Tolerance


def _are_mass_lists_equal(masses1: list[float], masses2: list[float], tolerance: Tolerance) -> bool:
    """Check if two sorted mass lists are equal within mass tolerance.

    Args:
        masses1: First list of sorted theoretical m/z masses.
        masses2: Second list of sorted theoretical m/z masses.
        tolerance: Tolerance configuration used for matching.

    Returns:
        True if all masses in both lists match within tolerance, False otherwise.
    """
    if len(masses1) != len(masses2):
        return False
    for m1, m2 in zip(masses1, masses2):
        if not tolerance.is_within(m1, m2):
            return False
    return True


def _cluster_and_update_score_group(
    score_results: list[tuple[int, MatchResult]],
) -> list[tuple[int, MatchResult]]:
    """Cluster same-score results within a single spectrum and set representatives.

    Args:
        score_results: A list of tuples containing original indices and MatchResult
            instances with matching spectrum_id and score.

    Returns:
        A list of tuples containing original indices and updated MatchResult instances
        with duplicate_representative fields populated.
    """
    # Sort by representative priority key:
    # key: (len(modifications), nerpa_score is None, -nerpa_score, structure_id)
    sorted_indexed = sorted(
        score_results,
        key=lambda item: (
            len(item[1].modifications),
            item[1].nerpa_score is None,
            -(item[1].nerpa_score if item[1].nerpa_score is not None else 0.0),
            item[1].structure_id,
        ),
    )

    # Greedy clustering
    groups: list[list[tuple[int, MatchResult]]] = []
    for idx, result in sorted_indexed:
        for group in groups:
            # The representative is the first member of the group
            _, representative = group[0]
            if _are_mass_lists_equal(
                result.post_modification_ion_masses,
                representative.post_modification_ion_masses,
                result.tolerance,
            ):
                group.append((idx, result))
                break
        else:
            groups.append([(idx, result)])

    # Set duplicate_representative
    updated_results = []
    for group in groups:
        rep_idx, representative = group[0]
        rep_updated = replace(representative, duplicate_representative=None)
        updated_results.append((rep_idx, rep_updated))

        for idx, m in group[1:]:
            m_updated = replace(m, duplicate_representative=rep_updated.result_id)
            updated_results.append((idx, m_updated))

    return updated_results


def _deduplicate_results(results: list[MatchResult]) -> list[MatchResult]:
    """Group duplicate same-score results and set representative, preserving original order.

    Priority for representative selection:
    1. Fewer modifications.
    2. Higher nerpa_score.
    3. Lower structure ID (lexicographically).

    Args:
        results: List of MatchResult objects to deduplicate.

    Returns:
        A list of MatchResult objects with duplicate_representative set.
    """
    # Associate each result with its original index
    indexed_results = list(enumerate(results))

    # Group by spectrum_id
    by_spec = defaultdict(list)
    for idx, r in indexed_results:
        by_spec[r.spectrum_id].append((idx, r))

    updated_indexed_results = []
    for spec_results in by_spec.values():
        # Group by kakapo_score
        by_score = defaultdict(list)
        for idx, r in spec_results:
            by_score[r.kakapo_score].append((idx, r))

        for score_results in by_score.values():
            updated_indexed_results.extend(_cluster_and_update_score_group(score_results))

    # Sort back by original index to preserve original order
    updated_indexed_results.sort(key=lambda item: item[0])
    return [r for _, r in updated_indexed_results]

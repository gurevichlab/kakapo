"""YAML parser for rBAN (Retrosynthetic Complexity-Based Aromatic Nitro) structures.

This module handles:
- Loading rBAN topology details from YAML files
- Converting nerpa-ms rBAN structures to kakapo's Molecule model
- Validating structural data integrity
"""

import yaml
from monomer_names_helper import Chirality as NerpaChirality
from rban_parsing.nrp_variant_types import NRP_Variant
from rban_parsing.rban_monomer import rBAN_Monomer

from kakapo.models import Chirality, Cycle, Molecule, Monomer


def _convert_chirality(nerpa_chirality: NerpaChirality) -> Chirality:
    """Convert nerpa-ms Chirality enum to kakapo Chirality enum.

    Args:
        nerpa_chirality: Chirality from nerpa-ms (Chirality.L, .D, or .UNKNOWN).

    Returns:
        Equivalent Chirality in kakapo model.
    """
    if nerpa_chirality == NerpaChirality.L:
        return Chirality.L
    elif nerpa_chirality == NerpaChirality.D:
        return Chirality.D
    else:
        return Chirality.UNKNOWN


def _rban_monomer_to_kakapo(rban_mon: rBAN_Monomer) -> Monomer:
    """Convert a nerpa-ms rBAN_Monomer to a kakapo Monomer.

    This adapter function bridges the two data models, extracting the fields
    kakapo cares about and discarding nerpa-ms-specific details.

    Args:
        rban_mon: An rBAN_Monomer instance from nerpa-ms.

    Returns:
        A kakapo Monomer instance.
    """
    return Monomer(
        norine_name=str(rban_mon.rban_name),
        residue=str(rban_mon.residue),
        is_methylated=rban_mon.methylated,
        chirality=_convert_chirality(rban_mon.chirality),
    )


def _extract_monomers_from_variant(variant: NRP_Variant) -> list[Monomer]:
    """Extract all monomers from an NRP_Variant in sequence order.

    For fragmented structures (e.g., linear peptides split by cycles),
    this concatenates all fragments in order, preserving linear structure.

    Args:
        variant: An NRP_Variant from nerpa-ms.

    Returns:
        An ordered list of kakapo Monomers.
    """
    all_monomers = []

    # Add monomers from all fragments in order
    for fragment in variant.fragments:
        for rban_mon in fragment.monomers:
            all_monomers.append(_rban_monomer_to_kakapo(rban_mon))

    # Add isolated monomers (e.g., unknowns not in main structure)
    for rban_mon in variant.isolated_unknown_monomers:
        all_monomers.append(_rban_monomer_to_kakapo(rban_mon))

    return all_monomers


def _extract_cycles_from_variant(variant: NRP_Variant, total_monomer_count: int) -> list[Cycle]:
    """Extract cyclic motifs from an NRP_Variant.

    Identifies fragments that represent cyclic structures and returns
    them as Cycle objects referencing monomer indices in the full sequence.

    Args:
        variant: An NRP_Variant from nerpa-ms.
        total_monomer_count: Total number of monomers in the variant
            (from main fragments + isolated unknowns).

    Returns:
        A list of Cycle objects representing cyclic motifs.

    Note:
        This function attempts to detect cyclic fragments using common
        nerpa-ms attributes (is_cyclic, cyclic, etc.). If the actual
        nerpa-ms API differs, this logic may need adjustment.
    """
    cycles = []
    monomer_index = 0

    # Process each fragment, tracking which monomers belong to cycles
    for fragment in variant.fragments:
        fragment_monomer_count = len(fragment.monomers)
        fragment_indices = tuple(range(monomer_index, monomer_index + fragment_monomer_count))

        # Check if fragment is cyclic (using nerpa-ms attributes)
        is_cyclic = getattr(fragment, "is_cyclic", False)

        if is_cyclic and fragment_monomer_count >= 2:
            cycles.append(Cycle(monomer_indices=fragment_indices))

        monomer_index += fragment_monomer_count

    # Note: isolated_unknown_monomers are typically not cyclic,
    # so we skip them when constructing cycles.

    return cycles


def parse_rban_yaml(yaml_path: str) -> Molecule:
    """Parse an rBAN YAML file and return a kakapo Molecule.

    This function:
    1. Loads YAML from the given file path
    2. Parses it into nerpa-ms NRP_Variant structures
    3. Extracts monomers and cyclic topology
    4. Converts to kakapo's Molecule model
    5. Validates the result

    Args:
        yaml_path: Path to the YAML file containing rBAN structure data.

    Returns:
        A kakapo Molecule instance representing the structure with
        cyclic topology information preserved.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the YAML does not contain valid rBAN structure data.
        AssertionError: If the parsed structure has no monomers.
    """
    # Load YAML file
    try:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"YAML file not found: {yaml_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e

    if yaml_data is None:
        raise ValueError(f"YAML file {yaml_path} is empty")

    # Parse into nerpa-ms NRP_Variant
    try:
        variant = NRP_Variant.from_yaml_dict(yaml_data)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(
            f"Failed to parse rBAN structure from {yaml_path}. "
            f"Expected NRP_Variant format. Error: {e}"
        ) from e

    # Extract monomers and cyclic topology
    monomers = _extract_monomers_from_variant(variant)

    if not monomers:
        raise ValueError(f"No monomers found in YAML file {yaml_path}")

    cycles = _extract_cycles_from_variant(variant, len(monomers))

    # Create molecule with structure ID, monomers, cycles, and optional SMILES
    molecule = Molecule(
        structure_id=variant.nrp_variant_id.nrp_id,
        monomers=tuple(monomers),
        cycles=tuple(cycles),
        smiles=variant.metadata.smiles,
    )

    return molecule


def parse_rban_yaml_batch(yaml_paths: list[str]) -> dict[str, Molecule]:
    """Parse multiple rBAN YAML files.

    Args:
        yaml_paths: List of paths to YAML files.

    Returns:
        Dictionary mapping structure IDs to Molecule instances.

    Raises:
        ValueError: If any file fails to parse (includes filename in error message).
    """
    molecules = {}
    errors = []

    for yaml_path in yaml_paths:
        try:
            molecule = parse_rban_yaml(yaml_path)
            molecules[molecule.structure_id] = molecule
        except Exception as e:
            errors.append(f"{yaml_path}: {e}")

    if errors:
        error_msg = "Failed to parse some YAML files:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return molecules

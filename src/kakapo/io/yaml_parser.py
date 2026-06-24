"""YAML parser for full Parsed_rBAN_Record structures.

This module loads atom-level rBAN records serialized by nerpa-ms and converts
them to kakapo's Molecule model while retaining the source Parsed_rBAN_Record.
"""

from collections.abc import Iterable
from pathlib import Path

import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader
from rban_parsing.rban_parser import AtomInfo, MonomerInfo, Parsed_rBAN_Record

from kakapo.constants import ATOMIC_MASSES
from kakapo.io.bond_classifier import classify_single_bond
from kakapo.models import (
    Atom,
    AtomicBond,
    AtomicEdge,
    Bond,
    Chirality,
    Molecule,
    Monomer,
)
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _calculate_monomer_exact_mass(
    monomer_info: MonomerInfo, atoms_dict: dict[int, AtomInfo]
) -> float | None:
    """Calculate the exact monoisotopic mass of a monomer from its atomic formula.

    Sums the exact isotopic masses of all non-hydrogen and hydrogen atoms belonging to the monomer.

    Args:
        monomer_info: The MonomerRecord parsing structure from the rBAN parser.
        atoms_dict: A dictionary mapping atom IDs to AtomRecord instances.

    Returns:
        The precalculated exact monoisotopic mass of the monomer, or None if the mass is zero.

    Raises:
        ValueError: If an unknown chemical element is encountered.
    """
    logger.debug("Calculating exact mass for monomer '%s'", monomer_info.name)
    total_mass = 0.0
    for atom_id in monomer_info.atoms:
        atom_info = atoms_dict.get(atom_id)
        if atom_info is not None:
            symbol = atom_info.name
            if symbol in ATOMIC_MASSES:
                total_mass += ATOMIC_MASSES[symbol]
            else:
                logger.error("Unknown element '%s' for atom %d", symbol, atom_id)
                raise ValueError(f"Unknown element '{symbol}' for atom {atom_id}")
            if atom_info.hydrogens > 0:
                total_mass += atom_info.hydrogens * ATOMIC_MASSES.get("H", 1.007825)
    logger.debug("Calculated mass for monomer '%s': %s", monomer_info.name, total_mass)
    return total_mass if total_mass > 0 else None


def _map_and_classify_bonds(
    record: Parsed_rBAN_Record,
    monomers: dict[int, Monomer],
    atoms: dict[int, Atom],
    atomic_bonds: dict[AtomicEdge, AtomicBond],
) -> list[Bond]:
    """Map monomer bonds and dynamically classify their subtypes directly using loaded rules.

    Args:
        record: The source Parsed_rBAN_Record.
        monomers: Dictionary mapping monomer IDs to Monomer instances.
        atoms: Dictionary mapping atom IDs to Atom instances.
        atomic_bonds: Dictionary mapping AtomicEdge instances to AtomicBond instances.

    Returns:
        List of Bond instances with classified subtypes.
    """
    bonds = []
    logger.debug("Mapping and classifying %d monomer bonds", len(record.monomer_bonds))
    for (u, v), bond_info_list in record.monomer_bonds.items():
        for bond_info in bond_info_list:
            bond_type = bond_info.atomic_edge.bond_type

            # Get the connecting atoms mapped to their monomer IDs
            mon_to_atom = bond_info.monomer_to_atom

            # Call the classifier method directly
            subtypes = classify_single_bond(
                source_monomer_idx=u,
                target_monomer_idx=v,
                monomer_a=monomers[u],
                monomer_b=monomers[v],
                mon_to_atom=mon_to_atom,
                atom_info_dict=atoms,
                atomic_edges=atomic_bonds,
            )

            bonds.append(
                Bond(
                    source_idx=u,
                    target_idx=v,
                    bond_type=bond_type,
                    subtypes=tuple(subtypes),
                )
            )
    return bonds


def _build_molecule_from_parsed_record(
    record: Parsed_rBAN_Record, structure_id: str | None = None
) -> Molecule:
    """Convert a Parsed_rBAN_Record into a kakapo Molecule instance.

    Maps the parsed record's atoms, monomers, and atomic bonds to Molecule components,
    precalculates the exact monoisotopic mass of each monomer using the atomic composition, and
    dynamically classifies inter-monomer bond subtypes using subgraph isomorphism.

    Args:
        record: The source Parsed_rBAN_Record parsed from YAML.
        structure_id: Optional string to override the record's default compound_id.

    Returns:
        A fully constructed Molecule instance with enriched properties and classified bonds.
    """
    # 1. Map atoms to Atom
    atoms = {
        atom_id: Atom(name=atom_info.name, hydrogens=atom_info.hydrogens)
        for atom_id, atom_info in record.atoms.items()
    }

    # 2. Map monomers to Monomer using original keys
    monomers = {}
    for mon_idx, mon_info in record.monomers.items():
        residue = (
            str(mon_info.nerpa_core) if mon_info.nerpa_core is not None else str(mon_info.name)
        )
        is_methylated = bool(mon_info.methylated) if mon_info.methylated is not None else False

        # Calculate exact mass from atomic composition
        try:
            mass = _calculate_monomer_exact_mass(mon_info, record.atoms)
        except ValueError as e:
            logger.warning(
                "Failed to calculate exact mass for monomer %s: %s",
                mon_info.name,
                e,
            )
            mass = None

        monomers[mon_idx] = Monomer(
            norine_name=str(mon_info.name),
            residue=residue,
            is_methylated=is_methylated,
            chirality=Chirality[mon_info.chirality.name],
            atoms=tuple(mon_info.atoms),
            mass=mass,
        )

    # 3. Map atomic bonds
    atomic_bonds = {}
    for (u, v), edge_info in record.atomic_bonds.items():
        atomic_bonds[AtomicEdge(u, v)] = AtomicBond(
            arity=edge_info.arity,
            bond_type=edge_info.bond_type,
        )

    # 4. Map monomer bonds and dynamically classify subtypes directly
    bonds = _map_and_classify_bonds(record, monomers, atoms, atomic_bonds)

    logger.debug(
        "Molecule %s built: %d atoms, %d monomers, %d atomic bonds, %d monomer bonds",
        record.compound_id,
        len(atoms),
        len(monomers),
        len(atomic_bonds),
        len(bonds),
    )

    # 5. Return the final Molecule
    return Molecule(
        structure_id=structure_id if structure_id is not None else record.compound_id,
        monomers=monomers,
        bonds=tuple(bonds),
        smiles=record.metadata.smiles if record.metadata else None,
        atoms=atoms,
        atomic_bonds=atomic_bonds,
        metadata=record.metadata,
        source_parsed_record=record,
    )


def parse_rban_yaml(
    yaml_path: str | Path, id_counts: dict[str, int] | None = None
) -> Molecule | dict[str, Molecule]:
    """Parse a Parsed_rBAN_Record YAML file and return a kakapo Molecule or a dict of Molecules.

    Args:
        yaml_path: Path to a YAML file containing one Parsed_rBAN_Record dict or a list of them.
        id_counts: Optional tracker dictionary mapping compound IDs to their occurrence count
            for suffix-based deduplication.

    Returns:
        A kakapo Molecule instance representing the structure with
        cyclic topology information preserved, or a dictionary mapping structure IDs to Molecule
        instances if the file contains a list of records.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the YAML is not a valid Parsed_rBAN_Record payload.
    """
    yaml_path = Path(yaml_path)
    logger.info("Parsing rBAN YAML from %s", yaml_path)
    if id_counts is None:
        id_counts = {}
    # Load YAML file
    try:
        with open(yaml_path) as f:
            yaml_data = yaml.load(f, Loader=SafeLoader)
    except FileNotFoundError as e:
        logger.error("YAML file not found: %s", yaml_path)
        raise FileNotFoundError(f"YAML file not found: {yaml_path}") from e
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in %s: %s", yaml_path, e)
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e

    if yaml_data is None:
        logger.warning("YAML file %s is empty", yaml_path)
        raise ValueError(f"YAML file {yaml_path} is empty")

    is_list = isinstance(yaml_data, list)
    if is_list and not yaml_data:
        logger.warning("YAML file %s contains an empty list", yaml_path)
        raise ValueError(f"YAML file {yaml_path} contains an empty list")

    records = yaml_data if is_list else [yaml_data]
    molecules = {}
    errors = []

    for i, item in enumerate(records):
        try:
            parsed_record = Parsed_rBAN_Record.from_dict(item)
            base_id = parsed_record.compound_id
            count = id_counts.get(base_id, 0)
            if count > 0:
                unique_id = f"{base_id}_{count + 1}"
            else:
                unique_id = base_id
            id_counts[base_id] = count + 1

            molecule = _build_molecule_from_parsed_record(parsed_record, structure_id=unique_id)
            molecules[unique_id] = molecule
        except (KeyError, ValueError, TypeError) as e:
            if is_list:
                logger.warning("Index %d failed to parse in list from %s: %s", i, yaml_path, e)
                errors.append(f"Index {i}: {e}")
            else:
                logger.error("Failed to parse Parsed_rBAN_Record from %s: %s", yaml_path, e)
                raise ValueError(
                    f"Failed to parse Parsed_rBAN_Record from {yaml_path}. Error: {e}"
                ) from e

    if errors:
        logger.error(
            "Failed to parse some Parsed_rBAN_Records from %s: %s",
            yaml_path,
            errors,
        )
        raise ValueError(
            f"Failed to parse Parsed_rBAN_Record list from {yaml_path}. Errors:\n"
            + "\n".join(errors)
        )

    if is_list:
        logger.info("Successfully parsed %d molecules from %s", len(molecules), yaml_path)
        return molecules

    molecule = list(molecules.values())[0]
    logger.info("Successfully parsed molecule %s from %s", molecule.structure_id, yaml_path)
    return molecule


def parse_rban_yaml_batch(yaml_paths: Iterable[str | Path]) -> dict[str, Molecule]:
    """Parse multiple rBAN YAML files.

    Args:
        yaml_paths: Iterable of paths to YAML files.

    Returns:
        Dictionary mapping structure IDs to Molecule instances.

    Raises:
        ValueError: If any file fails to parse (includes filename in error message).
    """
    molecules = {}
    id_counts = {}
    errors = []
    logger.info("Parsing batch of YAML file(s)...")

    for path in yaml_paths:
        yaml_path = Path(path)
        try:
            result = parse_rban_yaml(yaml_path, id_counts=id_counts)
            if isinstance(result, dict):
                molecules.update(result)
            else:
                molecules[result.structure_id] = result
        except Exception as e:
            logger.warning("Failed to parse YAML file %s in batch: %s", yaml_path, e)
            errors.append(f"{yaml_path}: {e}")

    if errors:
        logger.error("Failed to parse some YAML files: %s", errors)
        error_msg = "Failed to parse some YAML files:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    logger.info("Completed batch load. Loaded %d total molecules.", len(molecules))
    return molecules

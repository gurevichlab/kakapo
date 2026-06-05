"""YAML parser for full Parsed_rBAN_Record structures.

This module loads atom-level rBAN records serialized by nerpa-ms and converts
them to kakapo's Molecule model while retaining the source Parsed_rBAN_Record.
"""

import yaml
from rban_parsing.rban_parser import Parsed_rBAN_Record

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


def _calculate_monomer_exact_mass(monomer_info, atoms_dict) -> float | None:
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
    total_mass = 0.0
    for atom_id in monomer_info.atoms:
        atom_info = atoms_dict.get(atom_id)
        if atom_info is not None:
            symbol = atom_info.name
            if symbol in ATOMIC_MASSES:
                total_mass += ATOMIC_MASSES[symbol]
            else:
                raise ValueError(f"Unknown element '{symbol}' for atom {atom_id}")
            if atom_info.hydrogens > 0:
                total_mass += atom_info.hydrogens * ATOMIC_MASSES.get("H", 1.007825)
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


def _build_molecule_from_parsed_record(record: Parsed_rBAN_Record) -> Molecule:
    """Convert a Parsed_rBAN_Record into a kakapo Molecule instance.

    Maps the parsed record's atoms, monomers, and atomic bonds to Molecule components,
    precalculates the exact monoisotopic mass of each monomer using the atomic composition, and
    dynamically classifies inter-monomer bond subtypes using subgraph isomorphism.

    Args:
        record: The source Parsed_rBAN_Record parsed from YAML.

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
        except ValueError:
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

    # 5. Return the final Molecule
    return Molecule(
        structure_id=record.compound_id,
        monomers=monomers,
        bonds=tuple(bonds),
        smiles=record.metadata.smiles if record.metadata else None,
        atoms=atoms,
        atomic_bonds=atomic_bonds,
        metadata=record.metadata,
        source_parsed_record=record,
    )


def parse_rban_yaml(yaml_path: str) -> Molecule | dict[str, Molecule]:
    """Parse a Parsed_rBAN_Record YAML file and return a kakapo Molecule or a dict of Molecules.

    Args:
        yaml_path: Path to a YAML file containing one Parsed_rBAN_Record dict or a list of them.

    Returns:
        A kakapo Molecule instance representing the structure with
        cyclic topology information preserved, or a dictionary mapping structure IDs to Molecule
        instances if the file contains a list of records.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the YAML is not a valid Parsed_rBAN_Record payload.
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

    if isinstance(yaml_data, list):
        if not yaml_data:
            raise ValueError(f"YAML file {yaml_path} contains an empty list")

        molecules = {}
        errors = []
        for i, item in enumerate(yaml_data):
            try:
                parsed_record = Parsed_rBAN_Record.from_dict(item)
                molecule = _build_molecule_from_parsed_record(parsed_record)
                molecules[molecule.structure_id] = molecule
            except (KeyError, ValueError, TypeError) as e:
                errors.append(f"Index {i}: {e}")

        if errors:
            raise ValueError(
                f"Failed to parse Parsed_rBAN_Record list from {yaml_path}. Errors:\n"
                + "\n".join(errors)
            )

        return molecules

    try:
        parsed_record = Parsed_rBAN_Record.from_dict(yaml_data)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse Parsed_rBAN_Record from {yaml_path}. Error: {e}") from e

    return _build_molecule_from_parsed_record(parsed_record)


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
            result = parse_rban_yaml(yaml_path)
            if isinstance(result, dict):
                molecules.update(result)
            else:
                molecules[result.structure_id] = result
        except Exception as e:
            errors.append(f"{yaml_path}: {e}")

    if errors:
        error_msg = "Failed to parse some YAML files:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return molecules

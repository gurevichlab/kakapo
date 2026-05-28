"""YAML parser for full Parsed_rBAN_Record structures.

This module loads atom-level rBAN records serialized by nerpa-ms and converts
them to kakapo's Molecule model while retaining the source Parsed_rBAN_Record.
"""

import networkx as nx
import yaml
from rban_parsing.rban_parser import Parsed_rBAN_Record

from kakapo.models import Bond, Chirality, Cycle, Molecule, Monomer


def _extract_cycles_from_monomer_edges(
    monomer_indices: list[int], edges: list[tuple[int, int]]
) -> list[Cycle]:
    """Infer cyclic motifs from an undirected monomer-level graph."""
    if not monomer_indices:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(monomer_indices)
    graph.add_edges_from(edges)

    cycles = []
    for cycle_nodes in nx.cycle_basis(graph):
        if len(cycle_nodes) >= 2:
            cycles.append(Cycle(monomer_indices=tuple(cycle_nodes)))
    return cycles


def _extract_ring_edges_from_cycles(cycle_nodes: list[int]) -> set[tuple[int, int]]:
    """Convert cycle node order to undirected canonical edge pairs."""
    cycle_edges = set()
    if len(cycle_nodes) < 2:
        return cycle_edges

    for i in range(len(cycle_nodes)):
        u = cycle_nodes[i]
        v = cycle_nodes[(i + 1) % len(cycle_nodes)]
        cycle_edges.add((u, v) if u <= v else (v, u))
    return cycle_edges


def _build_molecule_from_parsed_record(record: Parsed_rBAN_Record) -> Molecule:
    """Convert Parsed_rBAN_Record into a kakapo Molecule, retaining source record."""
    ordered_monomer_indices = sorted(record.monomers.keys())
    monomers = []
    for mon_idx in ordered_monomer_indices:
        mon_info = record.monomers[mon_idx]
        residue = (
            str(mon_info.nerpa_core) if mon_info.nerpa_core is not None else str(mon_info.name)
        )
        is_methylated = bool(mon_info.methylated) if mon_info.methylated is not None else False
        monomers.append(
            Monomer(
                norine_name=str(mon_info.name),
                residue=residue,
                is_methylated=is_methylated,
                chirality=Chirality[mon_info.chirality.name],
            )
        )

    graph_index_of = {mon_idx: i for i, mon_idx in enumerate(ordered_monomer_indices)}
    indexed_edges = []
    for u, v in record.monomer_bonds.keys():
        if u in graph_index_of and v in graph_index_of:
            indexed_edges.append((graph_index_of[u], graph_index_of[v]))

    cycles = _extract_cycles_from_monomer_edges(
        list(range(len(ordered_monomer_indices))),
        indexed_edges,
    )
    ring_edges = set()
    for cycle in cycles:
        ring_edges.update(_extract_ring_edges_from_cycles(list(cycle.monomer_indices)))

    bonds = []
    for u, v in indexed_edges:
        canonical = (u, v) if u <= v else (v, u)

        bond_info_list = record.monomer_bonds.get(canonical, [])
        bond_type = bond_info_list[0].atomic_edge.bond_type if bond_info_list else "PEPTIDE"
        bonds.append(
            Bond(
                source_idx=u,
                target_idx=v,
                bond_type=bond_type,
                is_ring_bond=canonical in ring_edges,
            )
        )

    return Molecule(
        structure_id=record.compound_id,
        monomers=tuple(monomers),
        bonds=tuple(bonds),
        cycles=tuple(cycles),
        smiles=record.metadata.smiles,
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

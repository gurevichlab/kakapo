"""YAML parser for rBAN molecular structures.

Provides functions to parse single or batch YAML files containing rBAN structure
records, converting them into Kakapo's native Molecule domain models with dynamic
bond classification, exact monomer mass calculation, parallel parsing, and disk caching.
"""

import hashlib
import json
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from rdkit import Chem
from yaml import CSafeLoader as SafeLoader

from kakapo.constants import ATOMIC_MASSES
from kakapo.io.bond_classifier import classify_single_bond
from kakapo.models import (
    Atom,
    AtomicBond,
    AtomicEdge,
    Bond,
    BondSubtype,
    Chirality,
    Metadata,
    Molecule,
    Monomer,
)
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_single_yaml_block(yaml_str: str) -> list | dict:
    """Parse a single YAML block string (for parallel parsing)."""
    return yaml.load(yaml_str, Loader=SafeLoader)


def _serialize_molecule(mol: Molecule) -> dict:
    """Serialize a Molecule object into a JSON-serializable dictionary."""
    return {
        "structure_id": mol.structure_id,
        "compound_id": mol.structure_id,
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
    }


def _deserialize_molecule(data: dict) -> Molecule:
    """Deserialize a JSON-serializable dictionary back into a Molecule object."""
    monomers = {
        int(k): Monomer(
            norine_name=v["norine_name"],
            residue=v["residue"],
            is_methylated=v["is_methylated"],
            chirality=Chirality[v["chirality"]],
            atoms=tuple(v["atoms"]),
            mass=v["mass"],
        )
        for k, v in data["monomers"].items()
    }
    bonds = tuple(
        Bond(
            source_idx=b["source_idx"],
            target_idx=b["target_idx"],
            bond_type=b["bond_type"],
            subtypes=tuple(
                BondSubtype(name=s["name"], reversed=s["reversed"]) for s in b.get("subtypes", [])
            ),
        )
        for b in data.get("bonds", [])
    )
    atoms = {
        int(k): Atom(name=v["name"], hydrogens=v["hydrogens"])
        for k, v in data.get("atoms", {}).items()
    }
    atomic_bonds = {
        AtomicEdge(edge[0], edge[1]): AtomicBond(arity=ab["arity"], bond_type=ab["bond_type"])
        for edge, ab in data.get("atomic_bonds", [])
    }
    metadata = Metadata.from_dict(data.get("metadata"))
    smiles = data.get("smiles")
    if smiles is None and atoms:
        smiles = generate_smiles_from_atoms(atoms, atomic_bonds)
    return Molecule(
        structure_id=data["structure_id"],
        monomers=monomers,
        bonds=bonds,
        smiles=smiles,
        atoms=atoms,
        atomic_bonds=atomic_bonds,
        metadata=metadata,
    )


def generate_smiles_from_atoms(
    atoms: dict[int, Atom],
    atomic_bonds: dict[AtomicEdge, AtomicBond],
) -> str | None:
    """Generate a canonical SMILES string from atomic nodes and bond edges using RDKit.

    Args:
        atoms: Dictionary mapping atom IDs to Atom instances.
        atomic_bonds: Dictionary mapping AtomicEdge instances to AtomicBond instances.

    Returns:
        Canonical SMILES string if atoms are present, or None if atoms dictionary is empty.
    """
    if not atoms:
        return None

    rw_mol = Chem.RWMol()
    id_to_rdkit_idx = {}

    for atom_id in sorted(atoms.keys()):
        atom = atoms[atom_id]
        rd_atom = Chem.Atom(atom.name)
        if atom.hydrogens > 0:
            rd_atom.SetNumExplicitHs(atom.hydrogens)
        idx = rw_mol.AddAtom(rd_atom)
        id_to_rdkit_idx[atom_id] = idx

    for edge, ab in atomic_bonds.items():
        idx_u = id_to_rdkit_idx.get(edge.atom1)
        idx_v = id_to_rdkit_idx.get(edge.atom2)
        if idx_u is not None and idx_v is not None:
            b_type = Chem.BondType.SINGLE
            bond_type_str = str(ab.bond_type).upper() if ab.bond_type else ""
            arity_str = str(ab.arity)

            if bond_type_str == "DOUBLE" or arity_str == "2":
                b_type = Chem.BondType.DOUBLE
            elif bond_type_str == "TRIPLE" or arity_str == "3":
                b_type = Chem.BondType.TRIPLE
            elif bond_type_str == "AROMATIC" or arity_str in ("ar", "1.5"):
                b_type = Chem.BondType.AROMATIC

            rw_mol.AddBond(idx_u, idx_v, b_type)

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        logger.warning("Failed to sanitize molecule for SMILES generation: %s", e)

    return Chem.MolToSmiles(mol)


def _calculate_monomer_exact_mass(
    monomer_atoms: tuple[int, ...] | list[int],
    monomer_name: str,
    atoms_dict: dict[int, Atom],
) -> float | None:
    """Calculate the exact monoisotopic mass of a monomer from its atomic formula.

    Sums the exact isotopic masses of all non-hydrogen and hydrogen atoms belonging to the monomer.

    Args:
        monomer_atoms: Iterable of atom IDs belonging to the monomer.
        monomer_name: Name of the monomer for logging.
        atoms_dict: A dictionary mapping atom IDs to Atom instances.

    Returns:
        The precalculated exact monoisotopic mass of the monomer, or None if the mass is zero.

    Raises:
        ValueError: If an unknown chemical element is encountered.
    """
    logger.debug("Calculating exact mass for monomer '%s'", monomer_name)
    total_mass = 0.0
    for atom_id in monomer_atoms:
        atom = atoms_dict.get(atom_id)
        if atom is not None:
            symbol = atom.name
            if symbol in ATOMIC_MASSES:
                total_mass += ATOMIC_MASSES[symbol]
            else:
                logger.error("Unknown element '%s' for atom %d", symbol, atom_id)
                raise ValueError(f"Unknown element '{symbol}' for atom {atom_id}")
            if atom.hydrogens > 0:
                total_mass += atom.hydrogens * ATOMIC_MASSES.get("H", 1.007825)
    logger.debug("Calculated mass for monomer '%s': %s", monomer_name, total_mass)
    return total_mass if total_mass > 0 else None


def _map_and_classify_bonds(
    monomer_bonds_data: list | dict,
    monomers: dict[int, Monomer],
    atoms: dict[int, Atom],
    atomic_bonds: dict[AtomicEdge, AtomicBond],
) -> list[Bond]:
    """Map monomer bonds and dynamically classify their subtypes directly using loaded rules.

    Args:
        monomer_bonds_data: Raw monomer_bonds data from YAML/JSON payload.
        monomers: Dictionary mapping monomer IDs to Monomer instances.
        atoms: Dictionary mapping atom IDs to Atom instances.
        atomic_bonds: Dictionary mapping AtomicEdge instances to AtomicBond instances.

    Returns:
        List of Bond instances with classified subtypes.
    """
    bonds = []
    bond_items = []
    if isinstance(monomer_bonds_data, dict):
        bond_items = list(monomer_bonds_data.items())
    elif isinstance(monomer_bonds_data, list):
        for item in monomer_bonds_data:
            if isinstance(item, list) and len(item) == 2:
                u_v, edge_data = item
                bond_items.append((tuple(u_v), edge_data))

    logger.debug("Mapping and classifying %d monomer bonds", len(bond_items))
    for (u, v), edge_data in bond_items:
        u, v = int(u), int(v)
        edge_info_list = edge_data if isinstance(edge_data, list) else [edge_data]
        for bond_info in edge_info_list:
            if isinstance(bond_info, dict):
                monomer_to_atom_raw = bond_info.get("monomer_to_atom", {})
                mon_to_atom = {int(mk): int(av) for mk, av in monomer_to_atom_raw.items()}
                atomic_edge_info = bond_info.get("atomic_edge", bond_info)
                bond_type = atomic_edge_info.get("bond_type")
            else:
                continue

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


def _build_molecule_from_dict(  # noqa: C901
    data: dict, structure_id: str | None = None
) -> Molecule:
    """Convert a raw rBAN record dictionary into a kakapo Molecule instance.

    Maps the record's atoms, monomers, and atomic bonds to Molecule components,
    precalculates the exact monoisotopic mass of each monomer using the atomic composition, and
    dynamically classifies inter-monomer bond subtypes using subgraph isomorphism.

    Args:
        data: The raw dictionary parsed from YAML or JSON.
        structure_id: Optional string to override the record's default compound_id.

    Returns:
        A fully constructed Molecule instance with enriched properties and classified bonds.

    Raises:
        ValueError: If mandatory fields are missing or invalid.
    """
    final_id = str(structure_id or data.get("compound_id") or "")
    if not final_id:
        raise ValueError("Record is missing mandatory 'compound_id'")

    # 1. Map atoms to Atom
    atoms = {}
    raw_atoms = data.get("atoms")
    if raw_atoms is None:
        raise ValueError("Record is missing mandatory 'atoms' section")

    for atom_id_raw, atom_info in raw_atoms.items():
        atom_id = int(atom_id_raw)
        atoms[atom_id] = Atom(
            name=atom_info.get("name", "C"),
            hydrogens=int(atom_info.get("hydrogens", 0)),
        )

    # 2. Map monomers to Monomer using original keys
    raw_monomers = data.get("monomers")
    if raw_monomers is None:
        raise ValueError("Record is missing mandatory 'monomers' section")
    if len(raw_monomers) == 0:
        raise ValueError("Record must contain at least one monomer")

    monomers = {}
    for mon_idx_raw, mon_info in raw_monomers.items():
        mon_idx = int(mon_idx_raw)
        norine_name = str(mon_info.get("name", ""))
        nerpa_core = mon_info.get("nerpa_core")
        residue = str(nerpa_core) if nerpa_core is not None else norine_name
        is_methylated = bool(mon_info.get("methylated", False))

        chirality_str = str(mon_info.get("chirality", "UNKNOWN"))
        if chirality_str not in Chirality.__members__:
            raise KeyError(f"Invalid chirality: '{chirality_str}'")
        chirality = Chirality[chirality_str]

        atom_ids = tuple(int(a) for a in mon_info.get("atoms", []))

        # Calculate exact mass from atomic composition
        try:
            mass = _calculate_monomer_exact_mass(atom_ids, norine_name, atoms)
        except ValueError as e:
            logger.warning(
                "Failed to calculate exact mass for monomer %s: %s",
                norine_name,
                e,
            )
            mass = None

        monomers[mon_idx] = Monomer(
            norine_name=norine_name,
            residue=residue,
            is_methylated=is_methylated,
            chirality=chirality,
            atoms=atom_ids,
            mass=mass,
        )

    # 3. Map atomic bonds
    atomic_bonds = {}
    raw_atomic_bonds = data.get("atomic_bonds", [])
    for bond_entry in raw_atomic_bonds:
        if isinstance(bond_entry, (list, tuple)) and len(bond_entry) == 2:
            (u, v), edge_info = bond_entry
            arity = str(edge_info.get("arity", "1"))
            bond_type = edge_info.get("bond_type")
            atomic_bonds[AtomicEdge(int(u), int(v))] = AtomicBond(
                arity=arity,
                bond_type=bond_type,
            )

    # 4. Map monomer bonds and dynamically classify subtypes directly
    bonds = _map_and_classify_bonds(data.get("monomer_bonds", []), monomers, atoms, atomic_bonds)

    metadata = Metadata.from_dict(data.get("metadata"))
    smiles = generate_smiles_from_atoms(atoms, atomic_bonds)

    logger.debug(
        "Molecule %s built: %d atoms, %d monomers, %d atomic bonds, %d monomer bonds",
        final_id,
        len(atoms),
        len(monomers),
        len(atomic_bonds),
        len(bonds),
    )

    # 5. Return the final Molecule
    return Molecule(
        structure_id=final_id,
        monomers=monomers,
        bonds=tuple(bonds),
        smiles=smiles,
        atoms=atoms,
        atomic_bonds=atomic_bonds,
        metadata=metadata,
    )


def _parse_yaml_content_parallel(content: str) -> list[dict]:
    """Parse YAML list content in parallel by splitting it into single record blocks."""
    # Split content by list item pattern "\n- "
    raw_blocks = content.split("\n- ")
    if raw_blocks[0].startswith("- "):
        raw_blocks[0] = raw_blocks[0][2:]

    blocks = []
    for b in raw_blocks:
        if not b.strip():
            continue
        blocks.append("- " + b)

    num_workers = min(4, os.cpu_count() or 1)
    logger.info("Parsing YAML blocks in parallel using %d workers", num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_parse_single_yaml_block, blocks))

    parsed_dicts = []
    for res in results:
        if isinstance(res, list) and len(res) > 0:
            parsed_dicts.append(res[0])
        elif isinstance(res, dict):
            parsed_dicts.append(res)
    return parsed_dicts


def parse_rban_yaml(  # noqa: C901
    yaml_path: str | Path, id_counts: dict[str, int] | None = None
) -> Molecule | dict[str, Molecule]:
    """Parse an rBAN record YAML file and return a kakapo Molecule or a dict of Molecules.

    Args:
        yaml_path: Path to a YAML file containing one rBAN record dict or a list of them.
        id_counts: Optional tracker dictionary mapping compound IDs to their occurrence count
            for suffix-based deduplication.

    Returns:
        A kakapo Molecule instance representing the structure with
        cyclic topology information preserved, or a dictionary mapping structure IDs to Molecule
        instances if the file contains a list of records.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the YAML is not a valid rBAN record payload.
    """
    yaml_path = Path(yaml_path)
    logger.info("Parsing rBAN YAML from %s", yaml_path)
    if id_counts is None:
        id_counts = {}

    try:
        with open(yaml_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError as e:
        logger.error("YAML file not found: %s", yaml_path)
        raise FileNotFoundError(f"YAML file not found: {yaml_path}") from e

    if not content.strip():
        logger.warning("YAML file %s is empty", yaml_path)
        raise ValueError(f"YAML file {yaml_path} is empty")

    is_list = content.strip().startswith("-")

    try:
        if is_list:
            # Check how many blocks there are by counting "\n- "
            block_count = content.count("\n- ") + (1 if content.strip().startswith("- ") else 0)
            if block_count > 5:
                yaml_data = _parse_yaml_content_parallel(content)
            else:
                yaml_data = yaml.load(content, Loader=SafeLoader)
        else:
            yaml_data = yaml.load(content, Loader=SafeLoader)
    except yaml.YAMLError as e:
        logger.error("Invalid YAML in %s: %s", yaml_path, e)
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e
    except Exception as e:
        logger.error("Error parsing YAML file %s: %s", yaml_path, e)
        raise ValueError(f"Error parsing YAML file {yaml_path}: {e}") from e

    if yaml_data is None:
        logger.warning("YAML file %s is empty after load", yaml_path)
        raise ValueError(f"YAML file {yaml_path} is empty")

    if is_list and not yaml_data:
        logger.warning("YAML file %s contains an empty list", yaml_path)
        raise ValueError(f"YAML file {yaml_path} contains an empty list")

    records = yaml_data if is_list else [yaml_data]
    molecules = {}
    errors = []

    for i, item in enumerate(records):
        try:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dictionary item, got {type(item)}")
            base_id = item.get("compound_id")
            if not base_id:
                raise ValueError("Record is missing mandatory 'compound_id'")
            base_id = str(base_id)
            count = id_counts.get(base_id, 0)
            if count > 0:
                unique_id = f"{base_id}_{count + 1}"
            else:
                unique_id = base_id
            id_counts[base_id] = count + 1

            molecule = _build_molecule_from_dict(item, structure_id=unique_id)
            molecules[unique_id] = molecule
        except (KeyError, ValueError, TypeError) as e:
            if is_list:
                logger.warning("Index %d failed to parse in list from %s: %s", i, yaml_path, e)
                errors.append(f"Index {i}: {e}")
            else:
                logger.error("Failed to parse rBAN record from %s: %s", yaml_path, e)
                raise ValueError(f"Failed to parse rBAN record from {yaml_path}. Error: {e}") from e

    if errors:
        logger.error(
            "Failed to parse some rBAN records from %s: %s",
            yaml_path,
            errors,
        )
        raise ValueError(
            f"Failed to parse rBAN record list from {yaml_path}. Errors:\n" + "\n".join(errors)
        )

    if is_list:
        logger.info("Successfully parsed %d molecules from %s", len(molecules), yaml_path)
        return molecules

    molecule = list(molecules.values())[0]
    logger.info("Successfully parsed molecule %s from %s", molecule.structure_id, yaml_path)
    return molecule


def parse_rban_yaml_batch(  # noqa: C901
    yaml_paths: Iterable[str | Path],
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Molecule]:
    """Parse multiple rBAN YAML files.

    Args:
        yaml_paths: Iterable of paths to YAML files.
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
    logger.info("Parsing batch of YAML file(s)...")

    for path in yaml_paths:
        yaml_path = Path(path)

        # 1. Check Cache
        cached_mols = None
        cache_path = None
        if use_cache and cache_dir:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                stat_info = yaml_path.stat()
                mtime = stat_info.st_mtime
                size = stat_info.st_size
                hash_input = f"{yaml_path.resolve()}_{mtime}_{size}"
                file_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
                cache_path = cache_dir / f"{yaml_path.name}_{file_hash}.json"

                if cache_path.exists():
                    logger.info("Cache hit. Loading structures from cache %s", cache_path)
                    with open(cache_path, encoding="utf-8") as f:
                        cached_data = json.load(f)

                    cached_mols = {k: _deserialize_molecule(v) for k, v in cached_data.items()}
                    molecules.update(cached_mols)

                    # Update id_counts with base compound_ids
                    for mol_data in cached_data.values():
                        bid = mol_data.get("compound_id", mol_data["structure_id"])
                        id_counts[bid] = id_counts.get(bid, 0) + 1
                    continue
            except Exception as e:
                logger.warning("Failed to load structures from cache: %s", e)

        # 2. Parse YAML from scratch
        try:
            result = parse_rban_yaml(yaml_path, id_counts=id_counts)
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
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(serialized_data, f, indent=2)
                    temp_path.replace(cache_path)
                except Exception as e:
                    logger.warning("Failed to save structures to cache: %s", e)

        except Exception as e:
            logger.warning("Failed to parse YAML file %s in batch: %s", yaml_path, e)
            errors.append(f"{yaml_path}: {e}")

    if errors:
        logger.error("Failed to parse some YAML files: %s", errors)
        error_msg = "Failed to parse some YAML files:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    logger.info("Completed batch load. Loaded %d total molecules.", len(molecules))
    return molecules

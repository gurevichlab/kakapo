"""Unit tests for the JSON parser and its helper functions."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from kakapo.io.json_parser import (
    parse_nerpa_json,
    parse_nerpa_json_batch,
)
from kakapo.models import Molecule


class TestJsonParserBasic:
    """Test JSON parser with new BGC variant JSON structures."""

    def test_parse_nerpa_record_retains_atomic_payload(self):
        """Parse a full BGC variant JSON and retain source atom graph."""
        json_data = {
            "bgc_variant_id": {
                "bgc_id": {
                    "antiSMASH_file": "dummy.json",
                    "contig_idx": 1,
                    "bgc_idx": 1,
                },
                "variant_idx": 0,
            },
            "compound_id": "NRP_PARSED_1",
            "original_score": 10.0,
            "new_variants": {
                "variant_0": {
                    "score": 10.0,
                    "rank": 0,
                    "original_id": "NRP_PARSED_1",
                    "new_record": {
                        "compound_id": "NRP_PARSED_1",
                        "metadata": {
                            "name": "ParsedRecord",
                            "smiles": "NCC(=O)O",
                            "origin": None,
                            "inchikey": None,
                            "source": None,
                        },
                        "atoms": {
                            "0": {"name": "N", "hydrogens": 2},
                            "1": {"name": "C", "hydrogens": 1},
                            "2": {"name": "C", "hydrogens": 0},
                            "3": {"name": "O", "hydrogens": 0},
                            "4": {"name": "O", "hydrogens": 1},
                        },
                        "monomers": {
                            "0": {
                                "name": "Ala",
                                "atoms": [0, 1, 2, 3, 4],
                                "chirality": "L",
                                "is_pks_hybrid": False,
                            }
                        },
                        "monomer_bonds": [],
                        "atomic_bonds": [
                            [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                            [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            [[2, 3], {"arity": "2", "bond_type": "DOUBLE"}],
                            [[2, 4], {"arity": "1", "bond_type": "SINGLE"}],
                        ],
                    },
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, Molecule)
            molecule = res
            assert molecule.structure_id == "NRP_PARSED_1"
            assert molecule.smiles == "NCC(=O)O"
            assert len(molecule.monomers) == 1
            assert len(molecule.atomic_bonds) == 4

            # Verify monomer mass calculation check (alanine residue expected)
            ala = molecule.monomers[0]
            assert ala.mass is not None
            assert ala.mass > 70.0
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_record_list(self):
        """Parse a JSON file containing a list of BGC variant items."""
        json_data = [
            {
                "new_variants": {
                    "variant_0": {
                        "new_record": {
                            "compound_id": "NRP_PARSED_LIST_1",
                            "metadata": {
                                "name": "ParsedRecord1",
                                "smiles": "NCC(=O)O",
                                "origin": None,
                                "inchikey": None,
                                "source": None,
                            },
                            "atoms": {"0": {"name": "N", "hydrogens": 2}},
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0],
                                    "chirality": "L",
                                    "is_pks_hybrid": False,
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [],
                        }
                    }
                }
            },
            {
                "new_variants": {
                    "variant_0": {
                        "new_record": {
                            "compound_id": "NRP_PARSED_LIST_2",
                            "metadata": {
                                "name": "ParsedRecord2",
                                "smiles": "NCC(=O)O",
                                "origin": None,
                                "inchikey": None,
                                "source": None,
                            },
                            "atoms": {
                                "0": {"name": "N", "hydrogens": 2},
                                "1": {"name": "C", "hydrogens": 3},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1],
                                    "chirality": "L",
                                    "is_pks_hybrid": False,
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [],
                        }
                    }
                }
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, dict)
            molecules = res
            assert len(molecules) == 2
            assert "NRP_PARSED_LIST_1" in molecules
            assert "NRP_PARSED_LIST_2" in molecules
            assert isinstance(molecules["NRP_PARSED_LIST_1"], Molecule)
            assert isinstance(molecules["NRP_PARSED_LIST_2"], Molecule)
            assert molecules["NRP_PARSED_LIST_1"].structure_id == "NRP_PARSED_LIST_1"
            assert molecules["NRP_PARSED_LIST_2"].structure_id == "NRP_PARSED_LIST_2"
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_record_list_with_duplicates(self):
        """Parse list with duplicate compound IDs and apply suffixes."""
        json_data = [
            {
                "new_variants": {
                    "variant_0": {
                        "new_record": {
                            "compound_id": "NRP_DUP",
                            "metadata": {"name": "A", "smiles": "N"},
                            "atoms": {"0": {"name": "N", "hydrogens": 2}},
                            "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                            "monomer_bonds": [],
                            "atomic_bonds": [],
                        }
                    }
                }
            },
            {
                "new_variants": {
                    "variant_0": {
                        "new_record": {
                            "compound_id": "NRP_DUP",
                            "metadata": {"name": "B", "smiles": "N"},
                            "atoms": {
                                "0": {"name": "N", "hydrogens": 2},
                                "1": {"name": "C", "hydrogens": 3},
                            },
                            "monomers": {"0": {"name": "Ala", "atoms": [0, 1], "chirality": "L"}},
                            "monomer_bonds": [],
                            "atomic_bonds": [],
                        }
                    }
                }
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, dict)
            molecules = res
            assert len(molecules) == 2
            assert "NRP_DUP" in molecules
            assert "NRP_DUP_2" in molecules
        finally:
            os.unlink(json_path)


class TestJsonParserErrors:
    """Test error handling in JSON parser."""

    def test_parse_nonexistent_file(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            parse_nerpa_json("/nonexistent/path/file.json")

    def test_parse_invalid_json(self):
        """Should raise ValueError for invalid JSON syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json content}")
            json_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_nerpa_json(json_path)
        finally:
            os.unlink(json_path)

    def test_parse_empty_json(self):
        """Should raise ValueError for empty JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            json_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_nerpa_json(json_path)
        finally:
            os.unlink(json_path)

    def test_parse_missing_required_fields(self):
        """Should raise ValueError when Nerpa record fields are missing."""
        json_data = {
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP005",
                        "metadata": {"smiles": "NCC(=O)O"},
                    }
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_nerpa_json(json_path)
        finally:
            os.unlink(json_path)

    def test_parse_missing_monomers(self):
        """Should raise ValueError when Nerpa record has no monomers."""
        json_data = {
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP006",
                        "metadata": {
                            "name": "NoMonomers",
                            "smiles": "NCC(=O)O",
                            "origin": None,
                            "inchikey": None,
                            "source": None,
                        },
                        "atoms": {"0": {"name": "N", "hydrogens": 2}},
                        "atomic_bonds": [],
                        "monomer_bonds": [],
                        "monomers": {},
                    }
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            with pytest.raises(ValueError, match="at least one monomer"):
                parse_nerpa_json(json_path)
        finally:
            os.unlink(json_path)


class TestJsonParserBatch:
    """Test batch parsing functionality."""

    def test_parse_batch_multiple_files(self):
        """Parse multiple JSON files at once."""
        json_data_1 = {
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP_A",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            }
        }
        json_data_2 = {
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP_B",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {
                            "0": {"name": "N", "hydrogens": 1},
                            "1": {"name": "C", "hydrogens": 3},
                        },
                        "monomers": {"0": {"name": "Gly", "atoms": [0, 1], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path_1 = os.path.join(tmpdir, "nrp_a.json")
            json_path_2 = os.path.join(tmpdir, "nrp_b.json")

            with open(json_path_1, "w") as f:
                json.dump(json_data_1, f)
            with open(json_path_2, "w") as f:
                json.dump(json_data_2, f)

            molecules = parse_nerpa_json_batch([json_path_1, json_path_2])

            assert len(molecules) == 2
            assert "NRP_A" in molecules
            assert "NRP_B" in molecules
            assert molecules["NRP_A"].structure_id == "NRP_A"
            assert molecules["NRP_B"].structure_id == "NRP_B"

    def test_parse_batch_with_error(self):
        """Batch parsing should report all errors."""
        json_data_1 = {
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP_A",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path_1 = os.path.join(tmpdir, "nrp_a.json")
            json_path_2 = os.path.join(tmpdir, "nrp_b.json")

            with open(json_path_1, "w") as f:
                json.dump(json_data_1, f)
            with open(json_path_2, "w") as f:
                f.write("invalid json")

            with pytest.raises(ValueError, match="Failed to parse some JSON files"):
                parse_nerpa_json_batch([json_path_1, json_path_2])

    def test_parse_batch_with_caching(self):
        """Parse multiple files with and without cache hits."""
        json_data_1 = {
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP_C",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path_1 = os.path.join(tmpdir, "nrp_c.json")
            cache_dir = Path(tmpdir) / "cache"

            with open(json_path_1, "w") as f:
                json.dump(json_data_1, f)

            # First run: parses and saves to cache
            molecules = parse_nerpa_json_batch([json_path_1], cache_dir=cache_dir, use_cache=True)
            assert "NRP_C" in molecules

            # Second run: cache hit
            molecules2 = parse_nerpa_json_batch([json_path_1], cache_dir=cache_dir, use_cache=True)
            assert "NRP_C" in molecules2
            assert molecules2["NRP_C"].structure_id == "NRP_C"


class TestCalculateMonomerExactMass:
    """Unit tests for _calculate_monomer_exact_mass helper."""

    def test_calculate_mass_valid(self):
        """Should correctly sum atomic masses with implicit hydrogens."""
        from kakapo.io.json_parser import _calculate_monomer_exact_mass
        from kakapo.models import Atom

        atom_ids = (0, 1)
        atoms = {
            0: Atom(name="C", hydrogens=3),
            1: Atom(name="O", hydrogens=1),
        }
        # C (12.0) + O (15.994915) + 4 * H (1.007825) = 32.026215
        expected_mass = 12.0 + 15.994915 + 4 * 1.007825
        mass = _calculate_monomer_exact_mass(atom_ids, "mock", atoms)
        assert mass == pytest.approx(expected_mass)

    def test_calculate_mass_unknown_element(self):
        """Should raise ValueError for unknown elements."""
        from kakapo.io.json_parser import _calculate_monomer_exact_mass
        from kakapo.models import Atom

        atom_ids = (0,)
        atoms = {
            0: Atom(name="Xx", hydrogens=0),
        }
        with pytest.raises(ValueError, match="Unknown element 'Xx'"):
            _calculate_monomer_exact_mass(atom_ids, "mock", atoms)

    def test_calculate_mass_missing_atom_id(self):
        """Should ignore missing atom IDs in the dictionary."""
        from kakapo.io.json_parser import _calculate_monomer_exact_mass
        from kakapo.models import Atom

        atom_ids = (0, 999)
        atoms = {
            0: Atom(name="C", hydrogens=0),
        }
        # 999 is missing, should only calculate for atom 0
        mass = _calculate_monomer_exact_mass(atom_ids, "mock", atoms)
        assert mass == pytest.approx(12.0)

    def test_calculate_mass_empty_atoms(self):
        """Should return None if total mass is zero or no atoms are present."""
        from kakapo.io.json_parser import _calculate_monomer_exact_mass

        assert _calculate_monomer_exact_mass((), "mock", {}) is None


class TestGenerateSmilesFromAtoms:
    """Unit tests for generate_smiles_from_atoms helper."""

    def test_generate_smiles_empty(self):
        """Should return None if atoms dictionary is empty."""
        from kakapo.io.json_parser import generate_smiles_from_atoms

        assert generate_smiles_from_atoms({}, {}) is None

    def test_generate_smiles_simple_molecule(self):
        """Should generate canonical SMILES for simple glycine molecule."""
        from kakapo.io.json_parser import generate_smiles_from_atoms
        from kakapo.models import Atom, AtomicBond, AtomicEdge

        atoms = {
            0: Atom(name="N", hydrogens=2),
            1: Atom(name="C", hydrogens=2),
            2: Atom(name="C", hydrogens=0),
            3: Atom(name="O", hydrogens=0),
            4: Atom(name="O", hydrogens=1),
        }
        atomic_bonds = {
            AtomicEdge(0, 1): AtomicBond(arity="1", bond_type="SINGLE"),
            AtomicEdge(1, 2): AtomicBond(arity="1", bond_type="SINGLE"),
            AtomicEdge(2, 3): AtomicBond(arity="2", bond_type="DOUBLE"),
            AtomicEdge(2, 4): AtomicBond(arity="1", bond_type="SINGLE"),
        }
        smiles = generate_smiles_from_atoms(atoms, atomic_bonds)
        assert smiles == "NCC(=O)O"


class TestCacheSerializationRoundtrip:
    """Unit tests for cache serialization and deserialization helpers."""

    def test_cache_serialization_roundtrip(self):
        """Test roundtrip caching of a Molecule."""
        from kakapo.io.json_parser import (
            _deserialize_molecule_from_cache,
            _serialize_molecule_for_cache,
        )
        from kakapo.models import Chirality, Molecule, Monomer

        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        dummy_mol = Molecule(
            structure_id="Ala_structure_cache",
            monomers={0: monomer},
            bonds=(),
        )

        serialized = _serialize_molecule_for_cache(dummy_mol)
        assert serialized["structure_id"] == "Ala_structure_cache"
        assert serialized["compound_id"] == "Ala_structure_cache"

        restored = _deserialize_molecule_from_cache(serialized)
        assert restored.structure_id == "Ala_structure_cache"
        assert restored.monomers[0].norine_name == "Ala"

    def test_cache_serialization_roundtrip_with_nerpa_ms_data(self):
        """Test roundtrip caching of a Molecule with nerpa_ms_data."""
        from kakapo.io.json_parser import (
            _deserialize_molecule_from_cache,
            _serialize_molecule_for_cache,
        )
        from kakapo.models import Chirality, Molecule, Monomer, MonomerOrigin, NerpaMSData

        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        nerpa_data = NerpaMSData(
            rank=1,
            score=15.2,
            old_to_new_mon_map=((None, 7),),
            monomer_origins=(
                MonomerOrigin(compound_id="comp_100", monomer_name="Ala", monomer_idx=4),
            ),
        )
        dummy_mol = Molecule(
            structure_id="Nerpa_structure_cache",
            monomers={0: monomer},
            bonds=(),
            nerpa_ms_data=nerpa_data,
        )

        serialized = _serialize_molecule_for_cache(dummy_mol)
        assert serialized["nerpa_ms_data"]["rank"] == 1
        assert serialized["nerpa_ms_data"]["score"] == 15.2
        assert serialized["nerpa_ms_data"]["old_to_new_mon_map"] == [[None, 7]]
        assert serialized["nerpa_ms_data"]["monomer_origins"][0]["compound_id"] == "comp_100"

        restored = _deserialize_molecule_from_cache(serialized)
        assert restored.structure_id == "Nerpa_structure_cache"
        assert restored.nerpa_ms_data is not None
        assert restored.nerpa_ms_data.rank == 1
        assert restored.nerpa_ms_data.score == 15.2
        assert restored.nerpa_ms_data.old_to_new_mon_map == ((None, 7),)
        assert len(restored.nerpa_ms_data.monomer_origins) == 1
        assert restored.nerpa_ms_data.monomer_origins[0].monomer_name == "Ala"


class TestNerpaMSDataJsonParsing:
    """Granular tests for parsing NERPA-MS variant fields from JSON payloads."""

    def test_parse_nerpa_json_populates_rank_and_score(self):
        """Parse BGC variant JSON and verify rank and score extraction."""
        json_data = {
            "compound_id": "NRP_NERPA_RS",
            "new_variants": {
                "variant_0": {
                    "score": 14.5,
                    "rank": 1,
                    "new_record": {
                        "compound_id": "NRP_NERPA_RS",
                        "atoms": {},
                        "monomers": {"0": {"name": "Ala", "atoms": [], "chirality": "L"}},
                        "monomer_bonds": [],
                    },
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, Molecule)
            assert res.nerpa_ms_data is not None
            assert res.nerpa_ms_data.rank == 1
            assert res.nerpa_ms_data.score == pytest.approx(14.5)
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_json_populates_old_to_new_mon_map(self):
        """Parse BGC variant JSON and verify old_to_new_mon_map extraction."""
        json_data = {
            "compound_id": "NRP_NERPA_MAP",
            "new_variants": {
                "variant_0": {
                    "old_to_new_mon_map": [[None, 7], [1, 2]],
                    "new_record": {
                        "compound_id": "NRP_NERPA_MAP",
                        "atoms": {},
                        "monomers": {"0": {"name": "Ala", "atoms": [], "chirality": "L"}},
                        "monomer_bonds": [],
                    },
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, Molecule)
            assert res.nerpa_ms_data is not None
            assert res.nerpa_ms_data.old_to_new_mon_map == ((None, 7), (1, 2))
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_json_populates_monomer_origins(self):
        """Parse BGC variant JSON and verify monomer_origins extraction."""
        json_data = {
            "compound_id": "NRP_NERPA_ORIG",
            "new_variants": {
                "variant_0": {
                    "monomer_origins": [
                        {
                            "compound_id": "compound_001780",
                            "monomer_name": "Ala",
                            "monomer_idx": 4,
                        }
                    ],
                    "new_record": {
                        "compound_id": "NRP_NERPA_ORIG",
                        "atoms": {},
                        "monomers": {"0": {"name": "Ala", "atoms": [], "chirality": "L"}},
                        "monomer_bonds": [],
                    },
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, Molecule)
            assert res.nerpa_ms_data is not None
            assert len(res.nerpa_ms_data.monomer_origins) == 1
            origin = res.nerpa_ms_data.monomer_origins[0]
            assert origin.compound_id == "compound_001780"
            assert origin.monomer_name == "Ala"
            assert origin.monomer_idx == 4
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_json_without_nerpa_ms_fields(self):
        """Parse BGC variant JSON lacking nerpa_ms_data fields results in None."""
        json_data = {
            "compound_id": "NRP_NO_NERPA",
            "new_variants": {
                "variant_0": {
                    "new_record": {
                        "compound_id": "NRP_NO_NERPA",
                        "atoms": {},
                        "monomers": {"0": {"name": "Ala", "atoms": [], "chirality": "L"}},
                        "monomer_bonds": [],
                    },
                }
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, Molecule)
            assert res.nerpa_ms_data is None
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_json_deduplicates_by_smiles(self):
        """Verify deduplication by smiles and lowest nerpa rank selection."""
        # We will create a BGCVariantItem list with 3 variants sharing the same smiles
        json_data = [
            {
                "new_variants": {
                    "variant_0": {
                        "rank": 2,
                        "new_record": {
                            "compound_id": "MOL_B",
                            "atoms": {
                                "0": {"name": "C", "hydrogens": 3},
                                "1": {"name": "C", "hydrogens": 2},
                                "2": {"name": "O", "hydrogens": 1},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1, 2],
                                    "chirality": "L",
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [
                                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            ],
                        },
                    },
                    "variant_1": {
                        "rank": 1,
                        "new_record": {
                            "compound_id": "MOL_C",
                            "atoms": {
                                "0": {"name": "C", "hydrogens": 3},
                                "1": {"name": "C", "hydrogens": 2},
                                "2": {"name": "O", "hydrogens": 1},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1, 2],
                                    "chirality": "L",
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [
                                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            ],
                        },
                    },
                    "variant_2": {
                        # No rank
                        "new_record": {
                            "compound_id": "MOL_A",
                            "atoms": {
                                "0": {"name": "C", "hydrogens": 3},
                                "1": {"name": "C", "hydrogens": 2},
                                "2": {"name": "O", "hydrogens": 1},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1, 2],
                                    "chirality": "L",
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [
                                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            ],
                        },
                    },
                }
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, dict)
            # Only 1 molecule should remain since they all have the same SMILES
            assert len(res) == 1
            mol = list(res.values())[0]
            # MOL_C has the lowest rank (1) so it should be kept
            assert mol.structure_id == "MOL_C"
            assert set(mol.duplicated_structure_ids) == {"MOL_A", "MOL_B"}
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_json_deduplicates_by_smiles_tie_breaker(self):
        """Verify tie-breaker on structure_id when ranks are identical/missing."""
        # 3 variants: MOL_C and MOL_A have rank 2, MOL_B has no rank
        json_data = [
            {
                "new_variants": {
                    "variant_0": {
                        "rank": 2,
                        "new_record": {
                            "compound_id": "MOL_C",
                            "atoms": {
                                "0": {"name": "C", "hydrogens": 3},
                                "1": {"name": "C", "hydrogens": 2},
                                "2": {"name": "O", "hydrogens": 1},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1, 2],
                                    "chirality": "L",
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [
                                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            ],
                        },
                    },
                    "variant_1": {
                        "rank": 2,
                        "new_record": {
                            "compound_id": "MOL_A",
                            "atoms": {
                                "0": {"name": "C", "hydrogens": 3},
                                "1": {"name": "C", "hydrogens": 2},
                                "2": {"name": "O", "hydrogens": 1},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1, 2],
                                    "chirality": "L",
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [
                                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            ],
                        },
                    },
                    "variant_2": {
                        "new_record": {
                            "compound_id": "MOL_B",
                            "atoms": {
                                "0": {"name": "C", "hydrogens": 3},
                                "1": {"name": "C", "hydrogens": 2},
                                "2": {"name": "O", "hydrogens": 1},
                            },
                            "monomers": {
                                "0": {
                                    "name": "Ala",
                                    "atoms": [0, 1, 2],
                                    "chirality": "L",
                                }
                            },
                            "monomer_bonds": [],
                            "atomic_bonds": [
                                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
                            ],
                        },
                    },
                }
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_nerpa_json(json_path)
            assert isinstance(res, dict)
            assert len(res) == 1
            mol = list(res.values())[0]
            # MOL_A and MOL_C have lowest rank (2). MOL_A should be chosen by structure_id.
            assert mol.structure_id == "MOL_A"
            assert set(mol.duplicated_structure_ids) == {"MOL_B", "MOL_C"}
        finally:
            os.unlink(json_path)

    def test_parse_nerpa_json_batch_deduplicates_across_files(self):
        """Verify that parse_nerpa_json_batch deduplicates molecules across multiple files."""
        # File 1 has MOL_A (rank 2)
        # File 2 has MOL_B (rank 1)
        # Both share same structure/atoms/SMILES
        atom_payload = {
            "atoms": {
                "0": {"name": "C", "hydrogens": 3},
                "1": {"name": "C", "hydrogens": 2},
                "2": {"name": "O", "hydrogens": 1},
            },
            "monomers": {
                "0": {
                    "name": "Ala",
                    "atoms": [0, 1, 2],
                    "chirality": "L",
                }
            },
            "monomer_bonds": [],
            "atomic_bonds": [
                [[0, 1], {"arity": "1", "bond_type": "SINGLE"}],
                [[1, 2], {"arity": "1", "bond_type": "SINGLE"}],
            ],
        }
        json_data_1 = [
            {
                "new_variants": {
                    "variant_0": {
                        "rank": 2,
                        "new_record": {
                            "compound_id": "MOL_A",
                            **atom_payload,
                        },
                    }
                }
            }
        ]
        json_data_2 = [
            {
                "new_variants": {
                    "variant_0": {
                        "rank": 1,
                        "new_record": {
                            "compound_id": "MOL_B",
                            **atom_payload,
                        },
                    }
                }
            }
        ]

        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1,
            tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2,
        ):
            json.dump(json_data_1, f1)
            json.dump(json_data_2, f2)
            path1, path2 = f1.name, f2.name

        try:
            res = parse_nerpa_json_batch([path1, path2], use_cache=False)
            assert len(res) == 1
            mol = list(res.values())[0]
            # MOL_B has lower rank (1) so it should be kept
            assert mol.structure_id == "MOL_B"
            assert mol.duplicated_structure_ids == ("MOL_A",)
        finally:
            os.unlink(path1)
            os.unlink(path2)

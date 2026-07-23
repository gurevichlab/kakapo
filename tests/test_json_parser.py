"""Unit tests for the JSON parser and its helper functions."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from kakapo.io.json_parser import (
    parse_rban_json,
    parse_rban_json_batch,
)
from kakapo.models import Molecule


class TestJsonParserBasic:
    """Test JSON parser with new BGC variant JSON structures."""

    def test_parse_parsed_rban_record_retains_atomic_payload(self):
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
            "new_variants": [
                {
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
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_rban_json(json_path)
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

    def test_parse_parsed_rban_record_list(self):
        """Parse a JSON file containing a list of BGC variant items."""
        json_data = [
            {
                "new_variants": [
                    {
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
                ]
            },
            {
                "new_variants": [
                    {
                        "new_record": {
                            "compound_id": "NRP_PARSED_LIST_2",
                            "metadata": {
                                "name": "ParsedRecord2",
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
                ]
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_rban_json(json_path)
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

    def test_parse_parsed_rban_record_list_with_duplicates(self):
        """Parse list with duplicate compound IDs and apply suffixes."""
        json_data = [
            {
                "new_variants": [
                    {
                        "new_record": {
                            "compound_id": "NRP_DUP",
                            "metadata": {"name": "A", "smiles": "N"},
                            "atoms": {"0": {"name": "N", "hydrogens": 2}},
                            "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                            "monomer_bonds": [],
                            "atomic_bonds": [],
                        }
                    }
                ]
            },
            {
                "new_variants": [
                    {
                        "new_record": {
                            "compound_id": "NRP_DUP",
                            "metadata": {"name": "B", "smiles": "N"},
                            "atoms": {"0": {"name": "N", "hydrogens": 2}},
                            "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                            "monomer_bonds": [],
                            "atomic_bonds": [],
                        }
                    }
                ]
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            res = parse_rban_json(json_path)
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
            parse_rban_json("/nonexistent/path/file.json")

    def test_parse_invalid_json(self):
        """Should raise ValueError for invalid JSON syntax."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json content}")
            json_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_rban_json(json_path)
        finally:
            os.unlink(json_path)

    def test_parse_empty_json(self):
        """Should raise ValueError for empty JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            json_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_rban_json(json_path)
        finally:
            os.unlink(json_path)

    def test_parse_missing_required_fields(self):
        """Should raise ValueError when Parsed_rBAN_Record fields are missing."""
        json_data = {
            "new_variants": [
                {
                    "new_record": {
                        "compound_id": "NRP005",
                        "metadata": {"smiles": "NCC(=O)O"},
                    }
                }
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_rban_json(json_path)
        finally:
            os.unlink(json_path)

    def test_parse_missing_monomers(self):
        """Should raise ValueError when Parsed_rBAN_Record has no monomers."""
        json_data = {
            "new_variants": [
                {
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
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            json_path = f.name

        try:
            with pytest.raises(ValueError, match="at least one monomer"):
                parse_rban_json(json_path)
        finally:
            os.unlink(json_path)


class TestJsonParserBatch:
    """Test batch parsing functionality."""

    def test_parse_batch_multiple_files(self):
        """Parse multiple JSON files at once."""
        json_data_1 = {
            "new_variants": [
                {
                    "new_record": {
                        "compound_id": "NRP_A",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            ]
        }
        json_data_2 = {
            "new_variants": [
                {
                    "new_record": {
                        "compound_id": "NRP_B",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Gly", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path_1 = os.path.join(tmpdir, "nrp_a.json")
            json_path_2 = os.path.join(tmpdir, "nrp_b.json")

            with open(json_path_1, "w") as f:
                json.dump(json_data_1, f)
            with open(json_path_2, "w") as f:
                json.dump(json_data_2, f)

            molecules = parse_rban_json_batch([json_path_1, json_path_2])

            assert len(molecules) == 2
            assert "NRP_A" in molecules
            assert "NRP_B" in molecules
            assert molecules["NRP_A"].structure_id == "NRP_A"
            assert molecules["NRP_B"].structure_id == "NRP_B"

    def test_parse_batch_with_error(self):
        """Batch parsing should report all errors."""
        json_data_1 = {
            "new_variants": [
                {
                    "new_record": {
                        "compound_id": "NRP_A",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path_1 = os.path.join(tmpdir, "nrp_a.json")
            json_path_2 = os.path.join(tmpdir, "nrp_b.json")

            with open(json_path_1, "w") as f:
                json.dump(json_data_1, f)
            with open(json_path_2, "w") as f:
                f.write("invalid json")

            with pytest.raises(ValueError, match="Failed to parse some JSON files"):
                parse_rban_json_batch([json_path_1, json_path_2])

    def test_parse_batch_with_caching(self):
        """Parse multiple files with and without cache hits."""
        json_data_1 = {
            "new_variants": [
                {
                    "new_record": {
                        "compound_id": "NRP_C",
                        "metadata": {"name": None, "smiles": "N"},
                        "atoms": {"0": {"name": "N", "hydrogens": 1}},
                        "monomers": {"0": {"name": "Ala", "atoms": [0], "chirality": "L"}},
                        "monomer_bonds": [],
                        "atomic_bonds": [],
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path_1 = os.path.join(tmpdir, "nrp_c.json")
            cache_dir = Path(tmpdir) / "cache"

            with open(json_path_1, "w") as f:
                json.dump(json_data_1, f)

            # First run: parses and saves to cache
            molecules = parse_rban_json_batch([json_path_1], cache_dir=cache_dir, use_cache=True)
            assert "NRP_C" in molecules

            # Second run: cache hit
            molecules2 = parse_rban_json_batch([json_path_1], cache_dir=cache_dir, use_cache=True)
            assert "NRP_C" in molecules2
            assert molecules2["NRP_C"].structure_id == "NRP_C"

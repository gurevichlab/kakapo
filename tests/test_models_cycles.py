"""Test cyclic topology support in kakapo models."""

import pytest

from kakapo.models import Chirality, Cycle, Molecule, Monomer


class TestCycle:
    """Test the Cycle data model."""

    def test_cycle_creation(self):
        """Test creating a valid cycle."""
        cycle = Cycle(monomer_indices=(0, 1, 2))
        assert cycle.monomer_indices == (0, 1, 2)

    def test_cycle_minimum_size(self):
        """Cycle must have at least 2 monomers."""
        with pytest.raises(ValueError, match="at least 2 monomers"):
            Cycle(monomer_indices=(0,))

    def test_cycle_unique_indices(self):
        """Cycle indices must be unique."""
        with pytest.raises(ValueError, match="unique"):
            Cycle(monomer_indices=(0, 1, 0))


class TestMoleculeWithCycles:
    """Test Molecule model with cyclic topology."""

    @pytest.fixture
    def linear_monomer(self):
        """Create a standard test monomer."""
        return Monomer(
            norine_name="Ala",
            residue="A",
            is_methylated=False,
            chirality=Chirality.L,
        )

    def test_linear_molecule_no_cycles(self, linear_monomer):
        """Linear molecule has no cycles."""
        mol = Molecule(
            structure_id="MOL001",
            monomers=(linear_monomer, linear_monomer, linear_monomer),
        )
        assert not mol.is_cyclic
        assert len(mol.cycles) == 0

    def test_cyclic_molecule_single_cycle(self, linear_monomer):
        """Molecule with a single cyclic region."""
        cycle = Cycle(monomer_indices=(0, 1, 2))
        mol = Molecule(
            structure_id="MOL002",
            monomers=(linear_monomer, linear_monomer, linear_monomer),
            cycles=(cycle,),
        )
        assert mol.is_cyclic
        assert len(mol.cycles) == 1

    def test_cyclic_molecule_multiple_cycles(self, linear_monomer):
        """Molecule with multiple cyclic regions."""
        cycle1 = Cycle(monomer_indices=(0, 1))
        cycle2 = Cycle(monomer_indices=(3, 4, 5))
        monomers = tuple(linear_monomer for _ in range(6))
        mol = Molecule(
            structure_id="MOL003",
            monomers=monomers,
            cycles=(cycle1, cycle2),
        )
        assert mol.is_cyclic
        assert len(mol.cycles) == 2

    def test_cycle_index_validation(self, linear_monomer):
        """Invalid cycle indices raise ValueError."""
        cycle = Cycle(monomer_indices=(0, 1, 5))  # Index 5 out of bounds
        with pytest.raises(ValueError, match="invalid monomer index"):
            Molecule(
                structure_id="MOL004",
                monomers=(linear_monomer, linear_monomer),
                cycles=(cycle,),
            )

    def test_get_monomer_cycles(self, linear_monomer):
        """Query which cycles a monomer belongs to."""
        cycle1 = Cycle(monomer_indices=(0, 1, 2))
        cycle2 = Cycle(monomer_indices=(1, 3))
        monomers = tuple(linear_monomer for _ in range(4))
        mol = Molecule(
            structure_id="MOL005",
            monomers=monomers,
            cycles=(cycle1, cycle2),
        )

        # Monomer 0: in cycle1 only
        cycles_0 = mol.get_monomer_cycles(0)
        assert len(cycles_0) == 1
        assert cycles_0[0] == cycle1

        # Monomer 1: in both cycles
        cycles_1 = mol.get_monomer_cycles(1)
        assert len(cycles_1) == 2
        assert set(cycles_1) == {cycle1, cycle2}

        # Monomer 2: in cycle1 only
        cycles_2 = mol.get_monomer_cycles(2)
        assert len(cycles_2) == 1

        # Monomer 3: in cycle2 only
        cycles_3 = mol.get_monomer_cycles(3)
        assert len(cycles_3) == 1

    def test_is_monomer_cyclic(self, linear_monomer):
        """Query whether a monomer participates in any cycle."""
        cycle = Cycle(monomer_indices=(0, 1))
        monomers = tuple(linear_monomer for _ in range(3))
        mol = Molecule(
            structure_id="MOL006",
            monomers=monomers,
            cycles=(cycle,),
        )

        assert mol.is_monomer_cyclic(0)  # in cycle
        assert mol.is_monomer_cyclic(1)  # in cycle
        assert not mol.is_monomer_cyclic(2)  # not in cycle

    def test_get_linear_segments(self, linear_monomer):
        """Extract linear (non-cyclic) regions from molecule."""
        # Structure: Linear(0) - Cyclic(1,2) - Linear(3,4) - Cyclic(5,6) - Linear(7)
        cycle1 = Cycle(monomer_indices=(1, 2))
        cycle2 = Cycle(monomer_indices=(5, 6))
        monomers = tuple(linear_monomer for _ in range(8))

        mol = Molecule(
            structure_id="MOL007",
            monomers=monomers,
            cycles=(cycle1, cycle2),
        )

        segments = mol.get_linear_segments()
        assert segments == [(0, 0), (3, 4), (7, 7)]

    def test_get_linear_segments_all_cyclic(self, linear_monomer):
        """Molecule with all monomers in cycles has no linear segments."""
        cycle = Cycle(monomer_indices=(0, 1, 2))
        monomers = tuple(linear_monomer for _ in range(3))
        mol = Molecule(
            structure_id="MOL008",
            monomers=monomers,
            cycles=(cycle,),
        )

        assert mol.get_linear_segments() == []

    def test_get_linear_segments_all_linear(self, linear_monomer):
        """Linear molecule returns entire sequence as single segment."""
        monomers = tuple(linear_monomer for _ in range(5))
        mol = Molecule(
            structure_id="MOL009",
            monomers=monomers,
            cycles=(),
        )

        assert mol.get_linear_segments() == [(0, 4)]

    def test_monomer_index_out_of_bounds(self, linear_monomer):
        """Querying invalid monomer indices raises IndexError."""
        mol = Molecule(
            structure_id="MOL010",
            monomers=(linear_monomer,),
        )

        with pytest.raises(IndexError, match="out of bounds"):
            mol.get_monomer_cycles(5)

        with pytest.raises(IndexError, match="out of bounds"):
            mol.is_monomer_cyclic(-1)

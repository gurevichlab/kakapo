"""Unit tests for the spectrum fragmenter.

Comprehensive test coverage for:
- Fragment generation (bridges and 2-cuts)
- Mass calculations without water subtraction
- Ion type generation ([M+H]+, [M-H]-, neutral)
- Edge cases (single monomer, cycles, complex structures)
"""

import pytest

from kakapo.core.fragmenter import (
    Fragment,
    FragmentationGraph,
    MonomerMassCalculator,
    calculate_theoretical_masses,
    generate_all_theoretical_masses,
    generate_fragments,
    generate_theoretical_spectrum,
)
from kakapo.models import (
    Bond,
    Chirality,
    Cycle,
    Molecule,
    Monomer,
    TheoreticalIon,
    TheoreticalSpectrum,
)

# ============================================================================
# Fixtures: Reusable Test Data
# ============================================================================


@pytest.fixture
def simple_monomer():
    """Single monomer (ALA - Alanine)."""
    return Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
    )


@pytest.fixture
def linear_dimer():
    """Two monomers in a linear chain (A-B)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
    )
    bond = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE", is_ring_bond=False)

    return Molecule(
        structure_id="linear_dimer",
        monomers=(m1, m2),
        bonds=(bond,),
        cycles=(),
    )


@pytest.fixture
def linear_trimer():
    """Three monomers in a linear chain (A-B-C)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
    )

    bond1 = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE", is_ring_bond=False)
    bond2 = Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE", is_ring_bond=False)

    return Molecule(
        structure_id="linear_trimer",
        monomers=(m1, m2, m3),
        bonds=(bond1, bond2),
        cycles=(),
    )


@pytest.fixture
def cyclic_trimer():
    """Three monomers in a cycle (A-B-C-A)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
    )

    bond1 = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE", is_ring_bond=False)
    bond2 = Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE", is_ring_bond=False)
    bond3 = Bond(source_idx=2, target_idx=0, bond_type="PEPTIDE", is_ring_bond=True)

    cycle = Cycle(monomer_indices=(0, 1, 2))

    return Molecule(
        structure_id="cyclic_trimer",
        monomers=(m1, m2, m3),
        bonds=(bond1, bond2, bond3),
        cycles=(cycle,),
    )


@pytest.fixture
def mixed_structure():
    """Mixed: 2 linear monomers + 3 cyclic monomers (L1-L2-C1-C2-C3-C1)."""
    monomers = tuple(
        Monomer(
            norine_name=name,
            residue=name,
            is_methylated=False,
            chirality=Chirality.L,
        )
        for name in ["ALA", "GLY", "PRO", "SER", "THR"]
    )

    bonds = [
        Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE", is_ring_bond=False),  # L1-L2
        Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE", is_ring_bond=False),  # L2-C1
        Bond(source_idx=2, target_idx=3, bond_type="PEPTIDE", is_ring_bond=False),  # C1-C2
        Bond(source_idx=3, target_idx=4, bond_type="PEPTIDE", is_ring_bond=False),  # C2-C3
        Bond(source_idx=4, target_idx=2, bond_type="PEPTIDE", is_ring_bond=True),  # C3-C1 (ring)
    ]

    cycle = Cycle(monomer_indices=(2, 3, 4))

    return Molecule(
        structure_id="mixed_structure",
        monomers=monomers,
        bonds=tuple(bonds),
        cycles=(cycle,),
    )


@pytest.fixture
def linear_trimer_reversed_edges():
    """Three monomers in a linear chain with reversed bond orientation."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
    )

    # Reverse orientation versus index order: 1->0 and 2->1.
    bond1 = Bond(source_idx=1, target_idx=0, bond_type="PEPTIDE", is_ring_bond=False)
    bond2 = Bond(source_idx=2, target_idx=1, bond_type="PEPTIDE", is_ring_bond=False)

    return Molecule(
        structure_id="linear_trimer_reversed",
        monomers=(m1, m2, m3),
        bonds=(bond1, bond2),
        cycles=(),
    )


@pytest.fixture
def cyclic_trimer_mixed_directions():
    """3-cycle with mixed edge directions (no directed cycle, still undirected cycle)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
    )

    # Undirected topology is a triangle {0-1, 1-2, 0-2}, but directed simple cycle
    # 0->1->2->0 is intentionally absent.
    bond1 = Bond(source_idx=1, target_idx=0, bond_type="PEPTIDE", is_ring_bond=False)
    bond2 = Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE", is_ring_bond=False)
    bond3 = Bond(source_idx=0, target_idx=2, bond_type="PEPTIDE", is_ring_bond=True)

    cycle = Cycle(monomer_indices=(0, 1, 2))

    return Molecule(
        structure_id="cyclic_trimer_mixed_dir",
        monomers=(m1, m2, m3),
        bonds=(bond1, bond2, bond3),
        cycles=(cycle,),
    )


# ============================================================================
# Tests: Fragment Class
# ============================================================================


class TestFragment:
    """Test Fragment dataclass validation and properties."""

    def test_fragment_creation(self):
        """Fragment should be creatable with monomer indices and bonds."""
        frag = Fragment(
            monomer_indices=(0, 1, 2),
            removed_bonds=(),
            is_from_cycle=False,
        )
        assert frag.monomer_count == 3
        assert frag.is_from_cycle is False

    def test_fragment_empty_fails(self):
        """Fragment with no monomers should raise ValueError."""
        with pytest.raises(ValueError, match="must contain at least one monomer"):
            Fragment(monomer_indices=(), removed_bonds=())

    def test_fragment_duplicate_indices_fails(self):
        """Fragment with duplicate indices should raise ValueError."""
        with pytest.raises(ValueError, match="must be unique"):
            Fragment(monomer_indices=(0, 1, 0), removed_bonds=())


# ============================================================================
# Tests: FragmentationGraph
# ============================================================================


class TestFragmentationGraph:
    """Test graph operations for fragmentation."""

    def test_graph_creation_linear(self, linear_trimer):
        """Build graph from linear molecule."""
        graph = FragmentationGraph(linear_trimer)
        assert graph.graph.number_of_nodes() == 3
        assert graph.graph.number_of_edges() == 2

    def test_bridges_in_linear_chain(self, linear_trimer):
        """Linear chain should have 2 bridges (both edges)."""
        graph = FragmentationGraph(linear_trimer)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 2

    def test_bridges_in_cycle(self, cyclic_trimer):
        """Cyclic structure should have no bridges."""
        graph = FragmentationGraph(cyclic_trimer)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 0

    def test_cycles_detection_cyclic(self, cyclic_trimer):
        """Cyclic molecule should detect at least one cycle."""
        graph = FragmentationGraph(cyclic_trimer)
        cycles = graph.find_all_cycles()
        assert len(cycles) > 0

    def test_cycles_detection_linear(self, linear_trimer):
        """Linear molecule should have no cycles."""
        graph = FragmentationGraph(linear_trimer)
        cycles = graph.find_all_cycles()
        assert len(cycles) == 0

    def test_2cut_pairs_in_cycle(self, cyclic_trimer):
        """3-node cycle should have 3 possible 2-cut pairs."""
        graph = FragmentationGraph(cyclic_trimer)
        cycle = [0, 1, 2]
        pairs = graph.extract_2cut_pairs(cycle)
        # For a 3-node cycle: C(3,2) = 3 pairs
        assert len(pairs) == 3

    def test_connected_components_after_removal(self, linear_trimer):
        """Removing a bridge should disconnect the graph."""
        graph = FragmentationGraph(linear_trimer)
        components = graph.get_connected_components_after_removal([(0, 1)])
        assert len(components) == 2

    def test_reversed_edge_orientation_still_detects_bridges(self, linear_trimer_reversed_edges):
        """Bridge detection should be direction-agnostic."""
        graph = FragmentationGraph(linear_trimer_reversed_edges)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 2

    def test_mixed_direction_cycle_is_still_detected(self, cyclic_trimer_mixed_directions):
        """Undirected cycle detection should work for mixed directed orientations."""
        graph = FragmentationGraph(cyclic_trimer_mixed_directions)
        cycles = graph.find_all_cycles()
        assert len(cycles) > 0


# ============================================================================
# Tests: MonomerMassCalculator
# ============================================================================


class TestMonomerMassCalculator:
    """Test mass calculation and caching."""

    def test_mass_lookup_ala(self, simple_monomer):
        """Alanine should have correct approximate mass."""
        calc = MonomerMassCalculator([simple_monomer])
        mass = calc.get_mass(0)
        # ALA exact mass ≈ 71.0372 Da
        assert pytest.approx(mass, rel=1e-3) == 71.0372

    def test_mass_lookup_gly(self):
        """Glycine should be lighter than Alanine."""
        m_gly = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
        )
        calc = MonomerMassCalculator([m_gly])
        mass_gly = calc.get_mass(0)
        assert mass_gly < 71.0  # Lighter than ALA

    def test_mass_caching(self, simple_monomer):
        """Repeated lookups should use cache."""
        calc = MonomerMassCalculator([simple_monomer])
        mass1 = calc.get_mass(0)
        mass2 = calc.get_mass(0)
        assert mass1 == mass2
        assert 0 in calc._mass_cache

    def test_mass_index_out_of_bounds(self, simple_monomer):
        """Invalid index should raise IndexError."""
        calc = MonomerMassCalculator([simple_monomer])
        with pytest.raises(IndexError):
            calc.get_mass(10)

    def test_unknown_residue_fails(self):
        """Unknown residue should raise ValueError."""
        m_unknown = Monomer(
            norine_name="UNKNOWN",
            residue="XYZ",
            is_methylated=False,
            chirality=Chirality.L,
        )
        calc = MonomerMassCalculator([m_unknown])
        with pytest.raises(ValueError):
            calc.get_mass(0)

    def test_atomic_mass_calculation_with_parsed_record(self):
        """Monomer mass calculated from atomic composition should be more accurate."""
        from monomer_names_helper import Chirality as ParsingChirality
        from monomer_names_helper import NorineMonomerName
        from rban_parsing.rban_parser import AtomInfo, MonomerInfo, Parsed_rBAN_Record

        # Create a mock Parsed_rBAN_Record with minimal data
        # Simulating a simple amino acid: Glycine core (C2NO2 with attached H)
        # Atoms: C, C, N(+2H), O, O(+1H)
        # Total: C2H3NO2 = 2*12.0 + 14.003074 + 2*15.994915 + 3*1.007825 ≈ 73.016

        monomer_info = MonomerInfo(
            name=NorineMonomerName("GLY"),
            atoms=[0, 1, 2, 3, 4],  # 5 atoms
            chirality=ParsingChirality.L,
            is_pks_hybrid=False,
        )

        atoms = {
            0: AtomInfo(name="C", hydrogens=0),  # C
            1: AtomInfo(name="C", hydrogens=0),  # C
            2: AtomInfo(name="N", hydrogens=2),  # N with 2 H
            3: AtomInfo(name="O", hydrogens=0),  # O
            4: AtomInfo(name="O", hydrogens=1),  # O with 1 H
        }

        # Construct the mock parsed record
        mock_record = Parsed_rBAN_Record.__new__(Parsed_rBAN_Record)
        mock_record.compound_id = "GLY_TEST"
        mock_record.monomers = {0: monomer_info}
        mock_record.atoms = atoms
        mock_record.monomer_bonds = {}
        mock_record.atomic_bonds = {}
        mock_record.metadata = None  # type: ignore

        # Create monomer and calculator with parsed_record
        m_gly = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
        )
        calc = MonomerMassCalculator([m_gly], parsed_record=mock_record)

        # Get mass from atomic composition
        mass = calc.get_mass(0)

        # Expected mass: 2*C + 3*H + 1*N + 2*O
        # = 2*12.0 + 3*1.007825 + 14.003074 + 2*15.994915 ≈ 73.016
        expected_mass = (
            2 * 12.0  # 2 carbons
            + 3 * 1.007825  # 3 hydrogens (2 from N, 1 from O)
            + 14.003074  # 1 nitrogen
            + 2 * 15.994915  # 2 oxygens
        )

        assert pytest.approx(mass, rel=1e-2) == expected_mass

    def test_atomic_mass_fallback_to_residue(self):
        """If parsed_record is None, should fall back to residue lookup."""
        m_ala = Monomer(
            norine_name="ALA",
            residue="ALA",
            is_methylated=False,
            chirality=Chirality.L,
        )
        calc = MonomerMassCalculator([m_ala], parsed_record=None)

        mass = calc.get_mass(0)
        # Should use residue lookup
        assert pytest.approx(mass, rel=1e-3) == 71.0372

    def test_atomic_mass_unknown_element_raises_error(self):
        """Unknown element symbol should raise ValueError."""
        from monomer_names_helper import Chirality as ParsingChirality
        from monomer_names_helper import NorineMonomerName
        from rban_parsing.rban_parser import AtomInfo, MonomerInfo, Parsed_rBAN_Record

        monomer_info = MonomerInfo(
            name=NorineMonomerName("UNKNOWN"),
            atoms=[0],
            chirality=ParsingChirality.L,
            is_pks_hybrid=False,
        )

        atoms = {0: AtomInfo(name="Xx", hydrogens=0)}  # Unknown element

        mock_record = Parsed_rBAN_Record.__new__(Parsed_rBAN_Record)
        mock_record.compound_id = "UNKNOWN_TEST"
        mock_record.monomers = {0: monomer_info}
        mock_record.atoms = atoms
        mock_record.monomer_bonds = {}
        mock_record.atomic_bonds = {}
        mock_record.metadata = None  # type: ignore

        m_unknown = Monomer(
            norine_name="UNKNOWN",
            residue="UNKNOWN",
            is_methylated=False,
            chirality=Chirality.L,
        )
        calc = MonomerMassCalculator([m_unknown], parsed_record=mock_record)

        with pytest.raises(ValueError, match="Unknown element"):
            calc.get_mass(0)


# ============================================================================
# Tests: Fragment Generation
# ============================================================================


class TestFragmentGeneration:
    """Test fragment generation from molecules."""

    def test_single_monomer_no_fragments(self, simple_monomer):
        """Single monomer should generate 1 trivial fragment."""
        mol = Molecule(structure_id="single", monomers=(simple_monomer,), bonds=())
        fragments = generate_fragments(mol)
        assert len(fragments) == 1
        assert fragments[0].monomer_indices == (0,)

    def test_linear_dimer_fragments(self, linear_dimer):
        """Linear dimer should generate fragments from bridge cut."""
        fragments = generate_fragments(linear_dimer)
        # Should have fragments: [0], [1]
        assert len(fragments) >= 1
        monomer_sets = [set(f.monomer_indices) for f in fragments]
        assert {0} in monomer_sets
        assert {1} in monomer_sets

    def test_linear_trimer_fragments(self, linear_trimer):
        """Linear trimer should generate multiple fragments from bridge cuts.

        For A-B-C linear structure:
        - Cutting bridge 0-1: generates [A] and [B-C]
        - Cutting bridge 1-2: generates [A-B] and [C]
        So we expect: {0}, {1,2}, {0,1}, {2}
        Note: middle monomer B cannot be isolated with single bridge cuts.
        """
        fragments = generate_fragments(linear_trimer)
        assert len(fragments) >= 3
        # Should contain end monomers and combinations
        monomer_sets = [set(f.monomer_indices) for f in fragments]
        assert {0} in monomer_sets  # A alone
        assert {2} in monomer_sets  # C alone
        assert {0, 1} in monomer_sets or {1, 2} in monomer_sets  # A-B or B-C

    def test_cyclic_trimer_2cut_fragments(self, cyclic_trimer):
        """Cyclic trimer should generate fragments from 2-cuts."""
        fragments = generate_fragments(cyclic_trimer)
        assert len(fragments) > 0
        # All should be linear after 2-cuts
        assert all(not f.is_from_cycle for f in fragments) or all(
            f.is_from_cycle for f in fragments
        )

    def test_mixed_structure_fragments(self, mixed_structure):
        """Mixed linear+cyclic should generate diverse fragments."""
        fragments = generate_fragments(mixed_structure)
        assert len(fragments) > 0

    def test_reversed_linear_edges_generate_expected_fragments(self, linear_trimer_reversed_edges):
        """Linear fragments should be identical regardless of bond orientation."""
        fragments = generate_fragments(linear_trimer_reversed_edges)
        monomer_sets = [set(f.monomer_indices) for f in fragments]
        assert {0} in monomer_sets
        assert {2} in monomer_sets

    def test_mixed_direction_cycle_generates_2cut_fragments(self, cyclic_trimer_mixed_directions):
        """2-cut fragments should be produced even without a directed cycle."""
        fragments = generate_fragments(cyclic_trimer_mixed_directions)
        assert len(fragments) > 0


# ============================================================================
# Tests: Mass Calculation
# ============================================================================


class TestMassCalculation:
    """Test theoretical mass calculations."""

    def test_single_monomer_mass(self, simple_monomer):
        """Single monomer mass should be its residue mass."""
        calc = MonomerMassCalculator([simple_monomer])
        frag = Fragment(monomer_indices=(0,), removed_bonds=())
        masses = calculate_theoretical_masses(
            frag, calc, ion_types={"[M+H]+", "[M-H]-", "fragment"}
        )

        assert len(masses) == 3  # [M+H]+, [M-H]-, neutral
        ion_types = {ion_type for _, ion_type in masses}
        assert ion_types == {"[M+H]+", "[M-H]-", "fragment"}

    def test_protonation_adds_mass(self, simple_monomer):
        """[M+H]+ should be heavier than neutral fragment."""
        calc = MonomerMassCalculator([simple_monomer])
        frag = Fragment(monomer_indices=(0,), removed_bonds=())
        masses_dict = {
            ion_type: mass
            for mass, ion_type in calculate_theoretical_masses(
                frag, calc, ion_types={"[M+H]+", "[M-H]-", "fragment"}
            )
        }

        assert masses_dict["[M+H]+"] > masses_dict["fragment"]

    def test_deprotonation_reduces_mass(self, simple_monomer):
        """[M-H]- should be lighter than neutral fragment."""
        calc = MonomerMassCalculator([simple_monomer])
        frag = Fragment(monomer_indices=(0,), removed_bonds=())
        masses_dict = {
            ion_type: mass
            for mass, ion_type in calculate_theoretical_masses(
                frag, calc, ion_types={"[M+H]+", "[M-H]-", "fragment"}
            )
        }

        assert masses_dict["[M-H]-"] < masses_dict["fragment"]


# ============================================================================
# Tests: Integration
# ============================================================================


class TestIntegration:
    """End-to-end tests combining fragmentation and mass calculation."""

    def test_full_pipeline_linear_dimer(self, linear_dimer):
        """Generate fragments and masses for linear dimer."""
        masses = generate_all_theoretical_masses(
            linear_dimer, ion_types={"[M+H]+", "[M-H]-", "fragment"}
        )
        assert len(masses) == 6  # 2 * 3
        # Each mass entry is a TheoreticalIon
        assert all(isinstance(m, TheoreticalIon) for m in masses)
        # Verify masses are sorted
        mass_values = [m.mass for m in masses]
        assert mass_values == sorted(mass_values)

    def test_full_pipeline_cyclic_trimer(self, cyclic_trimer):
        """Generate fragments and masses for cyclic structure."""
        masses = generate_all_theoretical_masses(
            cyclic_trimer, ion_types={"[M+H]+", "[M-H]-", "fragment"}
        )
        assert len(masses) == 18  # 6 * 3

    def test_all_masses_positive(self, linear_trimer):
        """All theoretical masses should be positive."""
        masses = generate_all_theoretical_masses(
            linear_trimer, ion_types={"[M+H]+", "[M-H]-", "fragment"}
        )
        mass_values = [m.mass for m in masses]
        assert all(m > 0 for m in mass_values)

    def test_fragment_count_reasonable(self, linear_trimer):
        """Number of masses should be ≥ number of fragments * 3 (3 ion types)."""
        fragments = generate_fragments(linear_trimer)
        masses = generate_all_theoretical_masses(
            linear_trimer, ion_types={"[M+H]+", "[M-H]-", "fragment"}
        )
        # At least 3 masses per fragment
        assert len(masses) >= 3 * len(fragments)

    def test_rule_based_bond_filtering(self):
        """Verify that FragmentationRule rules successfully filter allowed cleavages."""
        from kakapo.constants import PROTON_MASS
        from kakapo.core.fragmenter import FragmentationRule
        from kakapo.models import Bond, Chirality, Molecule, Monomer

        m1 = Monomer(norine_name="ALA", residue="ALA", is_methylated=False, chirality=Chirality.L)
        m2 = Monomer(norine_name="GLY", residue="GLY", is_methylated=False, chirality=Chirality.L)
        m3 = Monomer(norine_name="PRO", residue="PRO", is_methylated=False, chirality=Chirality.L)

        # bond1 is PEPTIDE, bond2 is ESTER
        bond1 = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE", is_ring_bond=False)
        bond2 = Bond(source_idx=1, target_idx=2, bond_type="ESTER", is_ring_bond=False)

        mol = Molecule(
            structure_id="mixed_bonds",
            monomers=(m1, m2, m3),
            bonds=(bond1, bond2),
            cycles=(),
        )

        # When rules is restricted to only PEPTIDE, bond2 cannot be cleaved.
        # So we can only cut bond1. This generates two fragments: [0] (ALA) and [1, 2] (GLY-PRO).
        rules = (FragmentationRule("PEPTIDE", PROTON_MASS, -PROTON_MASS),)
        fragments = generate_fragments(mol, rules=rules)
        monomer_sets = {frozenset(f.monomer_indices) for f in fragments}
        assert frozenset({0}) in monomer_sets
        assert frozenset({1, 2}) in monomer_sets
        # Monomer 2 (PRO) should not be isolated because ESTER bond is not cleaved
        assert frozenset({2}) not in monomer_sets

    def test_rule_based_cleavage_shifts(self, linear_dimer):
        """Verify that rule-based cleavage shifts are correctly applied to fragment masses."""
        calc = MonomerMassCalculator(list(linear_dimer.monomers))
        fragments = generate_fragments(linear_dimer)

        # Find fragments corresponding to single monomer 0 (ALA) and monomer 1 (GLY)
        frag_ala = next(f for f in fragments if f.monomer_indices == (0,))
        frag_gly = next(f for f in fragments if f.monomer_indices == (1,))

        # Standard proton mass
        from kakapo.constants import PROTON_MASS

        # ALA is the source monomer of the PEPTIDE bond (gets -H right shift)
        # GLY is the target monomer of the PEPTIDE bond (gets +H left shift)
        masses_ala = calculate_theoretical_masses(frag_ala, calc, ion_types={"fragment"})
        masses_gly = calculate_theoretical_masses(frag_gly, calc, ion_types={"fragment"})

        neutral_ala = next(m for m, t in masses_ala if t == "fragment")
        neutral_gly = next(m for m, t in masses_gly if t == "fragment")

        expected_ala = calc.get_mass(0) - PROTON_MASS
        expected_gly = calc.get_mass(1) + PROTON_MASS

        assert pytest.approx(neutral_ala) == expected_ala
        assert pytest.approx(neutral_gly) == expected_gly

    def test_generate_theoretical_spectrum(self, linear_dimer):
        """Verify that theoretical spectrum generation works as expected."""
        theo_spec = generate_theoretical_spectrum(
            linear_dimer, ion_types={"[M+H]+", "[M-H]-", "fragment"}
        )
        assert isinstance(theo_spec, TheoreticalSpectrum)
        assert theo_spec.structure_id == "linear_dimer"
        assert len(theo_spec.ions) == 6

        # Check sorted ions helper
        sorted_ions = theo_spec.get_sorted_ions()
        assert len(sorted_ions) == 6
        assert sorted_ions == sorted(theo_spec.ions, key=lambda ion: ion.mass)

    def test_theoretical_spectrum_validation(self):
        """Test validation rules for TheoreticalSpectrum and TheoreticalIon."""
        # Empty fragment ID validation
        with pytest.raises(ValueError, match="Fragment ID cannot be empty"):
            TheoreticalIon(fragment_id="", mass=100.0, ion_type="[M+H]+")

        # Negative mass validation
        with pytest.raises(ValueError, match="Theoretical ion mass must be non-negative"):
            TheoreticalIon(fragment_id="frag_0", mass=-10.0, ion_type="[M+H]+")

        # Empty ion type validation
        with pytest.raises(ValueError, match="Ion type cannot be empty"):
            TheoreticalIon(fragment_id="frag_0", mass=100.0, ion_type="")

        # Empty structure ID for spectrum
        ion = TheoreticalIon(fragment_id="frag_0", mass=100.0, ion_type="[M+H]+")
        with pytest.raises(ValueError, match="Structure ID cannot be empty"):
            TheoreticalSpectrum(structure_id="", ions=frozenset({ion}))

        # Empty ions set for spectrum
        msg = "Theoretical spectrum must contain at least one ion"
        with pytest.raises(ValueError, match=msg):
            TheoreticalSpectrum(structure_id="spec_1", ions=frozenset())

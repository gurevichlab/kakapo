"""Unit tests for the spectrum fragmenter.

Comprehensive test coverage for:
- Fragment generation (bridges and 2-cuts)
- Mass calculations without water subtraction
- Ion type generation ([M+H]+, [M-H]-, neutral)
- Edge cases (single monomer, cycles, complex structures)
"""

import pytest

from kakapo.constants import H_MASS, FragmentationRule
from kakapo.core.fragmenter import (
    Fragment,
    FragmentationGraph,
    calculate_theoretical_ion,
    generate_all_theoretical_masses,
    generate_fragments,
    generate_theoretical_spectrum,
)
from kakapo.models import (
    Bond,
    BondSubtype,
    Chirality,
    Molecule,
    Monomer,
    RuleShift,
    TheoreticalIon,
    TheoreticalSpectrum,
)

# Explicit rules to use in tests to avoid relying on default rules
TEST_RULES = (
    FragmentationRule("PEPTIDE", H_MASS, -H_MASS),
    FragmentationRule("ESTER", H_MASS, -H_MASS),
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
        mass=71.03718,
    )


@pytest.fixture
def linear_dimer():
    """Two monomers in a linear chain (A-B)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
        mass=71.03718,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
        mass=57.02146,
    )
    bond = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE")

    return Molecule(
        structure_id="linear_dimer",
        monomers={0: m1, 1: m2},
        bonds=(bond,),
    )


@pytest.fixture
def linear_trimer():
    """Three monomers in a linear chain (A-B-C)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
        mass=71.03718,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
        mass=57.02146,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
        mass=97.05276,
    )

    bond1 = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE")
    bond2 = Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE")

    return Molecule(
        structure_id="linear_trimer",
        monomers={0: m1, 1: m2, 2: m3},
        bonds=(bond1, bond2),
    )


@pytest.fixture
def cyclic_trimer():
    """Three monomers in a cycle (A-B-C-A)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
        mass=71.03718,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
        mass=57.02146,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
        mass=97.05276,
    )

    bond1 = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE")
    bond2 = Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE")
    bond3 = Bond(source_idx=2, target_idx=0, bond_type="PEPTIDE")

    return Molecule(
        structure_id="cyclic_trimer",
        monomers={0: m1, 1: m2, 2: m3},
        bonds=(bond1, bond2, bond3),
    )


@pytest.fixture
def mixed_structure():
    """Mixed: 2 linear monomers + 3 cyclic monomers (L1-L2-C1-C2-C3-C1)."""
    names = ["ALA", "GLY", "PRO", "SER", "THR"]
    mass_values = {
        "ALA": 71.03718,
        "GLY": 57.02146,
        "PRO": 97.05276,
        "SER": 87.03203,
        "THR": 101.04768,
    }
    monomers = {
        i: Monomer(
            norine_name=names[i],
            residue=names[i],
            is_methylated=False,
            chirality=Chirality.L,
            mass=mass_values[names[i]],
        )
        for i in range(5)
    }

    bonds = [
        Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE"),  # L1-L2
        Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE"),  # L2-C1
        Bond(source_idx=2, target_idx=3, bond_type="PEPTIDE"),  # C1-C2
        Bond(source_idx=3, target_idx=4, bond_type="PEPTIDE"),  # C2-C3
        Bond(source_idx=4, target_idx=2, bond_type="PEPTIDE"),  # C3-C1 (ring)
    ]

    return Molecule(
        structure_id="mixed_structure",
        monomers=monomers,
        bonds=tuple(bonds),
    )


@pytest.fixture
def linear_trimer_reversed_edges():
    """Three monomers in a linear chain with reversed bond orientation."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
        mass=71.03718,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
        mass=57.02146,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
        mass=97.05276,
    )

    # Reverse orientation versus index order: 1->0 and 2->1.
    bond1 = Bond(source_idx=1, target_idx=0, bond_type="PEPTIDE")
    bond2 = Bond(source_idx=2, target_idx=1, bond_type="PEPTIDE")

    return Molecule(
        structure_id="linear_trimer_reversed",
        monomers={0: m1, 1: m2, 2: m3},
        bonds=(bond1, bond2),
    )


@pytest.fixture
def cyclic_trimer_mixed_directions():
    """3-cycle with mixed edge directions (no directed cycle, still undirected cycle)."""
    m1 = Monomer(
        norine_name="ALA",
        residue="ALA",
        is_methylated=False,
        chirality=Chirality.L,
        mass=71.03718,
    )
    m2 = Monomer(
        norine_name="GLY",
        residue="GLY",
        is_methylated=False,
        chirality=Chirality.L,
        mass=57.02146,
    )
    m3 = Monomer(
        norine_name="PRO",
        residue="PRO",
        is_methylated=False,
        chirality=Chirality.L,
        mass=97.05276,
    )

    bond1 = Bond(source_idx=1, target_idx=0, bond_type="PEPTIDE")
    bond2 = Bond(source_idx=1, target_idx=2, bond_type="PEPTIDE")
    bond3 = Bond(source_idx=0, target_idx=2, bond_type="PEPTIDE")

    return Molecule(
        structure_id="cyclic_trimer_mixed_dir",
        monomers={0: m1, 1: m2, 2: m3},
        bonds=(bond1, bond2, bond3),
    )


# ============================================================================
# Tests: Fragment Class
# ============================================================================


class TestFragment:
    """Test Fragment dataclass validation and properties."""

    def test_fragment_creation(self):
        """Fragment should be creatable with monomer indices and bonds."""
        frag = Fragment(
            fragment_id="frag",
            monomer_indices=(0, 1, 2),
            removed_bonds=(),
            is_from_cycle=False,
        )
        assert frag.monomer_count == 3
        assert frag.is_from_cycle is False
        assert frag.fragment_id == "frag"

    def test_fragment_id_custom(self):
        """Fragment should store a custom fragment_id correctly."""
        frag = Fragment(
            fragment_id="custom_id",
            monomer_indices=(0, 1),
            removed_bonds=(),
        )
        assert frag.fragment_id == "custom_id"

    def test_fragment_empty_fails(self):
        """Fragment with no monomers should raise ValueError."""
        with pytest.raises(ValueError, match="must contain at least one monomer"):
            Fragment(fragment_id="frag", monomer_indices=(), removed_bonds=())

    def test_fragment_duplicate_indices_fails(self):
        """Fragment with duplicate indices should raise ValueError."""
        with pytest.raises(ValueError, match="must be unique"):
            Fragment(fragment_id="frag", monomer_indices=(0, 1, 0), removed_bonds=())


# ============================================================================
# Tests: FragmentationGraph
# ============================================================================


class TestFragmentationGraph:
    """Test graph operations for fragmentation."""

    def test_graph_creation_linear(self, linear_trimer):
        """Build graph from linear molecule."""
        graph = FragmentationGraph(linear_trimer, rules=TEST_RULES)
        assert graph.graph.number_of_nodes() == 3
        assert graph.graph.number_of_edges() == 2

    def test_bridges_in_linear_chain(self, linear_trimer):
        """Linear chain should have 2 bridges (both edges)."""
        graph = FragmentationGraph(linear_trimer, rules=TEST_RULES)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 2

    def test_bridges_in_cycle(self, cyclic_trimer):
        """Cyclic structure should have no bridges."""
        graph = FragmentationGraph(cyclic_trimer, rules=TEST_RULES)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 0

    def test_cycles_detection_cyclic(self, cyclic_trimer):
        """Cyclic molecule should detect at least one cycle."""
        graph = FragmentationGraph(cyclic_trimer, rules=TEST_RULES)
        cycles = graph.find_all_cycles()
        assert len(cycles) > 0

    def test_cycles_detection_linear(self, linear_trimer):
        """Linear molecule should have no cycles."""
        graph = FragmentationGraph(linear_trimer, rules=TEST_RULES)
        cycles = graph.find_all_cycles()
        assert len(cycles) == 0

    def test_2cut_pairs_in_cycle(self, cyclic_trimer):
        """3-node cycle should have 3 possible 2-cut pairs."""
        graph = FragmentationGraph(cyclic_trimer, rules=TEST_RULES)
        cycle = [0, 1, 2]
        pairs = graph.extract_2cut_pairs(cycle)
        assert len(pairs) == 3

    def test_connected_components_after_removal(self, linear_trimer):
        """Removing a bridge should disconnect the graph."""
        graph = FragmentationGraph(linear_trimer, rules=TEST_RULES)
        components = graph.get_connected_components_after_removal([(0, 1)])
        assert len(components) == 2

    def test_reversed_edge_orientation_still_detects_bridges(self, linear_trimer_reversed_edges):
        """Bridge detection should be direction-agnostic."""
        graph = FragmentationGraph(linear_trimer_reversed_edges, rules=TEST_RULES)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 2

    def test_mixed_direction_cycle_is_still_detected(self, cyclic_trimer_mixed_directions):
        """Undirected cycle detection should work for mixed directed orientations."""
        graph = FragmentationGraph(cyclic_trimer_mixed_directions, rules=TEST_RULES)
        cycles = graph.find_all_cycles()
        assert len(cycles) > 0


# ============================================================================
# Tests: Mass Calculation Lookups and Errors
# ============================================================================


class TestMassCalculationLookups:
    """Test mass lookup validation and errors."""

    def test_mass_index_out_of_bounds(self, simple_monomer):
        """Invalid index should raise IndexError."""
        frag = Fragment(fragment_id="frag", monomer_indices=(10,), removed_bonds=())
        with pytest.raises(IndexError, match="out of bounds"):
            calculate_theoretical_ion(frag, {0: simple_monomer}, rules=TEST_RULES)

    def test_unknown_residue_fails(self):
        """Monomer with None mass should raise ValueError."""
        m_unknown = Monomer(
            norine_name="UNKNOWN",
            residue="XYZ",
            is_methylated=False,
            chirality=Chirality.L,
            mass=None,
        )
        frag = Fragment(fragment_id="frag", monomer_indices=(0,), removed_bonds=())
        with pytest.raises(ValueError, match="Cannot determine mass"):
            calculate_theoretical_ion(frag, {0: m_unknown}, rules=TEST_RULES)

    def test_atomic_mass_calculation_with_parsed_record(self):
        """Monomer mass precalculated from atomic composition should be correct."""
        expected_mass = (
            2 * 12.0  # 2 carbons
            + 3 * 1.007825  # 3 hydrogens
            + 14.003074  # 1 nitrogen
            + 2 * 15.994915  # 2 oxygens
        )
        m_gly = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
            mass=expected_mass,
        )
        frag = Fragment(fragment_id="frag", monomer_indices=(0,), removed_bonds=())
        ion = calculate_theoretical_ion(frag, {0: m_gly}, rules=TEST_RULES)
        assert pytest.approx(ion.mass, rel=1e-2) == expected_mass


# ============================================================================
# Tests: Fragment Generation
# ============================================================================


class TestFragmentGeneration:
    """Test fragment generation from molecules."""

    def test_single_monomer_no_fragments(self, simple_monomer):
        """Single monomer should generate 1 trivial fragment."""
        mol = Molecule(structure_id="single", monomers={0: simple_monomer}, bonds=())
        fragments = generate_fragments(mol, rules=TEST_RULES)
        assert len(fragments) == 1
        assert fragments[0].monomer_indices == (0,)
        assert fragments[0].fragment_id == "frag_0"

    def test_linear_dimer_fragments(self, linear_dimer):
        """Linear dimer should generate fragments from bridge cut."""
        fragments = generate_fragments(linear_dimer, rules=TEST_RULES)
        assert len(fragments) >= 1
        monomer_sets = [set(f.monomer_indices) for f in fragments]
        assert {0} in monomer_sets
        assert {1} in monomer_sets

    def test_linear_trimer_fragments(self, linear_trimer):
        """Linear trimer should generate multiple fragments from bridge cuts."""
        fragments = generate_fragments(linear_trimer, rules=TEST_RULES)
        assert len(fragments) >= 3
        monomer_sets = [set(f.monomer_indices) for f in fragments]
        assert {0} in monomer_sets
        assert {2} in monomer_sets
        assert {0, 1} in monomer_sets or {1, 2} in monomer_sets

    def test_cyclic_trimer_2cut_fragments(self, cyclic_trimer):
        """Cyclic trimer should generate fragments from 2-cuts."""
        fragments = generate_fragments(cyclic_trimer, rules=TEST_RULES)
        assert len(fragments) > 0

    def test_mixed_structure_fragments(self, mixed_structure):
        """Mixed linear+cyclic should generate diverse fragments."""
        fragments = generate_fragments(mixed_structure, rules=TEST_RULES)
        assert len(fragments) > 0

    def test_reversed_linear_edges_generate_expected_fragments(self, linear_trimer_reversed_edges):
        """Linear fragments should be identical regardless of bond orientation."""
        fragments = generate_fragments(linear_trimer_reversed_edges, rules=TEST_RULES)
        monomer_sets = [set(f.monomer_indices) for f in fragments]
        assert {0} in monomer_sets
        assert {2} in monomer_sets

    def test_mixed_direction_cycle_generates_2cut_fragments(self, cyclic_trimer_mixed_directions):
        """2-cut fragments should be produced even without a directed cycle."""
        fragments = generate_fragments(cyclic_trimer_mixed_directions, rules=TEST_RULES)
        assert len(fragments) > 0


# ============================================================================
# Tests: Mass Calculation
# ============================================================================


class TestMassCalculation:
    """Test theoretical mass calculations."""

    def test_single_monomer_mass(self, simple_monomer):
        """Single monomer mass should be its residue mass."""
        frag = Fragment(fragment_id="frag", monomer_indices=(0,), removed_bonds=())
        ion = calculate_theoretical_ion(frag, {0: simple_monomer}, rules=TEST_RULES)
        assert pytest.approx(ion.mass, rel=1e-3) == 71.0372
        assert ion.rules_shifts == ()

    def test_rules_shifts_population(self, linear_dimer):
        """Verify that rules_shifts tuple contains RuleShift instances mapped to their shifts."""
        fragments = generate_fragments(linear_dimer, rules=TEST_RULES)
        frag_ala = next(f for f in fragments if f.monomer_indices == (0,))
        ion_ala = calculate_theoretical_ion(frag_ala, linear_dimer.monomers, rules=TEST_RULES)

        assert len(ion_ala.rules_shifts) == 1
        assert ion_ala.rules_shifts[0].rule_name == "PEPTIDE"
        assert pytest.approx(ion_ala.rules_shifts[0].shift) == -H_MASS


# ============================================================================
# Tests: Integration
# ============================================================================


class TestIntegration:
    """End-to-end tests combining fragmentation and mass calculation."""

    def test_full_pipeline_linear_dimer(self, linear_dimer):
        """Generate fragments and masses for linear dimer."""
        masses = generate_all_theoretical_masses(linear_dimer, rules=TEST_RULES)
        assert (
            len(masses) == 2
        )  # A dimer has 2 fragments: {0} and {1} (since we cut the single bridge)
        assert all(isinstance(m, TheoreticalIon) for m in masses)
        mass_values = [m.mass for m in masses]
        assert mass_values == sorted(mass_values)

    def test_full_pipeline_cyclic_trimer(self, cyclic_trimer):
        """Generate fragments and masses for cyclic structure."""
        masses = generate_all_theoretical_masses(cyclic_trimer, rules=TEST_RULES)
        assert (
            len(masses) == 6
        )  # 3 fragments of size 1, 3 of size 2. Total 6 unique fragments from 2-cuts.

    def test_all_masses_positive(self, linear_trimer):
        """All theoretical masses should be positive."""
        masses = generate_all_theoretical_masses(linear_trimer, rules=TEST_RULES)
        mass_values = [m.mass for m in masses]
        assert all(m > 0 for m in mass_values)

    def test_fragment_count_reasonable(self, linear_trimer):
        """Number of masses should equal the number of fragments."""
        fragments = generate_fragments(linear_trimer, rules=TEST_RULES)
        masses = generate_all_theoretical_masses(linear_trimer, rules=TEST_RULES)
        assert len(masses) == len(fragments)

    def test_rule_based_bond_filtering(self):
        """Verify that FragmentationRule rules successfully filter allowed cleavages."""
        m1 = Monomer(
            norine_name="ALA",
            residue="ALA",
            is_methylated=False,
            chirality=Chirality.L,
            mass=71.03718,
        )
        m2 = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
            mass=57.02146,
        )
        m3 = Monomer(
            norine_name="PRO",
            residue="PRO",
            is_methylated=False,
            chirality=Chirality.L,
            mass=97.05276,
        )

        bond1 = Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE")
        bond2 = Bond(source_idx=1, target_idx=2, bond_type="ESTER")

        mol = Molecule(
            structure_id="mixed_bonds",
            monomers={0: m1, 1: m2, 2: m3},
            bonds=(bond1, bond2),
        )

        rules = (FragmentationRule("PEPTIDE", H_MASS, -H_MASS),)
        fragments = generate_fragments(mol, rules=rules)
        monomer_sets = {frozenset(f.monomer_indices) for f in fragments}
        assert frozenset({0}) in monomer_sets
        assert frozenset({1, 2}) in monomer_sets
        assert frozenset({2}) not in monomer_sets

    def test_rule_based_cleavage_shifts(self, linear_dimer):
        """Verify that rule-based cleavage shifts are correctly applied to fragment masses."""
        fragments = generate_fragments(linear_dimer, rules=TEST_RULES)

        frag_ala = next(f for f in fragments if f.monomer_indices == (0,))
        frag_gly = next(f for f in fragments if f.monomer_indices == (1,))

        ion_ala = calculate_theoretical_ion(frag_ala, linear_dimer.monomers, rules=TEST_RULES)
        ion_gly = calculate_theoretical_ion(frag_gly, linear_dimer.monomers, rules=TEST_RULES)

        expected_ala = linear_dimer.monomers[0].mass - H_MASS
        expected_gly = linear_dimer.monomers[1].mass + H_MASS

        assert pytest.approx(ion_ala.mass) == expected_ala
        assert pytest.approx(ion_gly.mass) == expected_gly

    def test_generate_theoretical_spectrum(self, linear_dimer):
        """Verify that theoretical spectrum generation works as expected."""
        theo_spec = generate_theoretical_spectrum(linear_dimer, rules=TEST_RULES)
        assert isinstance(theo_spec, TheoreticalSpectrum)
        assert theo_spec.structure_id == "linear_dimer"
        assert len(theo_spec.ions) == 2
        assert pytest.approx(theo_spec.structure_mass) == 128.05864
        assert theo_spec.monomer_count == 2

        sorted_ions = theo_spec.get_sorted_ions()
        assert len(sorted_ions) == 2
        assert sorted_ions == sorted(theo_spec.ions, key=lambda ion: ion.mass)

    def test_theoretical_spectrum_validation(self):
        """Test validation rules for TheoreticalSpectrum and TheoreticalIon."""
        with pytest.raises(ValueError, match="Fragment ID cannot be empty"):
            TheoreticalIon(fragment_id="", mass=100.0)

        with pytest.raises(ValueError, match="Theoretical ion mass must be non-negative"):
            TheoreticalIon(fragment_id="frag_0", mass=-10.0)

        ion = TheoreticalIon(
            fragment_id="frag_0", mass=100.0, rules_shifts=(RuleShift("PEPTIDE", 1.0),)
        )
        # Test hashability and set operations
        ions_set = {ion}
        assert ion in ions_set

        with pytest.raises(ValueError, match="Structure ID cannot be empty"):
            TheoreticalSpectrum(
                structure_id="",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=1,
            )

        msg = "Theoretical spectrum must contain at least one ion"
        with pytest.raises(ValueError, match=msg):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset(),
                structure_mass=100.0,
                monomer_count=1,
            )

        with pytest.raises(ValueError, match="Structure mass must be non-negative"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=-1.0,
                monomer_count=1,
            )

        with pytest.raises(ValueError, match="Monomer count must be positive"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=0,
            )

    def test_rule_based_bond_filtering_with_subtypes(self):
        """Verify that FragmentationRule rules filter cleavages based on bond subtypes."""
        m1 = Monomer(
            norine_name="ALA",
            residue="ALA",
            is_methylated=False,
            chirality=Chirality.L,
            mass=71.03718,
        )
        m2 = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
            mass=57.02146,
        )
        m3 = Monomer(
            norine_name="PRO",
            residue="PRO",
            is_methylated=False,
            chirality=Chirality.L,
            mass=97.05276,
        )

        # bond1 has no matching bond_type, but has a matching subtype "amide_1"
        subtype1 = BondSubtype(name="amide_1", reversed=False)
        bond1 = Bond(source_idx=0, target_idx=1, bond_type="OTHER", subtypes=(subtype1,))
        # bond2 has no matching bond_type or subtype
        bond2 = Bond(source_idx=1, target_idx=2, bond_type="ESTER")

        mol = Molecule(
            structure_id="subtype_bonds",
            monomers={0: m1, 1: m2, 2: m3},
            bonds=(bond1, bond2),
        )

        rules = (FragmentationRule("amide_1", H_MASS, -H_MASS),)
        fragments = generate_fragments(mol, rules=rules)
        monomer_sets = {frozenset(f.monomer_indices) for f in fragments}
        # bond1 should be cleaved since its subtype "amide_1" matches the rule.
        # bond2 should NOT be cleaved.
        assert frozenset({0}) in monomer_sets
        assert frozenset({1, 2}) in monomer_sets
        assert frozenset({2}) not in monomer_sets

    def test_subtype_cleavage_shifts(self):
        """Verify that rule-based shifts are correctly applied when matching via subtypes."""
        m1 = Monomer(
            norine_name="ALA",
            residue="ALA",
            is_methylated=False,
            chirality=Chirality.L,
            mass=71.03718,
        )
        m2 = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
            mass=57.02146,
        )

        subtype1 = BondSubtype(name="amide_1", reversed=False)
        bond = Bond(source_idx=0, target_idx=1, bond_type="OTHER", subtypes=(subtype1,))

        mol = Molecule(
            structure_id="subtype_shifts",
            monomers={0: m1, 1: m2},
            bonds=(bond,),
        )

        rules = (FragmentationRule("amide_1", 1.0, -1.0),)
        fragments = generate_fragments(mol, rules=rules)

        frag_0 = next(f for f in fragments if f.monomer_indices == (0,))
        frag_1 = next(f for f in fragments if f.monomer_indices == (1,))

        ion_0 = calculate_theoretical_ion(frag_0, mol.monomers, rules=rules)
        ion_1 = calculate_theoretical_ion(frag_1, mol.monomers, rules=rules)

        # target of bond (index 1) is in frag_1 -> receives left_shift (+1.0)
        # source of bond (index 0) is in frag_0 -> receives right_shift (-1.0)
        assert pytest.approx(ion_0.mass) == 71.03718 - 1.0
        assert pytest.approx(ion_1.mass) == 57.02146 + 1.0

    def test_subtype_reversed_cleavage_shifts(self):
        """Verify that mass shifts are inverted when the matching subtype is reversed."""
        m1 = Monomer(
            norine_name="ALA",
            residue="ALA",
            is_methylated=False,
            chirality=Chirality.L,
            mass=71.03718,
        )
        m2 = Monomer(
            norine_name="GLY",
            residue="GLY",
            is_methylated=False,
            chirality=Chirality.L,
            mass=57.02146,
        )

        # reversed = True
        subtype1 = BondSubtype(name="amide_1", reversed=True)
        bond = Bond(source_idx=0, target_idx=1, bond_type="OTHER", subtypes=(subtype1,))

        mol = Molecule(
            structure_id="subtype_shifts_reversed",
            monomers={0: m1, 1: m2},
            bonds=(bond,),
        )

        rules = (FragmentationRule("amide_1", 1.0, -1.0),)
        fragments = generate_fragments(mol, rules=rules)

        frag_0 = next(f for f in fragments if f.monomer_indices == (0,))
        frag_1 = next(f for f in fragments if f.monomer_indices == (1,))

        ion_0 = calculate_theoretical_ion(frag_0, mol.monomers, rules=rules)
        ion_1 = calculate_theoretical_ion(frag_1, mol.monomers, rules=rules)

        # Since reversed is True:
        # left_shift (+1.0) is applied to source (index 0) -> frag_0
        # right_shift (-1.0) is applied to target (index 1) -> frag_1
        assert pytest.approx(ion_0.mass) == 71.03718 + 1.0
        assert pytest.approx(ion_1.mass) == 57.02146 - 1.0

    def test_branched_structure_fragmentation(self):
        """Verify that a branched tree-like structure generates expected single-cut fragments."""
        monomers = {
            i: Monomer(
                norine_name="ALA",
                residue="ALA",
                is_methylated=False,
                chirality=Chirality.L,
                mass=71.03718,
            )
            for i in range(4)
        }
        bonds = (
            Bond(source_idx=0, target_idx=1, bond_type="PEPTIDE"),
            Bond(source_idx=0, target_idx=2, bond_type="PEPTIDE"),
            Bond(source_idx=0, target_idx=3, bond_type="PEPTIDE"),
        )
        mol = Molecule(
            structure_id="branched",
            monomers=monomers,
            bonds=bonds,
        )
        fragments = generate_fragments(mol, rules=TEST_RULES)
        monomer_sets = {frozenset(f.monomer_indices) for f in fragments}

        # Single cuts of each of the 3 bridges:
        # Cut (0, 1) -> {1}, {0, 2, 3}
        # Cut (0, 2) -> {2}, {0, 1, 3}
        # Cut (0, 3) -> {3}, {0, 1, 2}
        assert frozenset({1}) in monomer_sets
        assert frozenset({0, 2, 3}) in monomer_sets
        assert frozenset({2}) in monomer_sets
        assert frozenset({0, 1, 3}) in monomer_sets
        assert frozenset({3}) in monomer_sets
        assert frozenset({0, 1, 2}) in monomer_sets
        assert len(monomer_sets) == 6

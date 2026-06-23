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


def get_ion_mass(ion, masses_vector) -> float:
    """Helper to calculate mass of a TheoreticalIon dynamically in tests."""
    total = 0.0
    mask = ion.mask
    for val in masses_vector:
        if not mask:
            break
        if mask & 1:
            total += val
        mask >>= 1
    return total + ion.total_shift_mass


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
        """Fragment should be creatable with mask and bonds."""
        frag = Fragment(
            fragment_id="frag",
            mask=7,  # nodes 0, 1, 2
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
            mask=3,  # nodes 0, 1
            removed_bonds=(),
        )
        assert frag.fragment_id == "custom_id"

    def test_fragment_empty_fails(self):
        """Fragment with no monomers should raise ValueError."""
        with pytest.raises(ValueError, match="must contain at least one monomer"):
            Fragment(fragment_id="frag", mask=0, removed_bonds=())


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

    def test_reversed_edge_orientation_still_detects_bridges(self, linear_trimer_reversed_edges):
        """Bridge detection should be direction-agnostic."""
        graph = FragmentationGraph(linear_trimer_reversed_edges, rules=TEST_RULES)
        bridges = graph.find_all_bridges()
        assert len(bridges) == 2


# ============================================================================
# Tests: Mass Calculation Lookups and Errors
# ============================================================================


class TestMassCalculationLookups:
    """Test mass lookup validation and errors."""

    def test_mass_index_out_of_bounds(self, simple_monomer):
        """Invalid index should raise IndexError."""
        frag = Fragment(fragment_id="frag", mask=(1 << 10), removed_bonds=())
        with pytest.raises(IndexError, match="out of bounds"):
            calculate_theoretical_ion(
                frag,
                masses_vector=[simple_monomer.mass],
                rules=TEST_RULES,
            )

    def test_unknown_residue_fails(self):
        """Monomer with None mass should raise ValueError."""
        m_unknown = Monomer(
            norine_name="UNKNOWN",
            residue="XYZ",
            is_methylated=False,
            chirality=Chirality.L,
            mass=None,
        )
        mol = Molecule(
            structure_id="unknown_mass",
            monomers={0: m_unknown},
        )
        with pytest.raises(ValueError, match="Cannot determine mass"):
            generate_all_theoretical_masses(mol, rules=TEST_RULES)

    def test_atomic_mass_calculation_with_parsed_record(self):
        """Monomer mass precalculated from atomic composition should be correct."""
        expected_mass = (
            2 * 12.0  # 2 carbons
            + 3 * 1.007825  # 3 hydrogens
            + 14.003074  # 1 nitrogen
            + 2 * 15.994915  # 2 oxygens
        )
        frag = Fragment(fragment_id="frag", mask=(1 << 0), removed_bonds=())
        ion = calculate_theoretical_ion(
            frag,
            masses_vector=[expected_mass],
            rules=TEST_RULES,
        )
        assert pytest.approx(get_ion_mass(ion, [expected_mass]), rel=1e-2) == expected_mass


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
        assert fragments[0].mask == (1 << 0)
        assert fragments[0].fragment_id == "frag_0"

    def test_linear_dimer_fragments(self, linear_dimer):
        """Linear dimer should generate fragments from bridge cut."""
        fragments = generate_fragments(linear_dimer, rules=TEST_RULES)
        assert len(fragments) >= 1
        masks = [f.mask for f in fragments]
        assert (1 << 0) in masks
        assert (1 << 1) in masks

    def test_linear_trimer_fragments(self, linear_trimer):
        """Linear trimer should generate multiple fragments from bridge cuts."""
        fragments = generate_fragments(linear_trimer, rules=TEST_RULES)
        assert len(fragments) >= 3
        masks = [f.mask for f in fragments]
        assert (1 << 0) in masks
        assert (1 << 2) in masks
        assert ((1 << 0) | (1 << 1)) in masks or ((1 << 1) | (1 << 2)) in masks

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
        masks = [f.mask for f in fragments]
        assert (1 << 0) in masks
        assert (1 << 2) in masks

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
        frag = Fragment(fragment_id="frag", mask=(1 << 0), removed_bonds=())
        ion = calculate_theoretical_ion(
            frag,
            masses_vector=[simple_monomer.mass],
            rules=TEST_RULES,
        )
        assert pytest.approx(get_ion_mass(ion, [simple_monomer.mass]), rel=1e-3) == 71.0372
        assert ion.rules_shifts == ()

    def test_rules_shifts_population(self, linear_dimer):
        """Verify that rules_shifts tuple contains RuleShift instances mapped to their shifts."""
        fragments = generate_fragments(linear_dimer, rules=TEST_RULES)
        frag_ala = next(f for f in fragments if f.mask == (1 << 0))
        masses_vector = [linear_dimer.monomers[node].mass for node in linear_dimer.monomers.keys()]
        ion_ala = calculate_theoretical_ion(
            frag_ala,
            masses_vector=masses_vector,
            rules=TEST_RULES,
        )

        assert len(ion_ala.rules_shifts) == 1
        assert ion_ala.rules_shifts[0].rule_name == "PEPTIDE"
        assert pytest.approx(ion_ala.rules_shifts[0].shift) == H_MASS


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

    def test_full_pipeline_cyclic_trimer(self, cyclic_trimer):
        """Generate fragments and masses for cyclic structure."""
        masses = generate_all_theoretical_masses(cyclic_trimer, rules=TEST_RULES)
        assert (
            len(masses) == 6
        )  # 3 fragments of size 1, 3 of size 2. Total 6 unique fragments from 2-cuts.

    def test_all_masses_positive(self, linear_trimer):
        """All theoretical masses should be positive."""
        masses = generate_all_theoretical_masses(linear_trimer, rules=TEST_RULES)
        masses_vector = [
            linear_trimer.monomers[node].mass for node in linear_trimer.monomers.keys()
        ]
        mass_values = [get_ion_mass(m, masses_vector) for m in masses]
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
        masks = {f.mask for f in fragments}
        assert (1 << 0) in masks
        assert ((1 << 1) | (1 << 2)) in masks
        assert (1 << 2) not in masks

    def test_rule_based_cleavage_shifts(self, linear_dimer):
        """Verify that rule-based cleavage shifts are correctly applied to fragment masses."""
        fragments = generate_fragments(linear_dimer, rules=TEST_RULES)

        frag_ala = next(f for f in fragments if f.mask == (1 << 0))
        frag_gly = next(f for f in fragments if f.mask == (1 << 1))

        masses_vector = [linear_dimer.monomers[node].mass for node in linear_dimer.monomers.keys()]

        ion_ala = calculate_theoretical_ion(
            frag_ala,
            masses_vector=masses_vector,
            rules=TEST_RULES,
        )
        ion_gly = calculate_theoretical_ion(
            frag_gly,
            masses_vector=masses_vector,
            rules=TEST_RULES,
        )

        expected_ala = linear_dimer.monomers[0].mass + H_MASS
        expected_gly = linear_dimer.monomers[1].mass - H_MASS

        assert pytest.approx(get_ion_mass(ion_ala, masses_vector)) == expected_ala
        assert pytest.approx(get_ion_mass(ion_gly, masses_vector)) == expected_gly

    def test_generate_theoretical_spectrum(self, linear_dimer):
        """Verify that theoretical spectrum generation works as expected."""
        theo_spec = generate_theoretical_spectrum(linear_dimer, rules=TEST_RULES)
        assert isinstance(theo_spec, TheoreticalSpectrum)
        assert theo_spec.structure_id == "linear_dimer"
        assert len(theo_spec.ions) == 2
        assert pytest.approx(theo_spec.structure_mass) == 128.05864
        assert theo_spec.monomer_count == 2
        assert pytest.approx(theo_spec.monomer_mass_vector) == (71.03718, 57.02146)

        sorted_ions = sorted(
            theo_spec.ions,
            key=lambda ion: get_ion_mass(ion, theo_spec.monomer_mass_vector),
        )
        assert len(sorted_ions) == 2
        # Check that masks are correct (one is 1 << 0, one is 1 << 1)
        masks = {ion.mask for ion in theo_spec.ions}
        assert masks == {1, 2}

    def test_theoretical_ion_empty_id(self):
        """TheoreticalIon should raise ValueError if fragment_id is empty."""
        with pytest.raises(ValueError, match="Fragment ID cannot be empty"):
            TheoreticalIon(fragment_id="", mask=1)

    def test_theoretical_ion_mask_type(self):
        """TheoreticalIon should raise TypeError if mask is not an integer."""
        with pytest.raises(TypeError, match="Mask must be an integer"):
            TheoreticalIon(fragment_id="frag_0", mask="1")  # type: ignore

    def test_theoretical_ion_mask_value(self):
        """TheoreticalIon should raise ValueError if mask is non-positive."""
        with pytest.raises(ValueError, match="Mask must be positive"):
            TheoreticalIon(fragment_id="frag_0", mask=0)

    def test_theoretical_ion_hashability(self):
        """TheoreticalIon should support hashing and membership checks."""
        ion = TheoreticalIon(
            fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 1.0),)
        )
        assert ion in {ion}

    def test_theoretical_spectrum_empty_id(self):
        """TheoreticalSpectrum should raise ValueError if structure_id is empty."""
        ion = TheoreticalIon(fragment_id="frag_0", mask=1)
        with pytest.raises(ValueError, match="Structure ID cannot be empty"):
            TheoreticalSpectrum(
                structure_id="",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=1,
                monomer_mass_vector=(100.0,),
            )

    def test_theoretical_spectrum_empty_ions(self):
        """TheoreticalSpectrum should raise ValueError if ions set is empty."""
        with pytest.raises(ValueError, match="must contain at least one ion"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset(),
                structure_mass=100.0,
                monomer_count=1,
                monomer_mass_vector=(100.0,),
            )

    def test_theoretical_spectrum_negative_mass(self):
        """TheoreticalSpectrum should raise ValueError if structure_mass is negative."""
        ion = TheoreticalIon(fragment_id="frag_0", mask=1)
        with pytest.raises(ValueError, match="Structure mass must be non-negative"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=-1.0,
                monomer_count=1,
                monomer_mass_vector=(100.0,),
            )

    def test_theoretical_spectrum_invalid_monomer_count(self):
        """TheoreticalSpectrum should raise ValueError if monomer_count is non-positive."""
        ion = TheoreticalIon(fragment_id="frag_0", mask=1)
        with pytest.raises(ValueError, match="Monomer count must be positive"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=0,
                monomer_mass_vector=(),
            )

    def test_theoretical_spectrum_vector_type(self):
        """TheoreticalSpectrum should raise TypeError if monomer_mass_vector is not a tuple."""
        ion = TheoreticalIon(fragment_id="frag_0", mask=1)
        with pytest.raises(TypeError, match="Monomer mass vector must be a tuple"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=1,
                monomer_mass_vector=[100.0],  # type: ignore
            )

    def test_theoretical_spectrum_vector_values(self):
        """TheoreticalSpectrum should raise ValueError if vector has negative values."""
        ion = TheoreticalIon(fragment_id="frag_0", mask=1)
        with pytest.raises(ValueError, match="All monomer masses.*must be non-negative"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=1,
                monomer_mass_vector=(-5.0,),
            )

    def test_theoretical_spectrum_vector_length_mismatch(self):
        """TheoreticalSpectrum should raise ValueError if vector length mismatches count."""
        ion = TheoreticalIon(fragment_id="frag_0", mask=1)
        with pytest.raises(ValueError, match="length must match monomer count"):
            TheoreticalSpectrum(
                structure_id="spec_1",
                ions=frozenset({ion}),
                structure_mass=100.0,
                monomer_count=2,
                monomer_mass_vector=(100.0,),
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
        masks = {f.mask for f in fragments}
        # bond1 should be cleaved since its subtype "amide_1" matches the rule.
        # bond2 should NOT be cleaved.
        assert (1 << 0) in masks
        assert ((1 << 1) | (1 << 2)) in masks
        assert (1 << 2) not in masks

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

        frag_0 = next(f for f in fragments if f.mask == (1 << 0))
        frag_1 = next(f for f in fragments if f.mask == (1 << 1))

        masses_vector = [71.03718, 57.02146]

        ion_0 = calculate_theoretical_ion(
            frag_0,
            masses_vector=masses_vector,
            rules=rules,
        )
        ion_1 = calculate_theoretical_ion(
            frag_1,
            masses_vector=masses_vector,
            rules=rules,
        )

        # target of bond (index 1) is in frag_1 -> receives right_shift (-1.0)
        # source of bond (index 0) is in frag_0 -> receives left_shift (+1.0)
        assert pytest.approx(get_ion_mass(ion_0, masses_vector)) == 71.03718 + 1.0
        assert pytest.approx(get_ion_mass(ion_1, masses_vector)) == 57.02146 - 1.0

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

        frag_0 = next(f for f in fragments if f.mask == (1 << 0))
        frag_1 = next(f for f in fragments if f.mask == (1 << 1))

        masses_vector = [71.03718, 57.02146]

        ion_0 = calculate_theoretical_ion(
            frag_0,
            masses_vector=masses_vector,
            rules=rules,
        )
        ion_1 = calculate_theoretical_ion(
            frag_1,
            masses_vector=masses_vector,
            rules=rules,
        )

        # Since reversed is True:
        # left_shift (+1.0) is applied to target (index 1) -> frag_1
        # right_shift (-1.0) is applied to source (index 0) -> frag_0
        assert pytest.approx(get_ion_mass(ion_0, masses_vector)) == 71.03718 - 1.0
        assert pytest.approx(get_ion_mass(ion_1, masses_vector)) == 57.02146 + 1.0

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
        masks = {f.mask for f in fragments}

        # Single cuts of each of the 3 bridges:
        # Cut (0, 1) -> {1}, {0, 2, 3}
        # Cut (0, 2) -> {2}, {0, 1, 3}
        # Cut (0, 3) -> {3}, {0, 1, 2}
        assert (1 << 1) in masks
        assert ((1 << 0) | (1 << 2) | (1 << 3)) in masks
        assert (1 << 2) in masks
        assert ((1 << 0) | (1 << 1) | (1 << 3)) in masks
        assert (1 << 3) in masks
        assert ((1 << 0) | (1 << 1) | (1 << 2)) in masks
        assert len(masks) == 6

"""
Tests for the ``doped.complexes`` module and associated complex
generation/analysis functionality.
"""

import os
import random
import shutil
import unittest
import warnings
from collections import Counter

import numpy as np
import pytest
from monty.serialization import loadfn
from pymatgen.analysis.molecule_matcher import KabschMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from test_generation import _check_defect_entry
from test_utils import EXAMPLE_DIR

from doped.complexes import (
    are_equivalent_molecules,
    classify_vacancy_geometry,
    get_all_equiv_complexes,
    get_complex_defect_multiplicity,
    get_split_vacancies,
    get_split_vacancies_by_geometry,
    is_periodic_image,
)
from doped.core import (
    DefectComplex,
    Interstitial,
    Vacancy,
    _get_oxi_state_modes,
    defect_structure_from_sites,
)
from doped.generation import DefectsGenerator
from doped.utils.parsing import get_matching_site
from doped.utils.supercells import _get_unwrapped_complex_frac_coords
from doped.utils.symmetry import (
    get_all_equiv_sites,
    get_primitive_structure,
    get_sga,
    get_spglib_conv_structure,
    group_order_from_schoenflies,
    point_symmetry_from_site,
    point_symmetry_from_structure,
    swap_axes,
)

data_dir = os.path.join(os.path.dirname(__file__), "data")


def if_present_rm(path):
    """
    Remove the file/folder if it exists.
    """
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


class ComplexDefectGenerationTest(unittest.TestCase):
    def setUp(self):
        self.R3c_Ga2O3 = Structure.from_file(os.path.join(data_dir, "Ga2O3_R3c_POSCAR"))
        self.Ga2O3_defect_gen = DefectsGenerator(self.R3c_Ga2O3, extrinsic="Si")
        self.Ga2O3_candidate_split_vacs = get_split_vacancies(
            self.Ga2O3_defect_gen,
            [
                "Ga",
            ],
        )
        # self.Ga2O3_C3i_split_vac_dict = next(iter(  # TODO
        #     [vac_dict for vac_dict in self.Ga2O3_candidate_split_vacs.values() if
        #      vac_dict["VIV_dist_key"] == '4.63_3.60' and vac_dict["VIV_bond_angle_degrees"] == 73.627]))

    def test_Ga2O3_periodicity_breaking_supercell_multiplicities(self):
        """
        Test that the correct multiplicities of interstitials, vacancies and
        substitutions are returned, despite the periodicity-breaking supercell
        here for R-3c Ga2O3.
        """
        # incorrect symmetrized structure with periodicity-breaking cell:
        bulk_supercell_symm_struct = get_sga(
            self.Ga2O3_defect_gen.bulk_supercell
        ).get_symmetrized_structure()
        assert len(bulk_supercell_symm_struct.equivalent_sites) == 5
        # correct symmetrized structure with primitive cell:
        assert (
            len(
                get_sga(self.Ga2O3_defect_gen.primitive_structure)
                .get_symmetrized_structure()
                .equivalent_sites
            )
            == 2
        )

        # this tests that defect multiplicities are as expected
        conventional_structure, _conv_sga = get_spglib_conv_structure(
            get_sga(self.Ga2O3_defect_gen.primitive_structure.copy().remove_oxidation_states())
        )
        reoriented_conv_structure = swap_axes(
            conventional_structure, self.Ga2O3_defect_gen._BilbaoCS_conv_cell_vector_mapping
        )
        for name, defect_entry in self.Ga2O3_defect_gen.defect_entries.items():
            _check_defect_entry(defect_entry, name, self.Ga2O3_defect_gen, reoriented_conv_structure)

        # check that the wrong result is obtained with the standard pymatgen-analysis-defects approach
        # (using the symmetry of the (periodicity-breaking) supercell), while ``doped`` gives the expected
        # number of equivalent sites in the supercell (primitive cell multiplicity x number of cells):
        bulk_supercell = self.Ga2O3_defect_gen.bulk_supercell
        supercell_over_prim_factor = len(bulk_supercell) / len(self.Ga2O3_defect_gen.primitive_structure)
        for defect_entry_name, multiplicity in [("v_Ga_0", 4), ("Ga_O_0", 6)]:
            defect_entry = self.Ga2O3_defect_gen[defect_entry_name]
            assert defect_entry.defect.multiplicity == multiplicity
            num_equiv_supercell_sites = multiplicity * supercell_over_prim_factor
            assert (
                len(
                    bulk_supercell_symm_struct.find_equivalent_sites(
                        get_matching_site(
                            defect_entry.defect_supercell_site, bulk_supercell, anonymous=True
                        )  # pymatgen-analysis-defects result, incorrect due to periodicity-breaking
                    )
                )
                != num_equiv_supercell_sites
            )
            assert len(defect_entry.equivalent_supercell_sites) == num_equiv_supercell_sites

        for defect_entry_name, multiplicity in [("Ga_i_C1_0", 12), ("Ga_i_C3i_0", 2), ("Ga_i_D3_0", 2)]:
            defect_entry = self.Ga2O3_defect_gen[defect_entry_name]
            assert defect_entry.defect.multiplicity == multiplicity
            assert (
                len(get_all_equiv_sites(defect_entry.defect_supercell_site.frac_coords, bulk_supercell))
                == multiplicity * supercell_over_prim_factor
            )

    def test_Ga2O3_candidate_split_vac_generation(self):
        """
        Test the generation of candidate split vacancies in Ga2O3.
        """
        # candidate (low electrostatic energy) split vacancies, from electrostatic analysis:
        assert list(self.Ga2O3_candidate_split_vacs) == ["Ga"]
        split_vacs_info = self.Ga2O3_candidate_split_vacs["Ga"]
        assert len(split_vacs_info["split_vacancies_electrostatic_energy_dict"]) == 4
        assert split_vacs_info["num_split_vacancies_within_tol"] == 4
        assert split_vacs_info["num_lower_energy_split_vacancies"] == 2
        for key, energy in [
            ("min_split_vac_electrostatic_formation_energy", 67.63),
            ("min_point_vacancy_electrostatic_formation_energy", 89.12),
            ("min_isolated_viv_electrostatic_formation_energy", 156.44),
            ("min_interstitial_electrostatic_formation_energy", -21.80),
        ]:
            assert np.isclose(split_vacs_info[key], energy, atol=0.02)

        # all (inequivalent) candidate split vacancies, from geometry:
        bulk_supercell = self.Ga2O3_defect_gen.bulk_supercell
        candidate_split_vacs = get_split_vacancies_by_geometry(
            bulk_supercell,
            {
                defect_entry.defect_supercell_site
                for defect_entry in self.Ga2O3_defect_gen.defect_entries.values()
                if defect_entry.name.startswith("Ga_i")
            },
        )
        assert len(candidate_split_vacs) == 303
        # number of candidates for each interstitial site; for the general position (C1) site, all 253
        # (= 23C2) pairs of the 23 Ga sites within 5 Å of it are inequivalent:
        assert Counter(
            point_symmetry_from_site(vac_dict["interstitial_site"], bulk_supercell)
            for vac_dict in candidate_split_vacs.values()
        ) == {"C1": 253, "C3i": 22, "D3": 28}
        unique_VIV_distances = {  # unique (unordered) pairs of vacancy-interstitial distances
            tuple(sorted(vac_dict["VIV_distances"].split("_")))
            for vac_dict in candidate_split_vacs.values()
        }
        assert len(unique_VIV_distances) == 232

        unique_VIV_angle_split_vacs = {
            frozenset(
                (
                    round(vac_dict["vacancy_1_interstitial_distance"], 2),
                    round(vac_dict["vacancy_2_interstitial_distance"], 2),
                    round(vac_dict["VIV_bond_angle_degrees"], 1),
                )
            ): vac_dict
            for vac_dict in candidate_split_vacs.values()
        }
        # this is the current de-duplication strategy so should be same length:
        assert len(unique_VIV_angle_split_vacs) == len(candidate_split_vacs)

        # check they are all classified as split vacancies as expected:
        for split_vac_dict in candidate_split_vacs.values():
            structure = defect_structure_from_sites(
                bulk_supercell,
                vacancy_sites=[split_vac_dict["vacancy_1_site"], split_vac_dict["vacancy_2_site"]],
                interstitial_sites=split_vac_dict["interstitial_site"],
            )
            assert classify_vacancy_geometry(structure, bulk_supercell) == "Split Vacancy"

        # and point vacancies as simple vacancies:
        v_Ga_supercell = self.Ga2O3_defect_gen["v_Ga_0"].defect_supercell
        assert classify_vacancy_geometry(v_Ga_supercell, bulk_supercell) == "Simple Vacancy"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # not-a-vacancy warning
            with pytest.raises(ValueError, match="No species is missing from the defect supercell"):
                classify_vacancy_geometry(bulk_supercell, v_Ga_supercell)

    # def test_Ga2O3_defect_structure_from_sites(self):
    #     """
    #     Test the ``defect_structure_from_sites`` function, using R-3c
    #     Ga2O3 as an example case.
    #     """
    #     # TODO
    #     self.Ga2O3_defect_gen

    def test_complex_defect_same_site_warning(self):
        with warnings.catch_warnings(record=True) as w, pytest.raises(ValueError):
            defect_structure_from_sites(
                self.Ga2O3_defect_gen.bulk_supercell,
                vacancy_sites=[self.Ga2O3_defect_gen["v_Ga_-3"].defect_supercell_site],
                substitution_sites=[self.Ga2O3_defect_gen["Si_Ga_+1"].defect_supercell_site],
            )

        assert any(
            "Some input defect sites are less than 0.1" in str(warning.message)  # no Å codepoint issues
            for warning in w
        )

        with warnings.catch_warnings(record=True) as w:
            defect_structure_from_sites(
                self.Ga2O3_defect_gen.bulk_supercell,
                vacancy_sites=[self.Ga2O3_defect_gen["v_Ga_-3"].defect_supercell_site],
                # using one of the equivalent sites here, to avoid placing V_Ga and Si_Ga at same site:
                substitution_sites=[self.Ga2O3_defect_gen["Si_Ga_+1"].equivalent_supercell_sites[0]],
            )

        assert not any(
            "Some input defect sites are less than 0.1" in str(warning.message)  # no Å codepoint issues
            for warning in w
        )


class DefectComplexSerialisationTest(unittest.TestCase):
    def test_generated_complexes_json_round_trip(self):
        """
        Test that generated ``DefectComplex`` objects are unchanged by
        (de)serialisation (e.g. unwrapped constituent sites not mapped back to
        the unit cell upon reloading), such that complex generation with a
        reloaded ``DefectsGenerator`` (including one saved with the since-
        removed ``padding`` charge state option) does not fail or duplicate
        previously-generated complexes.
        """
        CdTe_prim = Structure.from_file(os.path.join(EXAMPLE_DIR, "CdTe/relaxed_primitive_POSCAR"))
        defect_gen = DefectsGenerator(CdTe_prim)
        defect_gen.generate_complex_chains(["v_Cd", "Cd_i"], chain_length=3, max_separation=3.0)
        defect_gen.generate_complex_chains(["v_Cd", "v_Te"], chain_length=2, max_separation=3.0)
        complex_entries = {
            name: defect_entry
            for name, defect_entry in defect_gen.defect_entries.items()
            if isinstance(defect_entry.defect, DefectComplex)
        }
        assert len({name.rsplit("_", 1)[0] for name in complex_entries}) == 31  # without charge states

        defect_gen.to_json("test_CdTe_complexes_defect_gen.json.gz")
        try:
            reloaded_defect_gen = DefectsGenerator.from_json("test_CdTe_complexes_defect_gen.json.gz")
        finally:
            if_present_rm("test_CdTe_complexes_defect_gen.json.gz")

        for name, defect_entry in complex_entries.items():
            defect_complex = defect_entry.defect
            reloaded_defect_complex = reloaded_defect_gen.defect_entries[name].defect
            assert reloaded_defect_complex == defect_complex
            for defect, reloaded_defect in zip(
                defect_complex.defects, reloaded_defect_complex.defects, strict=True
            ):
                assert not reloaded_defect.map_to_unit_cell
                assert np.allclose(defect.site.frac_coords, reloaded_defect.site.frac_coords)  # unwrapped

        # re-generating complexes with the reloaded generator does not add duplicates:
        reloaded_defect_gen.generate_complex_chains(["v_Cd", "v_Te"], chain_length=2, max_separation=3.0)
        assert set(reloaded_defect_gen.defect_entries) == set(defect_gen.defect_entries)

        # generators saved with the since-removed ``padding`` charge state generation option:
        defect_gen_dict = defect_gen.as_dict()
        defect_gen_dict["charge_state_gen_kwargs"] = {"padding": 2}
        old_defect_gen = DefectsGenerator.from_dict(defect_gen_dict)
        assert old_defect_gen.charge_state_gen_kwargs == {}
        old_defect_gen.generate_complex_chains(["v_Te", "Te_i"], chain_length=2, max_separation=3.0)
        assert len(old_defect_gen.defect_entries) > len(defect_gen.defect_entries)


class MoleculeMatcherTest(unittest.TestCase):
    def setUp(self):
        self.mol = Molecule(["X", "Ga", "X"], [[0.1, 0.1, 0.1], [0.1, 0.4, 0.1], [0.2, 0.2, 0.2]])
        self.primitive_structure = Structure.from_file(os.path.join(data_dir, "Ga2O3_R3c_POSCAR"))

    def test_translation_invariance(self):
        """
        Test molecule matching with translation invariance.

        Standard KabschMatcher works fine here.
        """
        translated_mol = Molecule(
            self.mol.species,
            _get_molecule_coords(self.mol) + np.array([0.15, -0.5, 10]),
        )
        assert are_equivalent_molecules(self.mol, translated_mol)
        assert np.isclose(KabschMatcher(self.mol).fit(translated_mol)[-1], 0)

    def test_rotation_invariance(self):
        """
        Test molecule matching with rotation invariance.

        Standard KabschMatcher works fine here.
        """
        rotated_mol = Molecule(
            self.mol.species, [rotate_45_degrees_x(coords) for coords in _get_molecule_coords(self.mol)]
        )
        assert are_equivalent_molecules(self.mol, rotated_mol)
        assert np.isclose(KabschMatcher(self.mol).fit(rotated_mol)[-1], 0)

    def test_permutation_invariance(self):
        """
        Test molecule matching with permutation invariance.

        Standard ``KabschMatcher`` does not work here.
        """
        permuted_mol = Molecule(
            [self.mol.species[i] for i in [1, 2, 0]], [self.mol.sites[i].coords for i in [1, 2, 0]]
        )
        assert are_equivalent_molecules(self.mol, permuted_mol)
        with pytest.raises(ValueError) as e:
            KabschMatcher(self.mol).fit(permuted_mol)
        assert "The order of the species aren't matching!" in str(e.value)

        # check it doesn't match if the species are mismatching:
        permuted_mol = Molecule(self.mol.species, [self.mol.sites[i].coords for i in [1, 2, 0]])
        assert not are_equivalent_molecules(self.mol, permuted_mol)

    def test_is_periodic_image(self):
        """
        Test the ``is_periodic_image`` function; whether two complexes are
        equal up to a `rigid` lattice vector translation, in any site order.
        """
        lattice = self.primitive_structure.lattice
        sites = [
            PeriodicSite(site.species, site.coords, lattice, coords_are_cartesian=True)
            for site in self.mol.sites
        ]

        def _shifted_sites(sites, frac_shifts):
            return [
                PeriodicSite(site.species, site.frac_coords + np.array(frac_shift), lattice)
                for site, frac_shift in zip(sites, frac_shifts, strict=True)
            ]

        # all sites shifted by the _same_ lattice vector:
        periodically_shifted_sites = _shifted_sites(sites, [[1, 0, -1]] * 3)
        assert is_periodic_image(sites, periodically_shifted_sites)

        # sites shifted by _different_ lattice vectors (so not a rigid translation of the complex):
        assert not is_periodic_image(sites, _shifted_sites(sites, [[4, -2, 0], [1, 0, -1], [1, 0, -1]]))

        # permutation invariance:
        assert is_periodic_image(sites, [periodically_shifted_sites[i] for i in [1, 2, 0]])

        # each site must be matched with a different site of the other complex, so not a periodic image if
        # a site is repeated (here the first site twice, in place of the second):
        assert not is_periodic_image(sites, [periodically_shifted_sites[i] for i in [0, 0, 2]])
        # nor if the species differ (here all Ga), or if a site is displaced (here by 0.1 in frac coords):
        assert not is_periodic_image(
            sites, [PeriodicSite("Ga", site.frac_coords, lattice) for site in sites]
        )
        assert not is_periodic_image(sites, _shifted_sites(sites, [[0.1, 0, 0], [0, 0, 0], [0, 0, 0]]))

        # complex with two same-species sites which are lattice translations of each other (e.g. the two
        # vacancies of a split vacancy, on the same primitive cell site in neighbouring cells), compared
        # to the same complex translated by a lattice vector, with these two sites listed in the other
        # order. Each of these sites is then a periodic image of `both` such sites in the other complex,
        # so matching sites individually by their closest periodic images is ambiguous, and can pair them
        # such that no single lattice vector relates all pairs (as in a previous implementation, which
        # then double-counted such complexes). They are periodic images, as one lattice vector maps all
        # sites of one complex onto those of the other...
        vac_1, vac_2, interstitial = (
            PeriodicSite("Ga", frac_coords, lattice)
            for frac_coords in ([0.2, 0.3, 0.4], [1.2, 0.3, 0.4], [0.75, 0.3, 0.6])
        )
        assert is_periodic_image(
            [vac_1, vac_2, interstitial], _shifted_sites([vac_2, vac_1, interstitial], [[-1, 1, 0]] * 3)
        )
        assert not is_periodic_image(  # ...but not if the interstitial is translated by a different one
            [vac_1, vac_2, interstitial],
            _shifted_sites([vac_2, vac_1, interstitial], [[-1, 1, 0], [-1, 1, 0], [0, 1, 0]]),
        )


def _get_molecule_coords(mol):
    return np.array([site.coords for site in mol.sites])


def rotate_45_degrees_x(coords):
    """
    Rotate the input coords by 45 degrees around the x-axis.
    """
    theta = np.pi / 4  # 45 degrees in radians
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix_x = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])

    return np.dot(np.array(coords), rotation_matrix_x.T)


class SymmetryMultiplicityTest(unittest.TestCase):
    def setUp(self):
        self.R3c_Ga2O3_split_vac_info_dict = loadfn(
            os.path.join(data_dir, "Split_Vacancies/Ga2O3-mp-1243.json.gz")
        )
        self.C2m_Ga2O3_split_vac_info_dict = loadfn(
            os.path.join(data_dir, "Split_Vacancies/Ga2O3-mp-886.json.gz")
        )
        self.energy_key = "small_f32_GOQN"
        # reference symmetries & multiplicities (of the complexes in the host crystal; i.e. independent of
        # any periodicity breaking in the supercell), generated with ``doped`` and cross-checked in
        # ``_check_complex_multiplicity_and_point_group`` below. ``relaxed_symm`` is ``None`` for relaxed
        # structures where the determined symmetry is sensitive to ``symprec`` (differs for 0.08 - 0.12):
        self.reference_values = loadfn(
            os.path.join(data_dir, "Split_Vacancies/doped_reference_values.json")
        )

    def _check_complex_multiplicity_and_point_group(
        self, info_dict, subdict, reference, complex_symm, num_symm_ops
    ):
        """
        Check the split vacancy multiplicity against the reference value, the
        orbit-stabiliser relation (multiplicity x point group order = number of
        symmetry operations of the primitive cell), and that the complex point
        group from the orbit analysis (``get_all_equiv_complexes``) matches
        that from the local symmetry analysis of the structure
        (``complex_symm``; from ``point_symmetry_from_structure``).

        Returns the primitive cell multiplicity of the complex.
        """
        bulk_supercell = info_dict["bulk_supercell"]
        bulk_prim = get_primitive_structure(bulk_supercell)
        supercell_over_prim_factor = len(bulk_supercell) / len(bulk_prim)
        sites_kwargs = {
            "vacancy_sites": [subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
            "interstitial_sites": [subdict["interstitial_site"]],
        }
        comp_mult = get_complex_defect_multiplicity(bulk_supercell=bulk_supercell, **sites_kwargs)
        assert comp_mult == reference["multiplicity"]
        assert comp_mult == get_complex_defect_multiplicity(  # same answer with primitive structure input
            bulk_supercell=bulk_supercell, primitive_structure=bulk_prim, **sites_kwargs
        )
        supercell_comp_mult = get_complex_defect_multiplicity(
            bulk_supercell=bulk_supercell,
            primitive_structure=bulk_prim,
            primitive_cell_multiplicity=False,
            **sites_kwargs,
        )
        assert supercell_comp_mult == reference["multiplicity"] * supercell_over_prim_factor

        sites = [*sites_kwargs["vacancy_sites"], *sites_kwargs["interstitial_sites"]]
        equiv_complexes, point_group = get_all_equiv_complexes(
            [
                PeriodicSite(site.species, frac_coords, bulk_supercell.lattice)
                for site, frac_coords in zip(
                    sites, _get_unwrapped_complex_frac_coords(bulk_supercell, sites), strict=True
                )
            ],
            bulk_supercell,
            return_point_group=True,
        )
        assert len(equiv_complexes) == supercell_comp_mult
        assert point_group == complex_symm == reference["unrelaxed_symm"]
        assert comp_mult * group_order_from_schoenflies(point_group) == num_symm_ops

        return comp_mult

    def test_symmetry_multiplicity_R3c_Ga2O3(self):
        """
        Test symmetry and multiplicity analysis for split vacancies in R-3c
        Ga2O3.

        For R-3c Ga2O3, we have periodicity breaking in the generated supercell
        which breaks the rotational (but not inversion) symmetries of the
        `supercell` (C3i -> Ci, C2 -> C1). The point symmetries determined here
        are those of the defects in the host crystal (e.g. C3i, C2, D3), and so
        are unaffected by this. Useful test case for future stenciling to avoid
        periodicity breaking.
        """
        info_dict = self.R3c_Ga2O3_split_vac_info_dict
        reference_values = self.reference_values["Ga2O3-mp-1243"]
        bulk_prim = get_primitive_structure(info_dict["bulk_supercell"])
        num_symm_ops = len(get_sga(bulk_prim).get_symmetry_operations())
        supercell_over_prim_factor = len(info_dict["bulk_supercell"]) / len(bulk_prim)
        molecule_dict = {}

        ml_calculated_split_vac_dicts = [
            (cation, energy, subdict)
            for cation, cation_dict in info_dict.items()
            if isinstance(cation_dict, dict) and "split_vacancies_energy_dict" in cation_dict
            for energy, subdict in cation_dict["split_vacancies_energy_dict"].items()
            if self.energy_key in subdict
        ]

        for cation, energy, subdict in ml_calculated_split_vac_dicts:
            reference = reference_values[f"{cation}|{energy}"]
            orig_split_vacancy = defect_structure_from_sites(
                info_dict["bulk_supercell"],
                vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                interstitial_sites=[subdict["interstitial_site"]],
            )
            orig_split_vacancy_symm = point_symmetry_from_structure(
                orig_split_vacancy,
                info_dict["bulk_supercell"],
                relaxed=True,
                verbose=False,
            )
            assert orig_split_vacancy_symm == reference["unrelaxed_symm"]

            for v_idx in [1, 2]:
                vac_symm = point_symmetry_from_site(
                    subdict[f"vac_defect_{v_idx}_site"], info_dict["bulk_supercell"]
                )
                assert vac_symm == "C3"  # only C3 Ga sites in R-3c Ga2O3
                assert vac_symm == reference["vac_symms"][v_idx - 1]

            int_symm = point_symmetry_from_site(subdict["interstitial_site"], info_dict["bulk_supercell"])
            assert int_symm in [
                "C2",
                "C3i",
                "D3",
            ]  # unrelaxed interstitial sites in R-3c Ga2O3
            assert int_symm == reference["int_symm"]
            relaxed_symm = point_symmetry_from_structure(
                Structure.from_ase_atoms(subdict[self.energy_key]["struct_atoms"]),
                info_dict["bulk_supercell"],
                verbose=False,
            )
            if reference["relaxed_symm"] is not None:  # ``None`` if sensitive to ``symprec``
                assert relaxed_symm == reference["relaxed_symm"]

            comp_mult = self._check_complex_multiplicity_and_point_group(
                info_dict, subdict, reference, orig_split_vacancy_symm, num_symm_ops
            )

            int_obj = Interstitial(info_dict["bulk_supercell"], subdict["interstitial_site"])
            v1_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_1_site"])
            v2_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_2_site"])

            assert int_obj.multiplicity == subdict["int_multiplicity"] * supercell_over_prim_factor
            assert v1_obj.multiplicity == subdict["vac_1_multiplicity"] * supercell_over_prim_factor
            assert v2_obj.multiplicity == subdict["vac_2_multiplicity"] * supercell_over_prim_factor

            # this isn't universally true for complex defects, but is for the split vacancies:
            assert (
                comp_mult
                <= (int_obj.multiplicity * v1_obj.multiplicity * v2_obj.multiplicity)
                / supercell_over_prim_factor**3
            )

            raw_complex_defect_sites = [
                subdict["vac_defect_1_site"],
                subdict["vac_defect_2_site"],
                subdict["interstitial_site"],
            ]
            molecule_dict[round(float(energy), 2)] = Molecule(
                [site.species for site in raw_complex_defect_sites],
                info_dict["bulk_supercell"].lattice.get_cartesian_coords(
                    [
                        site.frac_coords + raw_complex_defect_sites[0].distance_and_image(site)[1]
                        for site in raw_complex_defect_sites
                    ]
                ),
            )

        # in this case, two of the molecules are actually equivalent, with slightly different ES energies
        # because of the periodicity breaking in the generated supercell:
        for key, mol in molecule_dict.items():
            if key in [-4245.99, -4246.28]:
                assert are_equivalent_molecules(molecule_dict[-4245.99], mol)
            else:
                assert not are_equivalent_molecules(molecule_dict[-4245.99], mol)

    def test_symmetry_multiplicity_C2m_Ga2O3(self):
        """
        Test symmetry and multiplicity analysis for split vacancies in C2/m
        Ga2O3.

        For C2/m Ga2O3, we have no periodicity breaking. Ga site symmetries are
        all Cs, while we have Cs, C2h, C2 and C1 interstitial site symmetries.
        """
        info_dict = self.C2m_Ga2O3_split_vac_info_dict
        reference_values = self.reference_values["Ga2O3-mp-886"]
        bulk_prim = get_primitive_structure(info_dict["bulk_supercell"])
        num_symm_ops = len(get_sga(bulk_prim).get_symmetry_operations())
        supercell_over_prim_factor = len(info_dict["bulk_supercell"]) / len(bulk_prim)
        molecule_dict = {}

        ml_calculated_split_vac_dicts = [
            (cation, energy, subdict)
            for cation, cation_dict in info_dict.items()
            if isinstance(cation_dict, dict) and "split_vacancies_energy_dict" in cation_dict
            for energy, subdict in cation_dict["split_vacancies_energy_dict"].items()
            if self.energy_key in subdict
        ]

        for cation, energy, subdict in ml_calculated_split_vac_dicts:
            reference = reference_values[f"{cation}|{energy}"]
            orig_split_vacancy = defect_structure_from_sites(
                info_dict["bulk_supercell"],
                vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                interstitial_sites=[subdict["interstitial_site"]],
            )
            orig_split_vacancy_symm = point_symmetry_from_structure(
                orig_split_vacancy,
                info_dict["bulk_supercell"],
                relaxed=True,
                verbose=False,
            )
            assert orig_split_vacancy_symm == reference["unrelaxed_symm"]

            for v_idx in [1, 2]:
                vac_symm = point_symmetry_from_site(
                    subdict[f"vac_defect_{v_idx}_site"], info_dict["bulk_supercell"]
                )
                assert vac_symm == "Cs"  # only Cs Ga sites in C2/m Ga2O3
                assert vac_symm == reference["vac_symms"][v_idx - 1]

            int_symm = point_symmetry_from_site(subdict["interstitial_site"], info_dict["bulk_supercell"])
            assert int_symm in [
                "Cs",
                "C2h",
                "C2",
                "C1",
            ]  # unrelaxed interstitial sites in C2/m Ga2O3
            assert int_symm == reference["int_symm"]
            relaxed_symm = point_symmetry_from_structure(
                Structure.from_ase_atoms(subdict[self.energy_key]["struct_atoms"]),
                info_dict["bulk_supercell"],
                verbose=False,
            )
            if reference["relaxed_symm"] is not None:  # ``None`` if sensitive to ``symprec``
                assert relaxed_symm == reference["relaxed_symm"]

            comp_mult = self._check_complex_multiplicity_and_point_group(
                info_dict, subdict, reference, orig_split_vacancy_symm, num_symm_ops
            )

            int_obj = Interstitial(info_dict["bulk_supercell"], subdict["interstitial_site"])
            v1_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_1_site"])
            v2_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_2_site"])

            assert int_obj.multiplicity == subdict["int_multiplicity"] * supercell_over_prim_factor
            assert v1_obj.multiplicity == subdict["vac_1_multiplicity"] * supercell_over_prim_factor
            assert v2_obj.multiplicity == subdict["vac_2_multiplicity"] * supercell_over_prim_factor

            point_symm_mult_dict = {
                "Cs": 2,
                "C2h": 1,
                "C2": 2,
                "C1": 4,
            }
            for v_idx, v_obj in enumerate([v1_obj, v2_obj]):
                assert (
                    v_obj.multiplicity
                    == point_symm_mult_dict[reference["vac_symms"][v_idx]] * supercell_over_prim_factor
                )
            assert (
                int_obj.multiplicity
                == point_symm_mult_dict[reference["int_symm"]] * supercell_over_prim_factor
            )

            # this isn't universally true for complex defects, but is for the split vacancies:
            assert (
                comp_mult
                <= (int_obj.multiplicity * v1_obj.multiplicity * v2_obj.multiplicity)
                / supercell_over_prim_factor**3
            )

            # in C2/m Ga2O3, each split vacancy has the same multiplicity as the interstitial site, unless
            # its symmetry is lower than that of the interstitial site (-> greater multiplicity):
            assert comp_mult == (
                int_obj.multiplicity
                / supercell_over_prim_factor
                * group_order_from_schoenflies(int_symm)
                / group_order_from_schoenflies(orig_split_vacancy_symm)
            )

            raw_complex_defect_sites = [
                subdict["vac_defect_1_site"],
                subdict["vac_defect_2_site"],
                subdict["interstitial_site"],
            ]
            molecule_dict[round(float(energy), 2)] = Molecule(
                [site.species for site in raw_complex_defect_sites],
                info_dict["bulk_supercell"].lattice.get_cartesian_coords(
                    [
                        site.frac_coords + raw_complex_defect_sites[0].distance_and_image(site)[1]
                        for site in raw_complex_defect_sites
                    ]
                ),
            )

        # Here the `relaxed` -3538.32 structure goes to the -3540.98 structure, so can use this in future
        # tests of relaxed complex defect site determinations! (TODO)
        for key, mol in molecule_dict.items():
            for other_key, other_mol in molecule_dict.items():
                if key != other_key:
                    assert not are_equivalent_molecules(mol, other_mol)

    def test_symmetry_multiplicity_tricky_structures(self):
        """
        Test symmetry and multiplicity analysis for split vacancies in some
        complex structures.

        Some of these had issues with initial symmetry/multiplicity analysis
        prior to robustness updates. Here we test that the analysis proceeds
        fine and matches expected values.
        """
        database_folder = f"{data_dir}/Split_Vacancies"
        formula_mpid_files = [
            i for i in os.listdir(database_folder) if i.endswith(".json.gz") and "Ga2O3" not in i
        ]
        for formula_mpid_file in formula_mpid_files:
            info_dict = loadfn(f"{database_folder}/{formula_mpid_file}")
            reference_values = self.reference_values[formula_mpid_file.split(".json")[0]]
            bulk_prim = get_primitive_structure(info_dict["bulk_supercell"])
            num_symm_ops = len(get_sga(bulk_prim).get_symmetry_operations())
            supercell_over_prim_factor = len(info_dict["bulk_supercell"]) / len(bulk_prim)
            molecule_dict = {}

            ml_calculated_split_vac_dicts = [
                (cation, energy, subdict)
                for cation, cation_dict in info_dict.items()
                if isinstance(cation_dict, dict) and "split_vacancies_energy_dict" in cation_dict
                for energy, subdict in cation_dict["split_vacancies_energy_dict"].items()
                if self.energy_key in subdict
            ]
            if "GeI2" not in formula_mpid_file and "Li3Ti4O8" not in formula_mpid_file:
                # take a random subset of max 10:
                ml_calculated_split_vac_dicts = random.sample(
                    ml_calculated_split_vac_dicts, min(10, len(ml_calculated_split_vac_dicts))
                )
            for cation, energy, subdict in ml_calculated_split_vac_dicts:
                reference = reference_values[f"{cation}|{energy}"]
                orig_split_vacancy = defect_structure_from_sites(
                    info_dict["bulk_supercell"],
                    vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                    interstitial_sites=[subdict["interstitial_site"]],
                )
                orig_split_vacancy_symm = point_symmetry_from_structure(
                    orig_split_vacancy,
                    info_dict["bulk_supercell"],
                    relaxed=True,
                    verbose=False,
                )
                assert orig_split_vacancy_symm == reference["unrelaxed_symm"]

                for v_idx in [1, 2]:
                    vac_symm = point_symmetry_from_site(
                        subdict[f"vac_defect_{v_idx}_site"], info_dict["bulk_supercell"]
                    )
                    assert vac_symm == reference["vac_symms"][v_idx - 1]

                int_symm = point_symmetry_from_site(
                    subdict["interstitial_site"], info_dict["bulk_supercell"]
                )
                assert int_symm == reference["int_symm"]
                relaxed_symm = point_symmetry_from_structure(
                    Structure.from_ase_atoms(subdict[self.energy_key]["struct_atoms"]),
                    info_dict["bulk_supercell"],
                    verbose=False,
                )
                if reference["relaxed_symm"] is not None:  # ``None`` if sensitive to ``symprec``
                    assert relaxed_symm == reference["relaxed_symm"]

                comp_mult = self._check_complex_multiplicity_and_point_group(
                    info_dict, subdict, reference, orig_split_vacancy_symm, num_symm_ops
                )

                int_obj = Interstitial(info_dict["bulk_supercell"], subdict["interstitial_site"])
                v1_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_1_site"])
                v2_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_2_site"])

                assert int_obj.multiplicity == subdict["int_multiplicity"] * supercell_over_prim_factor
                assert v1_obj.multiplicity == subdict["vac_1_multiplicity"] * supercell_over_prim_factor
                assert v2_obj.multiplicity == subdict["vac_2_multiplicity"] * supercell_over_prim_factor

                for v_idx, v_obj in enumerate([v1_obj, v2_obj]):
                    assert (
                        v_obj.multiplicity
                        == subdict[f"vac_{v_idx + 1}_multiplicity"] * supercell_over_prim_factor
                    )
                assert int_obj.multiplicity == subdict["int_multiplicity"] * supercell_over_prim_factor

                # the split vacancy multiplicity is at least that of its interstitial site, as any symmetry
                # operation of the complex must also be a symmetry operation of its (only) interstitial:
                assert comp_mult >= int_obj.multiplicity / supercell_over_prim_factor

                raw_complex_defect_sites = [
                    subdict["vac_defect_1_site"],
                    subdict["vac_defect_2_site"],
                    subdict["interstitial_site"],
                ]
                molecule_dict[round(float(energy), 2)] = Molecule(
                    [site.species for site in raw_complex_defect_sites],
                    info_dict["bulk_supercell"].lattice.get_cartesian_coords(
                        [
                            site.frac_coords + raw_complex_defect_sites[0].distance_and_image(site)[1]
                            for site in raw_complex_defect_sites
                        ]
                    ),
                )

            eq_mol_count = 0
            for key, mol in molecule_dict.items():
                for other_key, other_mol in molecule_dict.items():
                    if (  # same cations, different molecules
                        sorted(mol.atomic_numbers) == sorted(other_mol.atomic_numbers) and key != other_key
                    ) and are_equivalent_molecules(mol, other_mol):
                        eq_mol_count += 1

            # (ordered) pairs of split vacancies with equivalent (V-I-V) geometries:
            if "GeI2" in formula_mpid_file:
                assert eq_mol_count == 8
            elif "Li3Ti4O8" in formula_mpid_file:
                assert eq_mol_count == 62
            elif "Na2Mg(NiO2)2" in formula_mpid_file:
                assert eq_mol_count <= 22  # 22 for the full set; random subset of 10 here
            else:
                assert eq_mol_count == 0
            # Note that a lot of these have matching equivalent relaxed complexes, so would make good test
            # cases for future functionality for this (TODO)


class GetOxiStateModesTest(unittest.TestCase):
    """
    Tests for the ``_get_oxi_state_modes`` helper, which returns the mode (most
    common, weighted by amount) oxidation state per element.
    """

    def _mixed_valence_structure(self):
        # 3x Fe3+, 1x Fe2+, 4x O2-:
        return Structure(
            Lattice.cubic(5.0),
            ["Fe3+", "Fe3+", "Fe3+", "Fe2+", "O2-", "O2-", "O2-", "O2-"],
            [[i * 0.1, i * 0.1, i * 0.1] for i in range(8)],
        )

    def test_mixed_valence_structure_returns_mode(self):
        # Fe3+ is more common than Fe2+, so the mode (not the last-encountered value) is returned:
        assert _get_oxi_state_modes(self._mixed_valence_structure()) == {"Fe": 3.0, "O": -2.0}

    def test_mixed_valence_composition_returns_mode(self):
        comp = self._mixed_valence_structure().composition
        assert _get_oxi_state_modes(comp) == {"Fe": 3.0, "O": -2.0}

    def test_single_valence_structure(self):
        struct = Structure(
            Lattice.cubic(5.0),
            ["Ga3+", "Ga3+", "O2-", "O2-", "O2-"],
            [[i * 0.1, i * 0.1, i * 0.1] for i in range(5)],
        )
        assert _get_oxi_state_modes(struct) == {"Ga": 3.0, "O": -2.0}

    def test_mode_differs_from_rounded_mean(self):
        # 2x Mn2+, 1x Mn4+: mode is +2, but the rounded mean would be
        # ``round((2 + 2 + 4) / 3) = round(2.67) = +3``, so this confirms the mode (not the rounded mean)
        # is correctly used:
        struct = Structure(
            Lattice.cubic(5.0),
            ["Mn2+", "Mn2+", "Mn4+", "O2-", "O2-", "O2-"],
            [[i * 0.1, i * 0.1, i * 0.1] for i in range(6)],
        )
        assert _get_oxi_state_modes(struct) == {"Mn": 2.0, "O": -2.0}

    def test_dict_passthrough(self):
        oxi_dict = {"Fe": 3, "O": -2}
        assert _get_oxi_state_modes(oxi_dict) == oxi_dict

    def test_undecorated_structure_defaults_to_zero(self):
        struct = Structure(
            Lattice.cubic(5.0),
            ["Ga", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        )
        assert _get_oxi_state_modes(struct) == {"Ga": 0, "O": 0}

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError, match="must be a pymatgen Structure, Composition or dict"):
            _get_oxi_state_modes("not a valid input")

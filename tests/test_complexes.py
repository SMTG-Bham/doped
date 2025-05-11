"""
Tests for the ``doped.complexes`` module and associated complex
generation/analysis functionality.
"""

import os
import shutil
import unittest

import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.molecule_matcher import KabschMatcher
from pymatgen.core.structure import Molecule, Structure
from pymatgen.util.coord import is_coord_subset_pbc
from test_generation import _check_defect_entry

from doped.complexes import (
    are_equivalent_molecules,
    classify_vacancy_geometry,
    generate_complex_from_defect_sites,
    get_complex_defect_multiplicity,
)
from doped.core import Interstitial, Vacancy
from doped.generation import DefectsGenerator
from doped.utils.parsing import get_matching_site
from doped.utils.symmetry import (
    get_all_equiv_sites,
    get_primitive_structure,
    get_sga,
    is_periodic_image,
    point_symmetry_from_site,
    point_symmetry_from_structure,
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
        self.Ga2O3_defect_gen = DefectsGenerator(self.R3c_Ga2O3, extrinsic="Se")
        self.Ga2O3_candidate_split_vacs = self.Ga2O3_defect_gen.get_candidate_split_vacancies_for_element(
            "Ga"
        )
        # self.Ga2O3_C3i_split_vac_dict = next(iter(  # TODO
        #     [vac_dict for vac_dict in self.Ga2O3_candidate_split_vacs.values() if
        #      vac_dict["VIV_dist_key"] == '4.63_3.60' and vac_dict["VIV_bond_angle_degrees"] == 73.627]))

    def test_Ga2O3_periodicity_breaking_supercell_multiplicities(self):
        """
        Test that the correct multiplicities of interstitials, vacancies and
        substitutions are returned, despite the periodicity-breaking supercell
        here for R-3c Ga2O3.

        Previously, incorrect multiplicities were returned with this
        periodicity-breaking supercell.
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
        for defect_name, defect_entry in self.Ga2O3_defect_gen.defect_entries.items():
            _check_defect_entry(defect_entry, defect_name, self.Ga2O3_defect_gen)

        # check that the wrong result is obtained with the standard pymatgen-analysis-defects approach:
        for defect_entry_name in ["v_Ga_0", "Ga_O_0"]:
            assert len(
                bulk_supercell_symm_struct.find_equivalent_sites(
                    get_matching_site(
                        self.Ga2O3_defect_gen[defect_entry_name].defect_supercell_site,
                        self.Ga2O3_defect_gen.bulk_supercell,
                        anonymous=True,
                    )
                )
            ) != (
                self.Ga2O3_defect_gen[defect_entry_name].defect.multiplicity
                * (
                    len(self.Ga2O3_defect_gen.primitive_structure)
                    / len(self.Ga2O3_defect_gen.bulk_supercell)
                )
            )
        assert len(
            get_all_equiv_sites(
                self.Ga2O3_defect_gen["Ga_i_C2_0"].defect_supercell_site.frac_coords,
                self.Ga2O3_defect_gen.bulk_supercell,
            )
        ) != (
            self.Ga2O3_defect_gen["Ga_i_C2_0"].defect.multiplicity
            * (len(self.Ga2O3_defect_gen.primitive_structure) / len(self.Ga2O3_defect_gen.bulk_supercell))
        )

    def test_Ga2O3_candidate_split_vac_generation(self):
        """
        Test the generation of candidate split vacancies in Ga2O3.
        """
        assert len(self.Ga2O3_candidate_split_vacs) == 269
        unique_VIV_split_vacs = {
            vac_dict["VIV_dist_key"]: vac_dict for vac_dict in self.Ga2O3_candidate_split_vacs.values()
        }
        assert len(unique_VIV_split_vacs) == 126

        unique_VIV_angle_split_vacs = {
            frozenset(
                (
                    round(vac_dict["vacancy_1_interstitial_distance"], 2),
                    round(vac_dict["vacancy_2_interstitial_distance"], 2),
                    round(vac_dict["VIV_bond_angle_degrees"], 1),
                )
            ): vac_dict
            for vac_dict in self.Ga2O3_candidate_split_vacs.values()
        }
        # this is the de-duplication strategy so should be same length:
        assert len(unique_VIV_angle_split_vacs) == len(self.Ga2O3_candidate_split_vacs)

        # check they are all classified as split vacancies as expected:
        for split_vac_dict in list(self.Ga2O3_candidate_split_vacs.values()):
            structure = generate_complex_from_defect_sites(
                self.Ga2O3_defect_gen.bulk_supercell,
                vacancy_sites=[split_vac_dict["vacancy_1_site"], split_vac_dict["vacancy_2_site"]],
                interstitial_sites=split_vac_dict["interstitial_site"],
            )
            split_vac_dict["structure"] = structure
            assert (
                classify_vacancy_geometry(structure, self.Ga2O3_defect_gen.bulk_supercell)
                == "Split Vacancy"
            )

    # def test_Ga2O3_generate_complex_from_defect_sites(self):
    #     """
    #     Test the ``generate_complex_from_defect_sites`` function, using R-3c
    #     Ga2O3 as an example case.
    #     """
    #     # TODO
    #     self.Ga2O3_defect_gen


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
        assert not is_periodic_image(_get_molecule_coords(self.mol), _get_molecule_coords(translated_mol))

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
        assert not is_periodic_image(_get_molecule_coords(self.mol), _get_molecule_coords(rotated_mol))

    def test_permutation_invariance(self):
        """
        Test molecule matching with permutation invariance.

        Standard KabschMatcher does not work here.
        """
        permuted_mol = Molecule(self.mol.species, [self.mol.sites[i].coords for i in [1, 2, 0]])
        assert are_equivalent_molecules(self.mol, permuted_mol)
        assert not np.isclose(KabschMatcher(self.mol).fit(permuted_mol)[-1], 0)

        # now it is counted as a periodic image as it's the same coordinates, just in a different order:
        assert is_periodic_image(_get_molecule_coords(self.mol), _get_molecule_coords(permuted_mol))

    def test_is_periodic_image(self):
        """
        Test the ``is_periodic_image`` function.
        """
        orig_frac_coords = [
            self.primitive_structure.lattice.get_fractional_coords(coords)
            for coords in _get_molecule_coords(self.mol)
        ]
        # test all shifted with the _same_ periodic image:
        periodically_shifted_frac_coords = orig_frac_coords + np.array([1, 0, -1])

        assert is_coord_subset_pbc(orig_frac_coords, periodically_shifted_frac_coords)
        assert is_periodic_image(orig_frac_coords, periodically_shifted_frac_coords)
        assert is_periodic_image(orig_frac_coords, periodically_shifted_frac_coords, same_image=True)

        # now test shifted with _different_ periodic images:
        periodically_shifted_frac_coords[0] = periodically_shifted_frac_coords[0] + np.array([3, -2, 1])
        assert is_coord_subset_pbc(orig_frac_coords, periodically_shifted_frac_coords)
        assert is_periodic_image(orig_frac_coords, periodically_shifted_frac_coords)
        assert not is_periodic_image(orig_frac_coords, periodically_shifted_frac_coords, same_image=True)

        # test permutation invariance:
        permuted_frac_coords = np.array([orig_frac_coords[i] for i in [1, 2, 0]])
        assert is_coord_subset_pbc(orig_frac_coords, permuted_frac_coords)
        assert is_periodic_image(orig_frac_coords, permuted_frac_coords)
        assert is_periodic_image(orig_frac_coords, permuted_frac_coords, same_image=True)

        # test unique matching (no duplicate matches allowed):
        duplicate_frac_coords = np.array([orig_frac_coords[i] for i in [0, 0, 2]])
        assert not is_coord_subset_pbc(orig_frac_coords, duplicate_frac_coords)
        assert not is_periodic_image(orig_frac_coords, duplicate_frac_coords)
        assert not is_periodic_image(orig_frac_coords, duplicate_frac_coords, same_image=True)

        duplicate_frac_coords[0] += np.array([1, 0, -1])
        assert not is_coord_subset_pbc(orig_frac_coords, duplicate_frac_coords)
        assert not is_periodic_image(orig_frac_coords, duplicate_frac_coords)
        assert not is_periodic_image(orig_frac_coords, duplicate_frac_coords, same_image=True)


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

    def test_symmetry_multiplicity_R3c_Ga2O3(self):
        """
        Test symmetry and multiplicity analysis for split vacancies in R-3c
        Ga2O3.

        Reference values have been manually checked and stored in the info
        dicts.

        For R-3c Ga2O3, we have periodicity breaking in the generated supercell
        which breaks the rotational (but not inversion) symmetries, so C3i ->
        Ci, C2 -> C1. Useful test case for future stenciling to avoid
        periodicity breaking.
        """
        info_dict = self.R3c_Ga2O3_split_vac_info_dict
        bulk_sga = get_sga(info_dict["bulk_supercell"])
        bulk_supercell_symm_ops = bulk_sga.get_symmetry_operations()
        bulk_prim = get_primitive_structure(info_dict["bulk_supercell"])
        supercell_over_prim_factor = len(info_dict["bulk_supercell"]) / len(bulk_prim)
        molecule_dict = {}
        for cation_dict in info_dict.values():
            if isinstance(cation_dict, dict) and "split_vacancies_energy_dict" in cation_dict:
                for energy, subdict in cation_dict["split_vacancies_energy_dict"].items():
                    if self.energy_key in subdict:
                        print(f"energy: {energy}")
                        orig_split_vacancy = generate_complex_from_defect_sites(
                            info_dict["bulk_supercell"],
                            vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                            interstitial_sites=[subdict["interstitial_site"]],
                        )
                        orig_split_vacancy_symm = point_symmetry_from_structure(
                            orig_split_vacancy, info_dict["bulk_supercell"], relaxed=True, verbose=False
                        )
                        assert orig_split_vacancy_symm == subdict["unrelaxed_split_vac_symm"]
                        vac1_symm = point_symmetry_from_site(
                            subdict["vac_defect_1_site"], info_dict["bulk_supercell"]
                        )
                        assert vac1_symm == "C3"  # only C3 Ga sites in R-3c Ga2O3
                        assert vac1_symm == subdict["vac_1_symm"]
                        vac2_symm = point_symmetry_from_site(
                            subdict["vac_defect_2_site"], info_dict["bulk_supercell"]
                        )
                        assert vac2_symm == "C3"  # only C3 Ga sites in R-3c Ga2O3
                        assert vac2_symm == subdict["vac_2_symm"]
                        int_symm = point_symmetry_from_site(
                            subdict["interstitial_site"], info_dict["bulk_supercell"]
                        )
                        assert int_symm in [
                            "C2",
                            "C3i",
                            "D3",
                        ]  # unrelaxed interstitial sites in R-3c Ga2O3
                        assert int_symm == subdict["int_symm"]
                        relaxed_symm = point_symmetry_from_structure(
                            Structure.from_ase_atoms(subdict[self.energy_key]["struct_atoms"]),
                            info_dict["bulk_supercell"],
                            verbose=False,
                        )
                        assert relaxed_symm == subdict["relaxed_split_vac_symm"]

                        comp_mult = get_complex_defect_multiplicity(
                            bulk_supercell=info_dict["bulk_supercell"],
                            vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                            interstitial_sites=[subdict["interstitial_site"]],
                        )
                        assert comp_mult == subdict["multiplicity"]
                        comp_mult = get_complex_defect_multiplicity(
                            bulk_supercell=info_dict["bulk_supercell"],
                            vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                            interstitial_sites=[subdict["interstitial_site"]],
                            primitive_structure=bulk_prim,
                            supercell_symm_ops=bulk_supercell_symm_ops,
                        )  # same answer with efficiency options
                        assert comp_mult == subdict["multiplicity"]
                        supercell_comp_mult = get_complex_defect_multiplicity(
                            bulk_supercell=info_dict["bulk_supercell"],
                            vacancy_sites=[subdict["vac_defect_1_site"], subdict["vac_defect_2_site"]],
                            interstitial_sites=[subdict["interstitial_site"]],
                            primitive_structure=bulk_prim,
                            supercell_symm_ops=bulk_supercell_symm_ops,
                            primitive_cell_multiplicity=False,
                        )
                        assert supercell_comp_mult == subdict["multiplicity"] * supercell_over_prim_factor

                        int_obj = Interstitial(info_dict["bulk_supercell"], subdict["interstitial_site"])
                        v1_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_1_site"])
                        v2_obj = Vacancy(info_dict["bulk_supercell"], subdict["vac_defect_2_site"])

                        assert (
                            int_obj.multiplicity
                            == subdict["int_multiplicity"] * supercell_over_prim_factor
                        )
                        assert (
                            v1_obj.multiplicity
                            == subdict["vac_1_multiplicity"] * supercell_over_prim_factor
                        )
                        assert (
                            v2_obj.multiplicity
                            == subdict["vac_2_multiplicity"] * supercell_over_prim_factor
                        )

                        # this isn't universally true for complex defects, but is for the split vacancies:
                        assert (
                            comp_mult <= int_obj.multiplicity * v1_obj.multiplicity * v2_obj.multiplicity
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
                                    site.frac_coords
                                    + raw_complex_defect_sites[0].distance_and_image(site)[1]
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


# TODO: Add similar C2/m tests

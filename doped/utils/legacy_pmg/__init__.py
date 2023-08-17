"""
This code has been copied over from pymatgen==2022.7.25, as it was deleted in
later versions.

This is a temporary measure while refactoring to use the new pymatgen-analysis-
defects package takes place.
"""
import copy
from abc import abstractmethod

from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher


class DefectCorrection(MSONable):
    """
    A Correction class modeled off the computed entry correction format.
    """

    @abstractmethod
    def get_correction(self, entry):
        """
        Returns correction for a single entry.

        Args:
            entry: A DefectEntry object.

        Returns:
            A single dictionary with the format
            correction_name: energy_correction

        Raises:
            CompatibilityError if entry is not compatible.
        """
        return

    def correct_entry(self, entry):
        """
        Corrects a single entry.

        Args:
            entry: A DefectEntry object.

        Returns:
            An processed entry.

        Raises:
            CompatibilityError if entry is not compatible.
        """
        entry.correction.update(self.get_correction(entry))
        return entry


class PointDefectComparator(MSONable):
    """
    A class that matches pymatgen Point Defect objects even if their Cartesian
    coordinates are different (compares sublattices for the defect).

    NOTE: for defect complexes (more than a single defect),
    this comparator will break.
    """

    def __init__(self, check_charge=False, check_primitive_cell=False, check_lattice_scale=False):
        """
        Args:
            check_charge (bool): Gives option to check
                if charges are identical.
                Default is False (different charged defects can be same)
            check_primitive_cell (bool): Gives option to
                compare different supercells of bulk_structure,
                rather than directly compare supercell sizes
                Default is False (requires bulk_structure in each defect to be same size)
            check_lattice_scale (bool): Gives option to scale volumes of
                structures to each other identical lattice constants.
                Default is False (enforces same
                lattice constants in both structures).
        """
        self.check_charge = check_charge
        self.check_primitive_cell = check_primitive_cell
        self.check_lattice_scale = check_lattice_scale

    def are_equal(self, d1, d2):
        """
        Args:
            d1: First defect. A pymatgen Defect object.
            d2: Second defect. A pymatgen Defect object.

        Returns:
            True if defects are identical in type and sublattice.
        """
        if not isinstance(d1, d2.__class__):
            return False
        if d1.site.specie != d2.site.specie:
            return False

        sm = StructureMatcher(
            ltol=0.01,
            primitive_cell=self.check_primitive_cell,
            scale=self.check_lattice_scale,
        )

        if not sm.fit(d1.structure, d2.structure):
            return False

        d1 = copy.deepcopy(d1)
        d2 = copy.deepcopy(d2)

        return sm.fit(d1.defect_structure, d2.defect_structure)  # edited by SK to work with new pmg

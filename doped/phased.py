"""
Improved version of the DefectPhaseDiagram
"""

import logging
from functools import lru_cache

import numpy as np
from pymatgen.analysis.defects.thermodynamics import DefectPhaseDiagram, HalfspaceIntersection
from pymatgen.analysis.structure_matcher import PointDefectComparator

logger = logging.getLogger(__name__)

class PointDefectComparatorWithCache(PointDefectComparator):
    
    @lru_cache(maxsize=100000)
    def are_equal(self, d1, d2):
        return super().are_equal(d1, d2)


class BetterDefectPhaseDiagram(DefectPhaseDiagram):
    """
    Enhanced version of the `DefectPhaseDiagram`
    """

    def find_stable_charges(self):
        """
        Sets the stable charges and transition states for a series of
        defect entries. This function uses scipy's HalfspaceInterection
        to oncstruct the polygons corresponding to defect stability as
        a function of the Fermi-level. The Halfspace Intersection
        constructs N-dimensional hyperplanes, in this case N=2,  based
        on the equation of defect formation energy with considering chemical
        potentials:
            E_form = E_0^{Corrected} + Q_{defect}*(E_{VBM} + E_{Fermi})

        Extra hyperplanes are constructed to bound this space so that
        the algorithm can actually find enclosed region.

        This code was modeled after the Halfspace Intersection code for
        the Pourbaix Diagram
        """



        # Limits for search
        # E_fermi = { -1 eV to band gap+1}
        # E_formation = { (min(Eform) - 30) to (max(Eform) + 30)}
        all_eform = [
            one_def.formation_energy(fermi_level=self.band_gap / 2.0)
            for one_def in self.entries
        ]
        min_y_lim = min(all_eform) - 30
        max_y_lim = max(all_eform) + 30
        limits = [[-1, self.band_gap + 1], [min_y_lim, max_y_lim]]

        stable_entries = {}
        finished_charges = {}
        transition_level_map = {}

        # Grouping by defect types
        for defects, index_list in similar_defects(tuple(self.entries)):
            defects = list(defects)

            # prepping coefficient matrix for half-space intersection
            # [-Q, 1, -1*(E_form+Q*VBM)] -> -Q*E_fermi+E+-1*(E_form+Q*VBM) <= 0  where E_fermi and E are the variables
            # in the hyperplanes
            hyperplanes = np.array(
                [
                    [
                        -1.0 * entry.charge,
                        1,
                        -1.0 * (entry.energy + entry.charge * self.vbm),
                    ]
                    for entry in defects
                ]
            )

            border_hyperplanes = [
                [-1, 0, limits[0][0]],
                [1, 0, -1 * limits[0][1]],
                [0, -1, limits[1][0]],
                [0, 1, -1 * limits[1][1]],
            ]
            hs_hyperplanes = np.vstack([hyperplanes, border_hyperplanes])

            interior_point = [self.band_gap / 2, min(all_eform) - 1.0]

            hs_ints = HalfspaceIntersection(hs_hyperplanes, np.array(interior_point))

            # Group the intersections and coresponding facets
            ints_and_facets = zip(hs_ints.intersections, hs_ints.dual_facets)
            # Only inlcude the facets corresponding to entries, not the boundaries
            total_entries = len(defects)
            ints_and_facets = filter(
                lambda int_and_facet: all(np.array(int_and_facet[1]) < total_entries),
                ints_and_facets,
            )
            # sort based on transition level
            ints_and_facets = list(
                sorted(ints_and_facets, key=lambda int_and_facet: int_and_facet[0][0])
            )

            # log a defect name for tracking (using full index list to avoid naming
            # in-equivalent defects with same name)
            str_index_list = [str(ind) for ind in sorted(index_list)]
            track_name = defects[0].name + "@" + str("-".join(str_index_list))

            if len(ints_and_facets):
                # Unpack into lists
                _, facets = zip(*ints_and_facets)
                # Map of transition level: charge states

                transition_level_map[track_name] = {
                    intersection[0]: [defects[i].charge for i in facet]
                    for intersection, facet in ints_and_facets
                }

                stable_entries[track_name] = list(
                    set([defects[i] for dual in facets for i in dual])
                )

                finished_charges[track_name] = [defect.charge for defect in defects]
            else:
                # if ints_and_facets is empty, then there is likely only one defect...
                if len(defects) != 1:
                    # confirm formation energies dominant for one defect over other identical defects
                    name_set = [
                        one_def.name + "_chg" + str(one_def.charge)
                        for one_def in defects
                    ]
                    vb_list = [
                        one_def.formation_energy(fermi_level=limits[0][0])
                        for one_def in defects
                    ]
                    cb_list = [
                        one_def.formation_energy(fermi_level=limits[0][1])
                        for one_def in defects
                    ]

                    vbm_def_index = vb_list.index(min(vb_list))
                    name_stable_below_vbm = name_set[vbm_def_index]
                    cbm_def_index = cb_list.index(min(cb_list))
                    name_stable_above_cbm = name_set[cbm_def_index]

                    if name_stable_below_vbm != name_stable_above_cbm:
                        raise ValueError(
                            "HalfSpace identified only one stable charge out of list: {}\n"
                            "But {} is stable below vbm and {} is "
                            "stable above cbm.\nList of VBM formation energies: {}\n"
                            "List of CBM formation energies: {}"
                            "".format(
                                name_set,
                                name_stable_below_vbm,
                                name_stable_above_cbm,
                                vb_list,
                                cb_list,
                            )
                        )
                    else:
                        logger.info(
                            "{} is only stable defect out of {}".format(
                                name_stable_below_vbm, name_set
                            )
                        )
                        transition_level_map[track_name] = {}
                        stable_entries[track_name] = list([defects[vbm_def_index]])
                        finished_charges[track_name] = [
                            one_def.charge for one_def in defects
                        ]
                else:
                    transition_level_map[track_name] = {}

                    stable_entries[track_name] = list([defects[0]])

                    finished_charges[track_name] = [defects[0].charge]

        self.transition_level_map = transition_level_map
        self.transition_levels = {
            defect_name: list(defect_tls.keys())
            for defect_name, defect_tls in transition_level_map.items()
        }
        self.stable_entries = stable_entries
        self.finished_charges = finished_charges
        self.stable_charges = {
            defect_name: [entry.charge for entry in entries]
            for defect_name, entries in stable_entries.items()
        }


@lru_cache(maxsize=1000)
def similar_defects(entryset):
    """
    Used for grouping similar defects of different charges
    Can distinguish identical defects even if they are not in same position
    """
    pdc = PointDefectComparator(
        check_charge=False, check_primitive_cell=True, check_lattice_scale=False
    )
    grp_def_sets = []
    grp_def_indices = []
    for ent_ind, ent in enumerate(entryset):
        # TODO: more pythonic way of grouping entry sets with PointDefectComparator.
        # this is currently most time intensive part of DefectPhaseDiagram
        matched_ind = None
        for grp_ind, defgrp in enumerate(grp_def_sets):
            if pdc.are_equal(ent.defect, defgrp[0].defect):
                matched_ind = grp_ind
                break
        if matched_ind is not None:
            grp_def_sets[matched_ind].append(ent.copy())
            grp_def_indices[matched_ind].append(ent_ind)
        else:
            grp_def_sets.append([ent.copy()])
            grp_def_indices.append([ent_ind])

    return zip(grp_def_sets, grp_def_indices)
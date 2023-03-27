# coding: utf-8

from __future__ import division

__status__ = "Development"

import warnings
from copy import deepcopy


class LDAUCorrection(object):
    """
    Correction method for defect formation energies and
    transition levels that implements the scheme proposed by
    Janotti and Van de Walle. The scheme uses the LDA and LDA+U
    gaps and transition levels along with experimental gap to
    compute the corrections to transition levels and energies
    Reference:
        A. Janotti, C. G. Van de Walle, PRB 76, 165202 (2007)
    Args:
        exp_gap: Experimental gap
        ldau_gap: Computed gap with LDA+U (or GGA+U)
        lda_gap: Computed gap with LDA (or GGA)
    """

    def __init__(self, exp_gap, ldau_gap, lda_gap):
        self.exp_gap = exp_gap
        self.ldau_gap = ldau_gap
        self.lda_gap = lda_gap

    def get_transition_correction(self, ldau_transition, lda_transition):
        """
        Correct the LDA+U transition level using LDA and LDA+U transition
        levels and bandgaps and experimental bandgap
        Args:
            ldau_transition: Transition level computed with LDA+U (or GGA+U)
            lda_transition: Transition computed with LDA (or GGA)
        Returns:
            Correction to the input LDA+U transition level
        """
        diff = (ldau_transition - lda_transition) / (self.ldau_gap - self.lda_gap)
        return diff * (self.exp_gap - self.ldau_gap)

    def get_energy_correction(self, occupancy, ldau_transition, lda_transition):
        """
        Correct the LDA+U defect formation energy using LDA and LDA+U
        transition levels and bandgaps and experimental bandgap
        Args:
            occupancy: Defect level occupancy
            ldau_transition: Transition level computed with LDA+U (or GGA+U)
            lda_transition: Transition computed with LDA (or GGA)
        """
        trans_correction = self.get_transition_correction(
            ldau_transition, lda_transition
        )
        return trans_correction * occupancy


def get_ldau_corrections(
    exp_gap, ldau_gap, lda_gap, ldau_trans, lda_trans, occupancies
):
    """
        NOTE from developers:
            As of 12/15/17 this code does not have unit test
            TODO: benchmark implementation and make unit test
    Compute the corrections to LDA+U computed transition levels and defect
    formation energies using the scheme by Janotti and van de Walle
    Reference:
        A. Janotti, C. G. Van de Walle, PRB 76, 165202 (2007)
    Args:
        exp_gap: Experimental bandgap
        ldau_gap: Computed gap with LDA+U (or GGA+U)
        lda_gap: Computed gap with LDA (or GGA)
        ldau_trans: LDA+U Transition levels between defect charges as dict
            Ex: {'vac_1_Cr': {(0,-1): 0.3, (0,-2): 0.1}}
        lda_trans: LDA Transition levels between defect charges as dict
            Ex: {'vac_1_Cr': {(0,-1): 0.3, (0,-2): 0.1}}
        occupancies: Occupancy of defect levels by electrons as dict
            Ex: {'vac_1_Cr': {0: 0, -2: 2}}
    Returns:
        (transition_level_corrections, energy_corrections)
    """
    # TODO: Use better variable names
    print("occ inside ggau_corr func")
    print(occupancies)
    energy_corrections = {}
    transition_corrections = {}
    corrector = LDAUCorrection(exp_gap, ldau_gap, lda_gap)
    for defect_name in ldau_trans:
        print("def_name", defect_name)
        energy_corrections[defect_name] = {}
        transition_corrections[defect_name] = {}
        ggau_levels = ldau_trans[defect_name]
        gga_levels = lda_trans[defect_name]
        occ = occupancies[defect_name]
        print(occ)
        zero_occ_q = occ["0_occupancy"]
        print(defect_name, zero_occ_q)
        for trans_pair in ggau_levels:
            ggau_transit = ggau_levels[trans_pair]
            search_val = set(list(trans_pair))
            for trans_pair1 in gga_levels:
                match_val = set(list(trans_pair1))
                if match_val == search_val:
                    gga_transit = gga_levels[trans_pair1]
                    break
            q = (set(list(deepcopy(trans_pair))) - set([zero_occ_q])).pop()
            q_occ = occ[q]

            trans_corr = corrector.get_transition_correction(ggau_transit, gga_transit)
            print("trans_corr", defect_name, trans_pair, trans_corr)

            transition_corrections[defect_name][trans_pair] = trans_corr
            new_transit = ggau_transit + trans_corr
            print("new_level", defect_name, trans_pair, new_transit)

            if zero_occ_q in trans_pair:
                enrgy_corr = corrector.get_energy_correction(
                    q_occ, ggau_transit, gga_transit
                )
                energy_corrections[defect_name][q] = enrgy_corr

    return transition_corrections, energy_corrections

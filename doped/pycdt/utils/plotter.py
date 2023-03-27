#!/usr/bin/env python

__status__ = "Development"

import matplotlib
import numpy as np

matplotlib.use("agg")

import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt

warnings.warn(
    "Replaced PyCDT usage of DefectPlotter objects with plotting ability of"
    "DefectPhaseDiagram from pymatgen.analysis.defects.thermodynamics\n"
    "Will remove DefectPlotter with Version 2.5 of PyCDT.",
    DeprecationWarning,
)


class DefectPlotter(object):
    """
    Class performing all the typical plots from a defects study
    """

    def __init__(self, dpd):
        """
        Args:
            dpd: DefectPhaseDiagram object from pymatgen.analysis.defects.thermodynamics
        """

        self._dpd = dpd

    def get_plot_form_energy(
        self,
        mu_elts,
        xlim=None,
        ylim=None,
        ax_fontsize=1.3,
        lg_fontsize=1.0,
        lg_position=None,
    ):
        """
        Formation energy vs Fermi energy plot
        Args:
            mu_elts:
                a dictionnary of {Element:value} giving the chemical
                potential of each element
            xlim:
                Tuple (min,max) giving the range of the x (fermi energy) axis
            ylim:
                Tuple (min,max) giving the range for the formation energy axis
            ax_fontsize:
                float  multiplier to change axis label fontsize
            lg_fontsize:
                float  multiplier to change legend label fontsize
            lg_position:
                Tuple (horizontal-position, vertical-position) giving the position
                to place the legend.
                Example: (0.5,-0.75) will likely put it below the x-axis.

        Returns:
            a matplotlib object

        """
        if xlim is None:
            xlim = (-0.5, self._dpd.band_gap + 1.5)
        xy = {}
        lower_cap = -100.0
        upper_cap = 100.0
        for defnom, def_tl in self._dpd.transition_level_map.items():
            xy[defnom] = [[], []]
            if def_tl:
                org_x = list(def_tl.keys())  # list of transition levels
                org_x.sort()  # sorted with lowest first

                # establish lower x-bound
                first_charge = max(def_tl[org_x[0]])
                for chg_ent in self._dpd.stable_entries[defnom]:
                    if chg_ent.charge == first_charge:
                        form_en = chg_ent.formation_energy(
                            chemical_potentials=mu_elts, fermi_level=lower_cap
                        )
                xy[defnom][0].append(lower_cap)
                xy[defnom][1].append(form_en)

                # iterate over stable charge state transitions
                for fl in org_x:
                    charge = min(def_tl[fl])
                    for chg_ent in self._dpd.stable_entries[defnom]:
                        if chg_ent.charge == charge:
                            form_en = chg_ent.formation_energy(
                                chemical_potentials=mu_elts, fermi_level=fl
                            )
                    xy[defnom][0].append(fl)
                    xy[defnom][1].append(form_en)

                # establish upper x-bound
                last_charge = min(def_tl[org_x[-1]])
                for chg_ent in self._dpd.stable_entries[defnom]:
                    if chg_ent.charge == last_charge:
                        form_en = chg_ent.formation_energy(
                            chemical_potentials=mu_elts, fermi_level=upper_cap
                        )
                xy[defnom][0].append(upper_cap)
                xy[defnom][1].append(form_en)
            else:
                # no transition - just one stable charge
                chg_ent = self._dpd.stable_entries[defnom][0]
                for x_extrem in [lower_cap, upper_cap]:
                    xy[defnom][0].append(x_extrem)
                    xy[defnom][1].append(
                        chg_ent.formation_energy(
                            chemical_potentials=mu_elts, fermi_level=x_extrem
                        )
                    )

        width, height = 12, 8
        plt.clf()
        if len(xy) <= 8:
            colors = cm.Dark2(np.linspace(0, 1, len(xy)))
        else:
            colors = cm.gist_rainbow(np.linspace(0, 1, len(xy)))

        # plot formation energy lines
        for_legend = []
        for cnt, defnom in enumerate(xy.keys()):
            plt.plot(xy[defnom][0], xy[defnom][1], linewidth=3, color=colors[cnt])
            for_legend.append(self._dpd.stable_entries[defnom][0].copy())

        # plot transtition levels
        for cnt, defnom in enumerate(xy.keys()):
            x_trans, y_trans = [], []
            for x_val, chargeset in self._dpd.transition_level_map[defnom].items():
                x_trans.append(x_val)
                for chg_ent in self._dpd.stable_entries[defnom]:
                    if chg_ent.charge == chargeset[0]:
                        form_en = chg_ent.formation_energy(
                            chemical_potentials=mu_elts, fermi_level=x_val
                        )
                y_trans.append(form_en)
            if len(x_trans):
                plt.plot(
                    x_trans,
                    y_trans,
                    marker="*",
                    color=colors[cnt],
                    markersize=12,
                    fillstyle="full",
                )

        # get latex-like legend titles
        legends_txt = []
        for dfct in for_legend:
            flds = dfct.name.split("_")
            if "Vac" == flds[0]:
                base = "$Vac"
                sub_str = "_{" + flds[1] + "}$"
            elif "Sub" == flds[0]:
                flds = dfct.name.split("_")
                base = "$" + flds[1]
                sub_str = "_{" + flds[3] + "}$"
            elif "Int" == flds[0]:
                base = "$" + flds[1]
                # sub_str = '_{i_{'+','.join(flds[2])+'}}$'
                sub_str = "_{inter}$"
            else:
                base = dfct.name
                sub_str = ""

            legends_txt.append(base + sub_str)

        if not lg_position:
            plt.legend(legends_txt, fontsize=lg_fontsize * width, loc=0)
        else:
            plt.legend(
                legends_txt,
                fontsize=lg_fontsize * width,
                ncol=3,
                loc="lower center",
                bbox_to_anchor=lg_position,
            )

        if ylim is not None:
            plt.ylim(ylim)

        plt.xlim(xlim)
        plt.plot(
            [xlim[0], xlim[1]], [0, 0], "k-"
        )  # black dashed line for Eformation = 0
        plt.axvline(
            x=0.0, linestyle="--", color="k", linewidth=3
        )  # black dashed lines for gap edges
        plt.axvline(x=self._dpd.band_gap, linestyle="--", color="k", linewidth=3)
        plt.xlabel("Fermi energy (eV)", size=ax_fontsize * width)
        plt.ylabel("Defect Formation\nEnergy (eV)", size=ax_fontsize * width)

        return plt


class StructureRelaxPlotter(object):
    """
    This class plots movement of atomic sites as a function of radius

    relaxation_data is list of [distance to defect, distance atom moved,
                                index in structure, percentage contribution to total relaxation]

    """

    def __init__(self, relaxation_data, sampling_radius):
        rd = relaxation_data[:]
        rd.sort()
        self.relaxation_data = np.array(rd)
        self.sampling_radius = sampling_radius

    def plot(self, title=""):
        plt.figure()
        plt.clf()
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Radius from Defect ($\AA$)", fontsize=20)

        ax1.plot(
            self.relaxation_data[:, 0],
            self.relaxation_data[:, 1],
            "k",
            marker="o",
            linestyle="--",
        )

        ax2 = ax1.twinx()
        ax2.plot(
            self.relaxation_data[:, 0],
            self.relaxation_data[:, 3],
            "k",
            marker="o",
            linestyle="--",
        )

        tmpx = [self.sampling_radius, max(self.relaxation_data[:, 0])]
        max_fill = max(self.relaxation_data[:, 3])
        min_fill = min(self.relaxation_data[:, 3])
        plt.fill_between(
            tmpx,
            min_fill,
            max_fill,
            facecolor="red",
            alpha=0.15,
            label="delocalization region",
        )

        ax1.set_ylabel("Relaxation amount ($\AA$)", color="b", fontsize=15)
        ax2.set_ylabel("Percentage of total relaxation (%)\n", color="r", fontsize=15)
        plt.legend(loc=0)

        plt.title(str(title) + " Atomic Relaxation", fontsize=20)

        return plt


class SingleParticlePlotter(object):
    """
    This class plots single particle KS wavefunction as a function of radiusfrom defect

    relaxation_data is list of [distance to defect, distance atom moved,
                                index in structure, percentage contribution to total relaxation]

    """

    def __init__(self, defect_ks_delocal_data):
        self.defect_ks_delocal_data = defect_ks_delocal_data
        self.nspin = len(defect_ks_delocal_data["localized_band_indices"])
        lbl_dict = defect_ks_delocal_data["localized_band_indices"]
        self.localized_bands = set(
            [band_index for spin_list in lbl_dict.values() for band_index in spin_list]
        )
        print("Localized KS wavefunction bands are {}".format(self.localized_bands))

    def plot(self, bandnum, title=""):
        if bandnum not in self.localized_bands:
            raise ValueError("{} is not in {}".format(bandnum, self.localized_bands))

        plt.figure()
        plt.clf()
        fig, ax1 = plt.subplots()

        final_out = self.defect_ks_delocal_data["followup_wf_parse"][bandnum]
        dat = final_out["0"]["rad_dist_data"]["tot"]

        ax1.set_xlabel("Radius from Defect ($\AA$)")
        ax1.plot(dat[0], dat[1], "b")
        ax2 = ax1.twinx()
        ax2.plot(dat[0], 100.0 * np.array(dat[2]), "r")

        if self.nspin == 2:
            dat = final_out["1"]["rad_dist_data"]["tot"]
            ax1.plot(dat[0], -1.0 * np.array(dat[1]), "b-")
            ax2.plot(dat[0], -100.0 * np.array(dat[2]), "r-")

        ax1.set_ylabel("Occupation", color="b")
        ax2.set_ylabel("Percentage of electron contained\n", color="r")
        plt.title(str(title) + " KS band index " + str(bandnum))

        return plt

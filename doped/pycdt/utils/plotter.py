#!/usr/bin/env python

__status__ = "Development"

import matplotlib
import numpy as np

matplotlib.use("agg")

import matplotlib.pyplot as plt

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

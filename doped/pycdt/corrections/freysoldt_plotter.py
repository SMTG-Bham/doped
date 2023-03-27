#!/usr/bin/env python

"""
This is just for plotting the axis files that come out of the sxdefectalignwrapper script. 
Created 8/11/15 (right after I finished the first version of the wrapper class)
"""

__status__ = "Development"

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp.outputs import Locpot

warnings.warn(
    "Replacing PyCDT usage of Freysoldt base classes with calls to "
    "corresponding objects in pymatgen.analysis.defects.corrections\n"
    "All correction plotting functionalities exist within pymatgen v2019.5.1."
    "Version 2.5 of PyCDT will remove pycdt.corrections.freysoldt_plotter.FreysoldtPlot.",
    DeprecationWarning,
)


class FreysoldtPlot(object):
    """
        NOTE from developers:
            This code has excessive electrostatic potential
            plotting methods. This code will not ever be used
            by command line option and has unit test
            It will not be maintained
            going forward (as of 12/15/2017).
            However, we are keeping function here to allow for
            current users to make use of it...
    This class applies the Freysoldt correction to remove electrostatic defect
    interaction contribution to energy and apply potential alignment.
    Ideally this sxdefectalign wrapper class should be replaced by a python
    code implementing a generalized anisotropic Freysoldt correction
    """

    def __init__(
        self, pathforaxis, site_frac_coords, name="", locpotbulk="", locpotdef=""
    ):
        """
        Args:
            pathforaxis:
                path for axis1vline-ev.dat type files.
            site_frac_coords:
                Fractional coordinates of defect site as list
            name:
                Name of the defect to write files
        """
        self._path = pathforaxis
        self._frac_coords = site_frac_coords
        self._name = name
        self._locpot_bulk = locpotbulk
        self._locpot_defect = locpotdef

    def plot_hartree_pot(self):
        # plot planar averages of bulk and defect (good for seeing global changes)
        if not self._locpot_bulk:
            print(
                "did not feed path to locpot file! Need this for plotting hartree pot"
            )
            return
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Locpot planar averaged potentials")
        bulkloc = Locpot.from_file(self._locpot_bulk)
        defloc = Locpot.from_file(self._locpot_defect)
        get_agrid = bulkloc.get_axis_grid
        get_baavg = bulkloc.get_average_along_axis
        get_daavg = defloc.get_average_along_axis
        for axis in [0, 1, 2]:
            ax = fig.add_subplot(3, 1, axis + 1)
            latt_len = self._lengths[axis]
            ax.plot(get_agrid(axis), get_baavg(axis), "r", label="Bulk potential")
            ax.plot(get_agrid(axis), get_daavg(axis), "b", label="Defect potential")
            ax.plot(
                [self._frac_coords[axis] * latt_len],
                [0],
                "or",
                markersize=4.0,
                label="Defect site",
            )
            ax.set_ylabel("axis " + str(axis + 1))
            if not axis:
                ax.legend()
        ax.set_xlabel("distance (Angstrom)")
        # plt.savefig('locpotavgplot.png')
        plt.show()

    def plot_hartree_pot_diff(self):
        # only plot the difference in planar averaged potentials
        if not self._locpot_bulk:
            print(
                "did not feed path to locpot file! Need this for plotting hartree pot diff"
            )
            return
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Locpot planar averaged potential difference")
        bulkloc = Locpot.from_file(self._locpot_bulk)
        defloc = Locpot.from_file(self._locpot_defect)
        for axis in [0, 1, 2]:
            ax = fig.add_subplot(3, 1, axis + 1)
            defect_axis = defloc.get_axis_grid(axis)
            defect_pot = defloc.get_average_along_axis(axis)
            pure_pot = bulkloc.get_average_along_axis(axis)
            latt_len = self._lengths[axis]
            ax.plot(
                defect_axis, defect_pot - pure_pot, "b", label="Defect-Bulk difference"
            )
            ax.plot(
                [self._frac_coords[axis] * latt_len],
                [0],
                "or",
                markersize=4.0,
                label="Defect site",
            )
            ax.set_ylabel("axis " + str(axis + 1))
            if not axis:
                ax.legend()
        ax.set_xlabel("distance (Angstrom)")
        # plt.savefig('locpotdiffplot.png')
        plt.show()

    def plot_all_hartree_pot(self):
        # plot planar averaged locpots, along with the difference for each axis
        if not self._locpot_bulk:
            print(
                "did not feed path to locpot file! Need this for plotting hartree pot"
            )
            return
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title("Locpot planar averaged potentials and difference")
        bulkloc = Locpot.from_file(self._locpot_bulk)
        defloc = Locpot.from_file(self._locpot_defect)
        for axis in [0, 1, 2]:
            ax = fig.add_subplot(3, 1, axis + 1)
            defect_axis = defloc.get_axis_grid(axis)
            defect_pot = defloc.get_average_along_axis(axis)
            pure_pot = bulkloc.get_average_along_axis(axis)
            ax.plot(defect_axis, pure_pot, "r", label="Bulk potential")
            ax.plot(defect_axis, defect_pot, "b", label="Defect potential")
            ax.plot(
                defect_axis, defect_pot - pure_pot, "k", label="Defect-Bulk difference"
            )
            ax.plot(
                [self._frac_coords[axis] * self._lengths[axis]],
                [0],
                "or",
                markersize=4.0,
                label="Defect site",
            )
            ax.set_ylabel("axis " + str(axis + 1))
            if not axis:
                ax.legend()
        ax.set_xlabel("distance (Angstrom)")
        # plt.savefig('locpotavgdiffplot.png')
        plt.show()

    def plot_vline(self):
        """
        Will plot the vline files based on whatever local directory you provided
        """
        plotvals = {"0": {}, "1": {}, "2": {}}
        platy = []
        for axis in [0, 1, 2]:
            print("do axis " + str(axis + 1))
            x_lr, y_lr = [], []
            x, y = [], []
            x_diff, y_diff = [], []
            loc = os.path.abspath(self._path)
            if not self._name:
                nom = str(loc) + "/axis" + str(axis) + "vline-eV.dat"
            else:
                nom = (
                    str(loc)
                    + "/"
                    + str(self._name)
                    + "axis"
                    + str(axis)
                    + "vline-eV.dat"
                )
            with open(nom, "r") as f_sr:  # read in potential
                for r in f_sr:
                    tmp = r.split("\t")
                    if len(tmp) < 3 and not r.startswith("&"):
                        x_lr.append(float(tmp[0]) / 1.889725989)
                        y_lr.append(float(tmp[1]))
                    if len(tmp) > 2:
                        x.append(float(tmp[0]) / 1.889725989)
                        x_diff.append(float(tmp[0]) / 1.889725989)
                        y.append(float(tmp[2].rstrip("\n")))
                        y_diff.append(float(tmp[1]))

            # Extract potential alignment term averaging window of +/- 1 Ang
            # around point halfway between neighboring defects

            latt_len = max(x_diff)
            if self._frac_coords[axis] >= 0.5:
                platx = (self._frac_coords[axis] - 0.5) * latt_len
            else:
                platx = (self._frac_coords[axis] + 0.5) * latt_len
            print("half way between defects is: ", platx)

            xmin = latt_len - (1 - platx) if platx < 1 else platx - 1
            xmax = 1 - (latt_len - platx) if platx > latt_len - 1 else 1 + platx
            print("means sampling region is (", xmin, ",", xmax, ")")

            tmpalign = []
            if xmax < xmin:
                print("wrap around detected, special alignment needed")
                for i in range(len(x)):
                    if x[i] < xmax or x[i] > xmin:
                        tmpalign.append(y[i])
                    else:
                        continue
            else:
                for i in range(len(x)):
                    if x[i] > xmin and x[i] < xmax:
                        tmpalign.append(y[i])
                    else:
                        continue

            print("alignment is ", -np.mean(tmpalign))
            platy.append(-np.mean(tmpalign))
            flag = 0
            for i in tmpalign:  # check to see if alignment region varies too much
                if np.abs(i - platy[-1]) > 0.2:
                    flag = 1
                else:
                    continue
            if flag != 0:
                print(
                    "Warning: potential aligned region varied by more "
                    + "than 0.2eV (in range of halfway between defects "
                    + "+/-1 \Angstrom). Might have issues with Freidel "
                    + "oscillations or atomic relaxation\n"
                )

            plotvals[str(axis)]["xylong"] = [x_lr, y_lr]
            plotvals[str(axis)]["xy"] = [x, y]
            plotvals[str(axis)]["xydiff"] = [x_diff, y_diff]

        fig = plt.figure(figsize=(15.0, 12.0))
        for axis in [0, 1, 2]:
            print("plot axis ", axis + 1)
            ax = fig.add_subplot(3, 1, axis + 1)
            ax.set_ylabel("axis " + str(axis + 1))
            vals_plot = plotvals[str(axis)]
            ax.plot(vals_plot["xy"][0], vals_plot["xy"][1])
            ax.plot(vals_plot["xydiff"][0], vals_plot["xydiff"][1], "r")
            ax.plot(vals_plot["xylong"][0], vals_plot["xylong"][1], "g")

            if not axis:
                ax.set_title("Electrostatic planar averaged potential")
                ax.legend(["V_defect-V_ref-V_lr", "V_defect-V_ref", "V_lr"])

            latt_len = max(vals_plot["xydiff"][0])
            fcoords = self._frac_coords[axis]
            ax.plot([fcoords * latt_len], [0], "or", markersize=4.0)
            if fcoords >= 0.5:
                ax.plot([(fcoords - 0.5) * latt_len], [0], "og", markersize=4.0)
            else:
                ax.plot([(fcoords + 0.5) * latt_len], [0], "og", markersize=4.0)
            plt.axhline(y=0.0, linewidth=0.8, color="black")

        if not self._name:
            plt.savefig(os.path.join(self._path, "locpotgraph.png"))
        else:
            plt.savefig(os.path.join(self._path, str(self._name) + "locpotgraph.png"))

        return


if __name__ == "__main__":
    pass

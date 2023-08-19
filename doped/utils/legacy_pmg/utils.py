# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
"""
Utilities for defects module.
"""

import logging
import math
from copy import deepcopy

import numpy as np
from monty.json import MSONable
from numpy.linalg import norm
from pymatgen.analysis.local_env import LocalStructOrderParams, MinimumDistanceNN, cn_opt_params
from pymatgen.analysis.phase_diagram import get_facets
from pymatgen.core.periodic_table import Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    peak_local_max_found = True
except ImportError:
    peak_local_max_found = False

__author__ = "Danny Broberg, Shyam Dwaraknath, Bharat Medasani, Nils Zimmermann, Geoffroy Hautier"
__copyright__ = "Copyright 2014, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Danny Broberg, Shyam Dwaraknath"
__email__ = "dbroberg@berkeley.edu, shyamd@lbl.gov"
__status__ = "Development"
__date__ = "January 11, 2018"

logger = logging.getLogger(__name__)
hart_to_ev = 27.2114
ang_to_bohr = 1.8897
invang_to_ev = 3.80986
kumagai_to_V = 1.809512739e2  # = Electron charge * 1e10 / VacuumPermittivity Constant

motif_cn_op = {}
for cn, di in cn_opt_params.items():
    for motif, li in di.items():
        motif_cn_op[motif] = {"cn": int(cn), "optype": li[0]}
        motif_cn_op[motif]["params"] = deepcopy(li[1]) if len(li) > 1 else None


class QModel(MSONable):
    """
    Model for the defect charge distribution.

    A combination of exponential tail and gaussian distribution is used (see
    Freysoldt (2011), DOI: 10.1002/pssb.201046289 ) q_model(r) = q [x
    exp(-r/gamma) + (1-x) exp(-r^2/beta^2)]         without normalization
    constants By default, gaussian distribution with 1 Bohr width is assumed.
    If defect charge is more delocalized, exponential tail is suggested.
    """

    def __init__(self, beta=1.0, expnorm=0.0, gamma=1.0):
        """
        Args:
            beta: Gaussian decay constant. Default value is 1 Bohr.
                  When delocalized (eg. diamond), 2 Bohr is more appropriate.
            expnorm: Weight for the exponential tail in the range of [0-1].
                     Default is 0.0 indicating no tail .
                     For delocalized charges ideal value is around 0.54-0.6.
            gamma: Exponential decay constant.
        """
        self.beta = beta
        self.expnorm = expnorm
        self.gamma = gamma

        self.beta2 = beta * beta
        self.gamma2 = gamma * gamma
        if expnorm and not gamma:
            raise ValueError("Please supply exponential decay constant.")

    def rho_rec(self, g2):
        """
        Reciprocal space model charge value for input squared reciprocal
        vector.

        Args:
            g2: Square of reciprocal vector.

        Returns:
            Charge density at the reciprocal vector magnitude
        """
        return self.expnorm / np.sqrt(1 + self.gamma2 * g2) + (1 - self.expnorm) * np.exp(
            -0.25 * self.beta2 * g2
        )

    @property
    def rho_rec_limit0(self):
        """
        Reciprocal space model charge value.

        close to reciprocal vector 0 . rho_rec(g->0) -> 1 + rho_rec_limit0 *
        g^2.
        """
        return -2 * self.gamma2 * self.expnorm - 0.25 * self.beta2 * (1 - self.expnorm)


def eV_to_k(energy):
    """
    Convert energy to reciprocal vector magnitude k via hbar*k^2/2m
    Args:
        a: Energy in eV.

    Returns:
        (double) Reciprocal vector magnitude (units of 1/Bohr).
    """
    return math.sqrt(energy / invang_to_ev) * ang_to_bohr


def genrecip(a1, a2, a3, encut):
    """
    Generate reciprocal lattice vectors with energy less than the given cutoff.

    Args:
        a1 (type): Description of a1.
        a2 (type): Description of a2.
        a3 (type): Description of a3.
        encut (type): Energy cut off in eV.

    Returns:
        Reciprocal lattice vectors with energy less than encut.
    """
    vol = np.dot(a1, np.cross(a2, a3))  # 1/bohr^3
    b1 = (2 * np.pi / vol) * np.cross(a2, a3)  # units 1/bohr
    b2 = (2 * np.pi / vol) * np.cross(a3, a1)
    b3 = (2 * np.pi / vol) * np.cross(a1, a2)

    # create list of recip space vectors that satisfy |i*b1+j*b2+k*b3|<=encut
    G_cut = eV_to_k(encut)
    # Figure out max in all recipricol lattice directions
    i_max = int(math.ceil(G_cut / norm(b1)))
    j_max = int(math.ceil(G_cut / norm(b2)))
    k_max = int(math.ceil(G_cut / norm(b3)))

    # Build index list
    i = np.arange(-i_max, i_max)
    j = np.arange(-j_max, j_max)
    k = np.arange(-k_max, k_max)

    # Convert index to vectors using meshgrid
    indices = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3)
    # Multiply integer vectors to get recipricol space vectors
    vecs = np.dot(indices, [b1, b2, b3])
    # Calculate radii of all vectors
    radii = np.sqrt(np.einsum("ij,ij->i", vecs, vecs))

    # Yield based on radii
    for vec, r in zip(vecs, radii):
        if r < G_cut and r != 0:
            yield vec


def generate_reciprocal_vectors_squared(a1, a2, a3, encut):
    """
    Generate reciprocal vector magnitudes within the cutoff along the specified
    lattice vectors.

    Args:
        a1: Lattice vector a (in Bohrs)
        a2: Lattice vector b (in Bohrs)
        a3: Lattice vector c (in Bohrs)
        encut: Reciprocal vector energy cutoff.

    Returns:
        [[g1^2], [g2^2], ...] Square of reciprocal vectors (1/Bohr)^2
        determined by a1, a2, a3 and whose magntidue is less than gcut^2.
    """
    for vec in genrecip(a1, a2, a3, encut):
        yield np.dot(vec, vec)


def closestsites(struct_blk, struct_def, pos):
    """
    Returns closest site to the input position
    for both bulk and defect structures
    Args:
        struct_blk: Bulk structure
        struct_def: Defect structure
        pos: Position
    Return: (site object, dist, index).
    """
    blk_close_sites = struct_blk.get_sites_in_sphere(pos, 5, include_index=True)
    blk_close_sites.sort(key=lambda x: x[1])
    def_close_sites = struct_def.get_sites_in_sphere(pos, 5, include_index=True)
    def_close_sites.sort(key=lambda x: x[1])

    return blk_close_sites[0], def_close_sites[0]


class StructureMotifInterstitial:
    """
    Generate interstitial sites at positions where the interstitialcy is
    coordinated by nearest neighbors in a way that resembles basic structure
    motifs (e.g., tetrahedra, octahedra), using the InFiT (Interstitialcy
    Finding Tool) algorithm introduced by Nils E.

    R. Zimmermann, Matthew K. Horton, Anubhav Jain, and Maciej Haranczyk
    (10.3389/fmats.2017.00034).
    """

    def __init__(
        self,
        struct,
        inter_elem,
        motif_types=("tetrahedral", "octahedral"),
        op_threshs=(0.3, 0.5),
        dl=0.2,
        doverlap=1,
        facmaxdl=1.01,
        verbose=False,
    ):
        """
        Generates symmetrically distinct interstitial sites at positions where
        the interstitial is coordinated by nearest neighbors in a pattern that
        resembles a supported structure motif (e.g., tetrahedra, octahedra).

        Args:
            struct (Structure): input structure for which symmetrically
                distinct interstitial sites are to be found.
            inter_elem (string): element symbol of desired interstitial.
            motif_types ([string]): list of structure motif types that are
                to be considered. Permissible types are:
                tet (tetrahedron), oct (octahedron).
            op_threshs ([float]): threshold values for the underlying order
                parameters to still recognize a given structural motif
                (i.e., for an OP value >= threshold the coordination pattern
                match is positive, for OP < threshold the match is
                negative.
            dl (float): grid fineness in Angstrom. The input
                structure is divided into a grid of dimension
                a/dl x b/dl x c/dl along the three crystallographic
                directions, with a, b, and c being the lengths of
                the three lattice vectors of the input unit cell.
            doverlap (float): distance that is considered
                to flag an overlap between any trial interstitial site
                and a host atom.
            facmaxdl (float): factor to be multiplied with the maximum grid
                width that is then used as a cutoff distance for the
                clustering prune step.
            verbose (bool): flag indicating whether (True) or not (False;
                default) to print additional information to screen.
        """
        # Initialize interstitial finding.
        self._structure = struct.copy()
        self._motif_types = motif_types[:]
        if len(self._motif_types) == 0:
            raise RuntimeError("no motif types provided.")
        self._op_threshs = op_threshs[:]
        self.cn_motif_lostop = {}
        self.target_cns = []
        for motif in self._motif_types:
            if motif not in list(motif_cn_op.keys()):
                raise RuntimeError(f"unsupported motif type: {motif}.")
            cn = int(motif_cn_op[motif]["cn"])
            if cn not in self.target_cns:
                self.target_cns.append(cn)
            if cn not in list(self.cn_motif_lostop.keys()):
                self.cn_motif_lostop[cn] = {}
            tmp_optype = motif_cn_op[motif]["optype"]
            if tmp_optype == "tet_max":
                tmp_optype = "tet"
            if tmp_optype == "oct_max":
                tmp_optype = "oct"
            self.cn_motif_lostop[cn][motif] = LocalStructOrderParams(
                [tmp_optype], parameters=[motif_cn_op[motif]["params"]], cutoff=-10.0
            )
        self._dl = dl
        self._defect_sites = []
        self._defect_types = []
        self._defect_site_multiplicity = []
        self._defect_cns = []
        self._defect_opvals = []

        rots, trans = SpacegroupAnalyzer(struct)._get_symmetry()
        nbins = [
            int(struct.lattice.a / dl),
            int(struct.lattice.b / dl),
            int(struct.lattice.c / dl),
        ]
        dls = [
            struct.lattice.a / float(nbins[0]),
            struct.lattice.b / float(nbins[1]),
            struct.lattice.c / float(nbins[2]),
        ]
        maxdl = max(dls)
        if verbose:
            print(f"Grid size: {nbins[0]} {nbins[1]} {nbins[2]}")
            print(f"dls: {dls[0]} {dls[1]} {dls[2]}")
        struct_w_inter = struct.copy()
        struct_w_inter.append(inter_elem, [0, 0, 0])
        natoms = len(list(struct_w_inter.sites))
        trialsites = []

        # Build index list
        i = np.arange(0, nbins[0]) + 0.5
        j = np.arange(0, nbins[1]) + 0.5
        k = np.arange(0, nbins[2]) + 0.5

        # Convert index to vectors using meshgrid
        indices = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3)
        # Multiply integer vectors to get recipricol space vectors
        vecs = np.multiply(indices, np.divide(1, nbins))

        # Loop over trial positions that are based on a regular
        # grid in fractional coordinate space
        # within the unit cell.
        for vec in vecs:
            struct_w_inter.replace(natoms - 1, inter_elem, coords=vec, coords_are_cartesian=False)
            if (
                len(struct_w_inter.get_sites_in_sphere(struct_w_inter.sites[natoms - 1].coords, doverlap))
                == 1
            ):
                neighs_images_weigths = MinimumDistanceNN(tol=0.8, cutoff=6).get_nn_info(
                    struct_w_inter, natoms - 1
                )
                neighs_images_weigths_sorted = sorted(
                    neighs_images_weigths, key=lambda x: x["weight"], reverse=True
                )
                for nsite in range(1, len(neighs_images_weigths_sorted) + 1):
                    if nsite not in self.target_cns:
                        continue

                    allsites = [neighs_images_weigths_sorted[i]["site"] for i in range(nsite)]
                    indices_neighs = list(range(len(allsites)))
                    allsites.append(struct_w_inter.sites[natoms - 1])
                    for motif, ops in self.cn_motif_lostop[nsite].items():
                        opvals = ops.get_order_parameters(
                            allsites, len(allsites) - 1, indices_neighs=indices_neighs
                        )
                        if opvals[0] > op_threshs[motif_types.index(motif)]:
                            cns = {}
                            for isite in range(nsite):
                                site = neighs_images_weigths_sorted[isite]["site"]
                                if isinstance(site.specie, Element):
                                    elem = site.specie.symbol
                                else:
                                    elem = site.specie.element.symbol
                                if elem in list(cns.keys()):
                                    cns[elem] = cns[elem] + 1
                                else:
                                    cns[elem] = 1
                            trialsites.append(
                                {
                                    "mtype": motif,
                                    "opval": opvals[0],
                                    "coords": struct_w_inter.sites[natoms - 1].coords[:],
                                    "fracs": vec,
                                    "cns": dict(cns),
                                }
                            )
                            break

        # Prune list of trial sites by clustering and find the site
        # with the largest order parameter value in each cluster.
        nintersites = len(trialsites)
        unique_motifs = []
        for ts in trialsites:
            if ts["mtype"] not in unique_motifs:
                unique_motifs.append(ts["mtype"])
        labels = {}
        connected = []
        for i in range(nintersites):
            connected.append([])
            for j in range(nintersites):
                dist, image = struct_w_inter.lattice.get_distance_and_image(
                    trialsites[i]["fracs"], trialsites[j]["fracs"]
                )
                connected[i].append(bool(dist < (maxdl * facmaxdl)))
        include = []
        for motif in unique_motifs:
            labels[motif] = []
            for i, ts in enumerate(trialsites):
                labels[motif].append(i if ts["mtype"] == motif else -1)
            change = True
            while change:
                change = False
                for i in range(nintersites - 1):
                    if change:
                        break
                    if labels[motif][i] == -1:
                        continue
                    for j in range(i + 1, nintersites):
                        if labels[motif][j] == -1:
                            continue
                        if connected[i][j] and labels[motif][i] != labels[motif][j]:
                            if labels[motif][i] < labels[motif][j]:
                                labels[motif][j] = labels[motif][i]
                            else:
                                labels[motif][i] = labels[motif][j]
                            change = True
                            break
            unique_ids = []
            for label in labels[motif]:
                if label != -1 and label not in unique_ids:
                    unique_ids.append(label)
            if verbose:
                print(f"unique_ids {motif} {unique_ids}")
            for uid in unique_ids:
                maxq = 0.0
                imaxq = -1
                for i in range(nintersites):
                    if labels[motif][i] == uid and (imaxq < 0 or trialsites[i]["opval"] > maxq):
                        imaxq = i
                        maxq = trialsites[i]["opval"]
                include.append(imaxq)

        # Prune by symmetry.
        multiplicity = {}
        discard = []
        for motif in unique_motifs:
            discard_motif = []
            for indi, i in enumerate(include):
                if trialsites[i]["mtype"] != motif or i in discard_motif:
                    continue
                multiplicity[i] = 1
                symposlist = [trialsites[i]["fracs"].dot(np.array(m, dtype=float)) for m in rots]
                for t in trans:
                    symposlist.append(trialsites[i]["fracs"] + np.array(t))
                for indj in range(indi + 1, len(include)):
                    j = include[indj]
                    if trialsites[j]["mtype"] != motif or j in discard_motif:
                        continue
                    for sympos in symposlist:
                        dist, image = struct.lattice.get_distance_and_image(sympos, trialsites[j]["fracs"])
                        if dist < maxdl * facmaxdl:
                            discard_motif.append(j)
                            multiplicity[i] += 1
                            break
            for i in discard_motif:
                if i not in discard:
                    discard.append(i)

        if verbose:
            print(
                f"Initial trial sites: {len(trialsites),}\nAfter clustering: {len(include),}\n"
                f"After symmetry pruning: {len(include) - len(discard)}"
            )
        for i in include:
            if i not in discard:
                self._defect_sites.append(
                    PeriodicSite(
                        Element(inter_elem),
                        trialsites[i]["fracs"],
                        self._structure.lattice,
                        to_unit_cell=False,
                        coords_are_cartesian=False,
                        properties=None,
                    )
                )
                self._defect_types.append(trialsites[i]["mtype"])
                self._defect_cns.append(trialsites[i]["cns"])
                self._defect_site_multiplicity.append(multiplicity[i])
                self._defect_opvals.append(trialsites[i]["opval"])

    def enumerate_defectsites(self):
        """
        Get all defect sites.

        Returns:
            defect_sites ([PeriodicSite]): list of periodic sites
                    representing the interstitials.
        """
        return self._defect_sites

    def get_motif_type(self, i):
        """
        Get the motif type of defect with index i (e.g., "tet").

        Returns:
            motif (string): motif type.
        """
        return self._defect_types[i]

    def get_defectsite_multiplicity(self, n):
        """
        Returns the symmtric multiplicity of the defect site at the index.
        """
        return self._defect_site_multiplicity[n]

    def get_coordinating_elements_cns(self, i):
        """
        Get element-specific coordination numbers of defect with index i.

        Returns:
            elem_cn (dict): dictionary storing the coordination numbers (int)
                    with string representation of elements as keys.
                    (i.e., {elem1 (string): cn1 (int), ...}).
        """
        return self._defect_cns[i]

    def get_op_value(self, i):
        """
        Get order-parameter value of defect with index i.

        Returns:
            opval (float): OP value.
        """
        return self._defect_opvals[i]

    def make_supercells_with_defects(self, scaling_matrix):
        """
        Generate a sequence of supercells in which each supercell contains a
        single interstitial, except for the first supercell in the sequence
        which is a copy of the defect-free input structure.

        Args:
            scaling_matrix (3x3 integer array): scaling matrix
                to transform the lattice vectors.

        Returns:
            scs ([Structure]): sequence of supercells.
        """
        scs = []
        sc = self._structure.copy()
        sc.make_supercell(scaling_matrix)
        scs.append(sc)
        for ids, defect_site in enumerate(self._defect_sites):
            sc_with_inter = sc.copy()
            sc_with_inter.append(
                defect_site.species_string,
                defect_site.frac_coords,
                coords_are_cartesian=False,
                validate_proximity=False,
                properties=None,
            )
            if not sc_with_inter:
                raise RuntimeError(f"could not generate supercell with interstitial {ids + 1}")
            scs.append(sc_with_inter.copy())
        return scs


def calculate_vol(coords):
    """
    Calculate volume given a set of coords.

    :param coords: List of coords.
    :return: Volume
    """
    if len(coords) == 4:
        coords_affine = np.ones((4, 4))
        coords_affine[:, 0:3] = np.array(coords)
        return abs(np.linalg.det(coords_affine)) / 6

    simplices = get_facets(coords, joggle=True)
    center = np.average(coords, axis=0)
    vol = 0
    for s in simplices:
        c = [coords[i] for i in s]
        c.append(center)
        vol += calculate_vol(c)
    return vol


def converge(f, step, tol, max_h):
    """
    Simple newton iteration based convergence function.
    """
    g = f(0)
    dx = 10000
    h = step
    while dx > tol:
        g2 = f(h)
        dx = abs(g - g2)
        g = g2
        h += step

        if h > max_h:
            raise Exception(f"Did not converge before {h}")
    return g

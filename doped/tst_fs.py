from doped.fermi_solver import FermiSolverDoped
from monty.serialization import loadfn
import matplotlib.pyplot as plt

# we can use the doped style file to make the plots look consistent
style_file = "../doped/utils/doped.mplstyle"
plt.style.use(style_file)

# we need to specify the path to the vasprun.xml file
# that was used for the DOS calculation. This is because
# we need to accurately account for the electronic carrier concentrations
# as well as the defect concentrations
vasprun_path = "../examples/CdTe/CdTe_bulk/vasp_ncl/vasprun.xml.gz"

# the defect phase diagram contains all the information about the
# defect formation energies and transition levels. We'll use a version
# of the defect phase diagram that doesn't include the metastable states
# for the purposes of this example
thermodynamics = loadfn("../examples/CdTe/CdTe_thermo_v2.3_wout_meta.json")

# and the chemical potentials can then be used to specify the
# defect formation energies under different conditions, and act as a parameter
# space we can scan over to interrogate the defect concentrations
chemical_potentials = loadfn("../examples/CdTe/CdTe_chempots.json")

# initialize the FermiSolver object
fs = FermiSolverDoped(defect_thermodynamics=thermodynamics, dos=vasprun_path)
pdc = fs.scan_anneal_and_quench(chemical_potentials, annealing_temperatures=range(300, 1420, 20), quenching_temperatures=[300])
dc = fs.scan_temperature(chemical_potentials, temperature_range=range(300, 1420, 20))

pdc.to_csv("pseudo_equilibrium_concentrations.csv")

import seaborn as sns

plt.yscale("log")
sns.lineplot(data=dc, x="Temperature", y="Concentration (cm^-3)", hue="Defect", ls="--", marker="o")
sns.lineplot(data=pdc, x="Annealing Temperature", y="Total Concentration (cm^-3)", hue="Defect")
plt.show()

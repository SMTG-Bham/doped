"""This is a setup.py script to install doped"""

import os
import glob
import subprocess
import sys
import warnings

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))


def readme():
    """
    Set GitHub repo README as package README.
    """
    with open("README.md") as readme_file:
        return readme_file.read()


def pmg_analysis_defects_warning():
    """
    Print warning message if pymatgen-analysis-defects is installed.
    """
    installed_packages = str(subprocess.check_output([sys.executable, "-m", "pip", "list"]))
    if "pymatgen-analysis-defects" in installed_packages:
        print("Test!!")
        warnings.warn("pymatgen-analysis-defects is installed, which will cause incompatibility issues with doped. "
                      "Please uninstall pymatgen-analysis-defects with 'pip uninstall pymatgen-analysis-defects'.")


class PostInstallCommand(install):
    """Post-installation for installation mode.

    Subclass of the setup tools install class in order to run custom commands
    after installation. Note that this only works when using 'python setup.py install'
    but not 'pip install .' or 'pip install -e .'.
    """

    def run(self):
        """
        Performs the usual install process and then checks if pymatgen-analysis-defects
        is installed. If so, prints a warning message that this needs to be
        uninstalled for doped to work (at present).
        """
        # Perform the usual install process
        install.run(self)
        pmg_analysis_defects_warning()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        """
        Performs the usual install process and then checks if pymatgen-analysis-defects
        is installed. If so, prints a warning message that this needs to be
        uninstalled for doped to work (at present).
        """
        develop.run(self)
        pmg_analysis_defects_warning()


class CustomEggInfoCommand(egg_info):
    """Post-installation"""

    def run(self):
        """
        Performs the usual install process and then checks if pymatgen-analysis-defects
        is installed. If so, prints a warning message that this needs to be
        uninstalled for doped to work (at present).
        """
        egg_info.run(self)
        pmg_analysis_defects_warning()

setup(
    name="doped",
    packages=find_packages(),
    version="0.2.1",
    install_requires=[
        "numpy>=1.20.0",
        "pymatgen<2022.8.23",
        "matplotlib",
        "monty>=3.0.2",
        "tabulate",
        "ase",
        "shakenbreak==22.11.1",  # pymatgen<2022.8.23
    ],
    package_data={"doped.pycdt.utils": ["*.yaml"], "doped": ["default_POTCARs.yaml"]},
    author="Seán Kavanagh",
    author_email="sean.kavanagh.19@ucl.ac.uk",
    maintainer="Seán Kavanagh",
    maintainer_email="sean.kavanagh.19@ucl.ac.uk",
    url="https://github.com/SMTG-UCL/doped",
    description="Collection of Python modules & functions to perform "
    "and process solid-state defect calculations",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="MIT",
    scripts=glob.glob(os.path.join(SETUP_PTH, "scripts", "*")),
    test_suite="nose.collector",
    tests_require=["nose", "pytest"],
    cmdclass={
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
)


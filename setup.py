"""This is a setup.py script to install doped"""

import os
import glob

from setuptools import setup, find_packages

SETUP_PTH = os.path.dirname(os.path.abspath(__file__))


def readme():
    """
    Set GitHub repo README as package README.
    """
    with open("README.md") as readme_file:
        return readme_file.read()


setup(
    name="doped",
    packages=find_packages(),
    version="0.1.9",
    install_requires=[
        "numpy",
        "pymatgen<2022.8.23",
        "matplotlib",
        "monty>=3.0.2",
        "tabulate",
        "ase",
        "shakenbreak",
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
)

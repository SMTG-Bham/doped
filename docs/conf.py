# Configuration file for the Sphinx documentation builder.
# build with ``sphinx-build . _build`` (add ``-b linkcheck`` to check links)

import os
import sys

from recommonmark.transform import AutoStructify

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "doped"
copyright = "2023, Seán R. Kavanagh"
author = "Seán R. Kavanagh"  # https://sam-lab.net

# The full version, including alpha/beta/rc tags
release = "3.1.0"


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # for automatic documentation
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_click",
    "sphinx_design",
    "myst_nb",  # for jupyter notebooks
    "sphinx_copybutton",
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "JOSS", "Dev_ToDo.md", "Future_ToDo.md"]

myst_enable_extensions = [
    "html_admonition",
    "html_image",  # to parse html syntax to insert images
    "dollarmath",  # "amsmath", # to parse Latex-style math
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "renku"  # "sphinx_book_theme"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "Images/doped_logo_inverted.png"
html_title = "doped"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

html_theme_options = {  # renku unsupported options commented out here
    # "repository_url": "https://github.com/SMTG-Bham/doped",
    # "github_repo": "https://github.com/SMTG-Bham/doped",
    "github_button": True,
    "github_user": "SMTG-Bham",  # Username
    # "description": "Python package for setting up, parsing and analysing charged defect supercell calculations",
    # "repository_branch": "develop",
    # "path_to_docs": "docs",
    # "use_repository_button": True,
    # "home_page_in_toc": True,
    # "launch_buttons": {
    #     "binderhub_url": "https://mybinder.org",
    #     "colab_url": "https://colab.research.google.com",
    # },
}

# Adding “Edit Source” links on your Sphinx theme
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "SMTG-Bham",  # Username
    "github_repo": "doped",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "pymatgen": ("http://pymatgen.org/", None),
    "shakenbreak": ("https://shakenbreak.readthedocs.io/en/latest/", None),
}

# -- Options for autodoc -----------------------------------------------------
autoclass_content = "both"

# -- Options for nb extension -----------------------------------------------
nb_execution_mode = "off"
# nb_render_image_options = {"height": "300",}  # Reduce plots size
# myst_render_markdown_format = "gfm"
myst_heading_anchors = 2


def setup(app):
    app.add_config_value("myst_parser_config", {"auto_toc_tree_section": "Contents"}, True)
    app.add_transform(AutoStructify)


# ignore non-consecutive level header warnings, and attempted image editing:
suppress_warnings = ["myst.header", "mystnb.image"]

# -- Global substitutions for external links ------------------------------------
# These substitutions are available in all RST files
rst_prolog = """
.. |ShakeNBreak| replace:: `ShakeNBreak <https://shakenbreak.readthedocs.io>`__
.. |ShakeNBreakDocs| replace:: `ShakeNBreak documentation <https://shakenbreak.readthedocs.io>`__
.. |ShakeNBreakTips| replace:: `ShakeNBreak tips <https://shakenbreak.readthedocs.io/en/latest/Tips.html>`__
.. |DeepWiki| replace:: `DeepWiki <https://deepwiki.com/SMTG-Bham/doped>`__
.. |Guidelines Perspective| replace:: `Guidelines for robust and reproducible point defect simulations in crystals <https://doi.org/10.26434/chemrxiv-2025-3lb5k>`__
"""

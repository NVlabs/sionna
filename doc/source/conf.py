#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('./rt'))


# -- Project information -----------------------------------------------------

project = "Sionna"
copyright = "2021-2025 NVIDIA CORPORATION"

# Read version number from sionna.__init__
from importlib.machinery import SourceFileLoader
release = SourceFileLoader("version",
                           "../../src/sionna/__init__.py").load_module().__version__


# -- General configuration ---------------------------------------------------

# Load custom extensions
sys.path.append(os.path.abspath("./_ext"))

#import sphinx_rtd_theme
extensions = ["sphinx_rtd_theme",
              "sphinx.ext.napoleon",
              "sphinx_autodoc_typehints",
              "sphinx.ext.viewcode",
              "sphinx.ext.mathjax",
              "sphinx_copybutton",
              "nbsphinx",
              "roles",
              "made_with_sionna",
             ]
autodoc_typehints = "description"
typehints_fully_qualified = True
simplify_optional_unions = True
nbsphinx_execute = 'never'
suppress_warnings = ['nbsphinx.localfile']
# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": False,
    "navigation_depth": 5,
    }
html_show_sourcelink = False
pygments_style = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ['css/sionna.css']

napoleon_custom_sections = [("Input shape", "params_style"),
                            ("Output shape", "params_style"),
                            ("Attributes", "params_style"),
                            ("Input", "params_style"),
                            ("Output", "params_style"),
                            ("Keyword Arguments", "params_style"),
                            ]
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_keyword = True
numfig = True

# do not re-execute jupyter notebooks when building the docs
nbsphinx_execute = 'never'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Add a custom prolog to each notebook to automatically add github/colab/download links
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}
{% set fullpath = env.doc2path(env.docname, base='source') %}
{% set path_parts = fullpath.split('/') %}
{% set module_name = path_parts[-3] if path_parts|length > 2 else '' %}
{% set notebook_name = path_parts[-1] if path_parts|length > 1 else '' %}

{# Set the appropriate links based on module name #}
{% if module_name == 'phy' %}
    {% set add_navigation_bar = True %}
    {% set github_link = 'https://github.com/nvlabs/sionna/blob/main/tutorials/phy/' + notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = 'https://colab.research.google.com/github/nvlabs/sionna/blob/main/tutorials/phy/' + notebook_name|string|e %}
{% elif module_name == 'sys' %}
    {% set add_navigation_bar = True %}
    {% set github_link = 'https://github.com/nvlabs/sionna/blob/main/tutorials/sys/' + notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = 'https://colab.research.google.com/github/nvlabs/sionna/blob/main/tutorials/sys/' + notebook_name|string|e %}
{% elif module_name == 'rt' %}
    {% set add_navigation_bar = True %}
    {% set github_link = 'https://github.com/nvlabs/sionna-rt/blob/main/tutorials/' + notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = 'https://colab.research.google.com/github/nvlabs/sionna-rt/blob/main/tutorials/' + notebook_name|string|e %}
{% elif module_name == 'rk' %}
    {% set add_navigation_bar = False %}
{% else %}
    {% set github_link = notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = notebook_name|string|e %}
{% endif %}

{% if add_navigation_bar %}
.. raw:: html

    <style>h3 {display: block !important}</style>
    <div style="margin-bottom:15px;">
        <table>
            <td style="padding: 0px 0px;">
                <a href="{{ colab_link }}" style="vertical-align:text-bottom">
                    <img alt="Colab logo" src="../../_static/colab_logo.svg" style="width: 40px; min-width: 40px">
                </a>
            </td>
            <td style="padding: 4px 0px;">
                <a href="{{ colab_link }}" style="vertical-align:text-bottom">
                    Run in Google Colab
                </a>
            </td>

            <td style="padding: 0px 15px;">
            </td>

            <td class="wy-breadcrumbs-aside" style="padding: 0 30px;">
                <a href="{{ github_link }}" style="vertical-align:text-bottom">
                    <i class="fa fa-github" style="font-size:24px;"></i>
                    View on GitHub
                </a>
            </td>

            <td class="wy-breadcrumbs-aside" style="padding: 0 35px;">
                <a href="{{ download_link|e }}" download target="_blank" style="vertical-align:text-bottom">
                    <i class="fa fa-download" style="font-size:24px;"></i>
                    Download notebook
                </a>

            </td>
        </table>
    </div>
{% endif %}

"""

# Make sure that nbsphinx picks the HTML output rather
# than trying to auto-expose the widgets (too complicated).
import nbsphinx
nbsphinx.DISPLAY_DATA_PRIORITY_HTML = tuple(
    m for m in nbsphinx.DISPLAY_DATA_PRIORITY_HTML
    if not m.startswith('application/')
)
# Avoid duplicate display of widgets, see: https://github.com/spatialaudio/nbsphinx/issues/378#issuecomment-573599835
nbsphinx_widgets_path = ''


#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("./rt"))
sys.path.insert(0, os.path.abspath("."))

# Load the RT stack here so that it is already initialized when Sphinx forks
# its parallel read workers. Importing Mitsuba/Dr.Jit inside a forked child
# segfaults, which aborts parallel builds. Missing or broken installations are
# ignored; autodoc then reports the RT pages as import failures.
try:
    import sionna.rt  # noqa: F401  pylint: disable=unused-import
except Exception:  # pylint: disable=broad-except
    pass


# -- Project information -----------------------------------------------------
project = "Sionna"
copyright = "2021-2026 NVIDIA CORPORATION"

# Read version number from sionna.__init__
from importlib.machinery import SourceFileLoader

import importlib.util

spec = importlib.util.spec_from_file_location("version", "../../src/sionna/__init__.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
release = module.__version__
version = release
html_title = f"{project} {version}"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "_ext.list_registry",
    "sphinxcontrib.bibtex",
    "_ext.input_output_fields",
    "_ext.typing_links",
    "_ext.pytorch_links",
    "_ext.mitsuba_drjit_links",
    "_ext.made_with_sionna",
    "_ext.bibtex_crossdoc_backrefs",
]

# -- sphinxcontrib-bibtex ----------------------------------------------------
from pybtex.plugin import register_plugin
from _ext.bibtex_key_label import KeyLabelStyle, KeyLabelPlainStyle

register_plugin("pybtex.style.labels", "keylabel", KeyLabelStyle)
register_plugin("pybtex.style.formatting", "plain_keylabel", KeyLabelPlainStyle)

bibtex_bibfiles = ["phy/phy.bib", "sys/sys.bib", "rt/rt.bib"]
bibtex_default_style = "plain_keylabel"
bibtex_reference_style = "label"
bibtex_tooltips = True


# -- nbsphinx configuration --------------------------------------------------
nbsphinx_execute = "never"  # Never execute notebooks during build
# Suppress image warnings - nbsphinx stores images in doctrees and paths are
# fixed by Makefile post-processing before moving to versioned folder
suppress_warnings = ["image.not_readable"]

# Exclude intentionally orphaned documents from the build
exclude_patterns = [
    "phy/tutorials/notebooks/Discover_Sionna.ipynb",
]

# -- nbsphinx prolog (GitHub/Colab/Download links per notebook) ---------------
nbsphinx_prolog = r"""
{% set path_parts = env.docname.split('/') %}
{% set module_name = path_parts[0] if path_parts|length > 0 else '' %}
{% set notebook_stem = path_parts[-1] if path_parts|length > 0 else '' %}
{% set notebook_name = notebook_stem + '.ipynb' %}
{% set static_prefix = '../' * (path_parts|length - 1) %}

{% if module_name == 'phy' %}
    {% set add_navigation_bar = True %}
    {% set github_link = 'https://github.com/NVlabs/sionna/blob/v' + env.config.release + '/tutorials/phy/' + notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = 'https://colab.research.google.com/github/NVlabs/sionna/blob/v' + env.config.release + '/tutorials/phy/' + notebook_name|string|e %}
{% elif module_name == 'sys' %}
    {% set add_navigation_bar = True %}
    {% set github_link = 'https://github.com/NVlabs/sionna/blob/v' + env.config.release + '/tutorials/sys/' + notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = 'https://colab.research.google.com/github/NVlabs/sionna/blob/v' + env.config.release + '/tutorials/sys/' + notebook_name|string|e %}
{% elif module_name == 'rt' %}
    {% set add_navigation_bar = True %}
    {% set github_link = 'https://github.com/NVlabs/sionna-rt/blob/v' + env.config.release + '/tutorials/' + notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = 'https://colab.research.google.com/github/NVlabs/sionna-rt/blob/v' + env.config.release + '/tutorials/' + notebook_name|string|e %}
{% else %}
    {% set add_navigation_bar = False %}
    {% set github_link = notebook_name|string|e %}
    {% set download_link = notebook_name|string|e %}
    {% set colab_link = notebook_name|string|e %}
{% endif %}

{% if add_navigation_bar %}
.. raw:: html

    <div style="margin-bottom:15px; display:flex; flex-wrap:wrap; align-items:center; gap:28px; background:transparent;">
        <a href="{{ colab_link }}" style="vertical-align:text-bottom; display:inline-flex; align-items:center; gap:6px; text-decoration:none !important; border-bottom:none !important; color:inherit;">
            <img alt="Colab logo" src="{{ static_prefix }}_static/colab_logo.svg" style="width:18px; height:18px; background:transparent;">
            <span>Run in Google Colab</span>
        </a>
        <a href="{{ github_link }}" style="vertical-align:text-bottom; display:inline-flex; align-items:center; gap:6px; text-decoration:none !important; border-bottom:none !important; color:inherit;">
            <i class="fa-brands fa-github" aria-hidden="true"></i>
            <span>View on GitHub</span>
        </a>
        <a href="{{ download_link|e }}" download target="_blank" style="vertical-align:text-bottom; display:inline-flex; align-items:center; gap:6px; text-decoration:none !important; border-bottom:none !important; color:inherit;">
            <span aria-hidden="true">⬇</span>
            <span>Download notebook</span>
        </a>
    </div>
{% endif %}

"""

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = "groupwise"
autodoc_typehints = "both"
autodoc_typehints_format = "fully-qualified"  # Show full paths like np.random.Generator
# Prefer built-in type names for display; typing_links extension rewrites links to stdtypes
autodoc_type_aliases = {
    "List": "list",
    "Dict": "dict",
}
# Globally exclude inherited Keras/Block plumbing methods from rendered docs.
# `build` is a Keras lifecycle hook, not a user-facing API.
autodoc_default_options = {
    "exclude-members": "build",
}
autosummary_generate = True  # Generate autosummary pages for all modules
autosummary_generate_overwrite = True  # Overwrite existing autosummary pages

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
# Base path for hosting (read from environment, default empty for local/root hosting)
# Set BASE_PATH= (empty) for local/root hosting
base_path = os.environ.get("BASE_PATH", "/sionna")
html_baseurl = f"{base_path}/"  # Base URL for version switcher absolute paths
html_static_path = ["_static"]
templates_path = ["_templates"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
]
html_css_files = ["custom.css"]
numfig = True

# -- MathJax (v4) configuration ----------------------------------------------
# Sphinx 9 defaults to MathJax 4, which scales inline math so that the math
# font's ex-height matches the surrounding text ("matchFontHeight", on by
# default). With the pydata theme's sans-serif font stack this overshoots and
# renders the serif math noticeably larger than the body text. Disabling it
# makes inline math render at its natural size relative to the text.
# The final visual size is then set via CSS (``mjx-container`` in custom.css),
# which is deterministic and not overridden by MathJax's persisted contextual
# menu "Scale All Math" setting.
# (nbsphinx merges the required ``tex``/``options`` entries into this dict.)
mathjax4_config = {
    "chtml": {
        "matchFontHeight": False,
    },
}

html_theme_options = {
    "logo": {
        "text": "Sionna",
    },
    "header_links_before_dropdown": 5,
    "switcher": {
        "json_url": f"{base_path}/versions.json",
        "version_match": version,
    },
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",    
    "check_switcher": False,
    "show_version_warning_banner": True,
    "navbar_center": ["navbar-nav"],
    "navbar_end": [
        "version-switcher",
        "navbar-icon-links",
        "theme-switcher",
    ],
    "search_bar_text": "Search...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVlabs/sionna",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/sionna/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "navigation_with_keys": False,
    "show_nav_level": 1,
    "navigation_depth": 5,
    "show_toc_level": 2,
    "navbar_align": "left",
}

# -- Intersphinx: link to external docs (Python, NumPy, PyTorch, Mitsuba, Dr.Jit) -
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "mitsuba": ("https://mitsuba.readthedocs.io/en/stable", None),
    "drjit": ("https://drjit.readthedocs.io/en/stable", None),
}

# Make sure that nbsphinx picks the HTML output rather
# than trying to auto-expose the widgets (too complicated).
import nbsphinx

nbsphinx.DISPLAY_DATA_PRIORITY_HTML = tuple(
    m for m in nbsphinx.DISPLAY_DATA_PRIORITY_HTML if not m.startswith("application/")
)
# Avoid duplicate display of widgets, see: https://github.com/spatialaudio/nbsphinx/issues/378#issuecomment-573599835
nbsphinx_widgets_path = ""

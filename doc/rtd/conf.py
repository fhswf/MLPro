# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "MLPro Documentations"
copyright = "2025 South Westphalia University of Applied Sciences, Germany"
author = "Detlef Arend, Steve Yuwono, Mochammad Rizky Diprasetya, Laxmikant Shrikant Baheti et al"

# The full version, including alpha/beta/rc tags
release = "2.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx_multitoc_numbering",
    "sphinxcontrib.jquery",
    "ablog",
#    'sphinx.ext.intersphinx',
]
autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = "default"

html_logo = "_static/logo_mlpro.png"
html_favicon = "_static/favicon.ico"


def setup(app):
    app.add_css_file("custom.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# html_js_files = [
#     "jquery-3.6.4.min.js",
# ]

html_context = {
    "display_github": True,
    "github_user": "fhswf",
    "github_repo": "MLPro",
    "github_version": "main/doc/docs/",
}


#
# RSS feed
#
blog_title = 'MLPro News'
blog_baseurl = 'https://mlpro.readthedocs.io'  
blog_path = 'news'
fontawesome_included = True
blog_feed_archives = True
blog_feed_fulltext = True
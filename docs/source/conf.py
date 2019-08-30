# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'SubMAPP'
copyright = '2019, Natacha'
author = 'Natacha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    #'sphinx_automodapi.automodapi',
    'sphinx.ext.autosummary', 
    'sphinx.ext.mathjax',
]



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']




def rst_file_transform(docname):
    if docname == 'index':
        docname = 'home'
    return docname.title() + rst_file_suffix



# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__'
    'inherited-members',

}

mathjax_config = {
    'extensions': ['tex2jax.js'],
    'jax': ['input/TeX', 'output/HTML-CSS'],
}

numpydoc_class_members_toctree = False
autoclass_content = 'class'
autosummary_generate = True
autodoc_member_order = 'bysource'
html_sidebars = {'**': ['localtoc.html', 'sourcelink.html', 'searchbox.html']}

# see https://stackoverflow.com/questions/11697634/have-sphinx-replace-docstring-text/11746519
# and https://sphinx.readthedocs.io/en/1.0/ext/appapi.html
# and https://pypi.org/project/sphinxcontrib-restbuilder/#files

# 
#def get_name(full_module_name):
#    """
#    Pull out the name from the full_module_name
#    """
#    #split the full_module_name by "."'s
#    list_modules = full_module_name.split('.')
#    return list_modules[-1]

#def process_docstring(app, what, name, obj, options, lines):
#    true_name = get_name(name)
#    name = true_name
#    print(obj.title())
#    print(name)

#def setup(app):
#    app.connect('autodoc-process-docstring', process_docstring)



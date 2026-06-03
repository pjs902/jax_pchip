import sys, os
sys.path.insert(0, os.path.abspath(".."))

project = "jax_pchip"
author = "Peter Smith"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme = "furo"
autodoc_member_order = "bysource"

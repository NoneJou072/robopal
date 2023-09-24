# Configuration file for the Sphinx documentation builder.
import os
import sys
project_path = '../../robopal'
sys.path.insert(0, os.path.abspath(project_path))

# ...

apidoc_module_dir = project_path
apidoc_output_dir = 'python_apis'
# apidoc_excluded_paths = ['tests']
apidoc_separate_modules = True
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'robopal'
copyright = '2023, Haoran Zhou'
author = 'Haoran Zhou'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinxcontrib.apidoc',
    'sphinx.ext.viewcode',
    'm2r2'
]
source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'press'
html_static_path = ['_static']

master_doc = 'index'

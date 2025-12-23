# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

# Insert the project root directory into sys.path so that asnumpy can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

MOCK_MODULES = ['asnumpy.lib.asnumpy_core', 'asnumpy.lib.asnumpy_core.array', 'asnumpy.lib.asnumpy_core.cann', 'asnumpy.lib.asnumpy_core.linalg', 'asnumpy.lib.asnumpy_core.logic', 'asnumpy.lib.asnumpy_core.math', 'asnumpy.lib.asnumpy_core.random', 'asnumpy.lib.asnumpy_core.sorting', 'asnumpy.lib.asnumpy_core.utils']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

import asnumpy

try:
    __version__ = asnumpy.__version__
    if isinstance(__version__, Mock):
        __version__ = "0.0.0"
except AttributeError:
    __version__ = "0.0.0"
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
]

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'AsNumpy'
year = datetime.now().year
copyright = u'{}, AISS Group at Harbin Institute of Technology'.format(year)
author = u'AISS Group at Harbin Institute of Technology'

version = __version__
release = __version__

language = 'zh_CN'

exclude_patterns = []

pygments_style = 'sphinx'

todo_include_todos = False

napoleon_use_ivar = True
napoleon_include_special_with_doc = True

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

# -- Options for HTML output ----------------------------------------------

# Using pydata_sphinx_theme as a standard theme
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-repo/asnumpy",
            "icon": "fab fa-github-square",
        },
    ],
}

htmlhelp_basename = 'AsNumpydoc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
}

latex_documents = [
    (master_doc, 'AsNumpy.tex', u'AsNumpy Documentation',
     author, 'manual'),
]

man_pages = [
    (master_doc, 'asnumpy', u'AsNumpy Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'AsNumpy', u'AsNumpy Documentation',
     author, 'AsNumpy', 'One line description of project.',
     'Miscellaneous'),
]

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
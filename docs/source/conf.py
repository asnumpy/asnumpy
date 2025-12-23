# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime, timezone
import asnumpy
# Insert the project root directory into sys.path so that asnumpy can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
__version__ = asnumpy.__version__
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
copyright = u'{}, AsNumpy Contributors'.format(year)
author = u'AsNumpy Contributors'

version = __version__
release = __version__

language = 'zh_CN'

html_search_language = 'zh'

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
html_logo = '../images/asnumpy_logo.png'
html_theme_options = {
    "logo": {
        "text": "AsNumpy",
    },
    "icon_links": [
        {
            "name": "GitCode",
            "url": "https://gitcode.com/cann/asnumpy",
            "icon": "fab fa-git",
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
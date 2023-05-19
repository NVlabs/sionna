#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Custom extension to automate the "Made with Sionna" section

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

class MadeWithSionna(SphinxDirective):
    has_content = True
    required_arguments = 0
    option_spec = {
        'title': directives.unchanged_required,
        'authors': directives.unchanged_required,
        'version': directives.unchanged_required,
        'year': directives.unchanged_required,
        'link_colab': directives.unchanged,
        'link_arxiv': directives.unchanged,
        'link_pdf': directives.unchanged,
        'link_github': directives.unchanged,
        'link_code': directives.unchanged,
        'abstract': directives.unchanged_required,
    }
    def run(self):

        title = self.options.get('title')
        authors = self.options.get('authors')
        version = self.options.get('version')
        year = self.options.get('year')
        abstract = self.options.get('abstract')

        # parse optional arguments
        try:
            link_colab = self.options.get('link_colab')
        except:
            link_colab = None
        else:
            pass
        try:
            link_arxiv = self.options.get('link_arxiv')
        except:
            link_arxiv = None
        else:
            pass
        try:
            link_github = self.options.get('link_github')
        except:
            link_github = None
        else:
            pass
        try:
            link_pdf = self.options.get('link_pdf')
        except:
            link_pdf = None
        else:
            pass
        try:
            link_code = self.options.get('link_code')
        except:
            link_code = None
        else:
            pass

        html_str = f'<embed>' \
            f'<h2  style="margin-bottom:0; font-size:19px;">{title}</h2>' \
            f'<i style="margin-bottom:0; font-size:16px;">{authors}</i>' \
            f'<p style="margin-top: 1; margin-bottom:0; padding-top:0;">' \
            f'Released in {year} and based on Sionna v{version}.</p>' \
            f'<div style="margin-top:2; margin-bottom:10px;">'\
            f'<table>'

        if link_arxiv is not None:
            html_str += f'<td style="padding: 4px 0px;">'\
                f'<a href="{link_arxiv}" style="vertical-align:text-bottom">'\
                f'<img alt="Arxiv logo" src="_static/arxiv_logo.png" ' \
                f'style="width: 40px; min-width: 40px"></a>'\
                f'</td><td style="padding: 4px 4px;">'\
                f'<a href="{link_arxiv}" style="vertical-align:text-bottom">'\
                f'Read on arXiv</a></td>'\

        if link_pdf is not None:
            html_str += f'<td class="wy-breadcrumbs-aside"' \
                f' style="padding: 0 0px;">'\
                f'<a href="{link_pdf}" style="vertical-align:text-bottom">'\
                f'<i class="fa fa-file" style="font-size:24px;"></i>'\
                f' View Paper</a></td>'

        if link_github is not None:
            html_str += f'<td class="wy-breadcrumbs-aside"' \
                f' style="padding: 0 30px;">'\
                f'<a href="{link_github}" style="vertical-align:text-bottom">'\
                f'<i class="fa fa-github" style="font-size:24px;"></i>'\
                f' View on GitHub</a></td>'

        if link_code is not None:
            html_str += f'<td class="wy-breadcrumbs-aside"' \
                f' style="padding: 0 30px;">'\
                f'<a href="{link_code}" style="vertical-align:text-bottom">'\
                f'<i class="fa fa-code" style="font-size:24px;"></i>'\
                f' View Code</a></td>'

        if link_colab is not None:
            html_str += f'<td style="padding: 0px 0px;">'\
                f'<a href="{link_colab}"' \
                f'style="vertical-align:text-bottom">'\
                f'<img alt="Colab logo" src="_static/colab_logo.svg" '\
                f'style="width: 40px; min-width: 40px"></a></td>'\
                f'<td style="padding: 4px 0px;">'\
                f'<a href="{link_colab}" style="vertical-align:text-bottom">'\
                f'Run in Google Colab</a></td>'

        html_str += f'</table></div>'\
                    f'<p>{abstract}</p></embed>'

        node = nodes.raw(text=html_str, format='html')

        return [node]


def setup(app):
    app.add_directive("made-with-sionna", MadeWithSionna)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }



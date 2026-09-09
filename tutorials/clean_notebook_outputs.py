#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Clean simulation progress noise from executed notebook outputs.

Merges consecutive stream outputs (handling \\r carriage returns) and strips
intermediate sim_ber() progress lines so only the final result per SNR point
remains.  Run after ``jupyter nbconvert --execute``.

Usage:
    python clean_notebook_outputs.py NOTEBOOK [NOTEBOOK ...]
"""
import re
import sys

import nbformat
from nbconvert.preprocessors import CoalesceStreamsPreprocessor

_SIM_PROGRESS_RE = re.compile(r"[^\n]*\|iter: \d+/\d+\r?\n?")
_TRUNC_PLACEHOLDER_RE = re.compile(
    r"[^\n]*Simulation progress output truncated[^\n]*\n?"
)


def clean_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    original = nbformat.writes(nb)

    # Merge consecutive stream entries and apply bare-\r collapsing
    pp = CoalesceStreamsPreprocessor(enabled=True)
    nb, _ = pp.preprocess(nb, {})

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") != "stream":
                continue
            text = out.text if isinstance(out.text, str) else "".join(out.text)
            cleaned = _SIM_PROGRESS_RE.sub("", text)
            cleaned = _TRUNC_PLACEHOLDER_RE.sub("", cleaned)
            if cleaned != text:
                out.text = cleaned

    cleaned_notebook = nbformat.writes(nb)
    if cleaned_notebook != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned_notebook)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        clean_notebook(path)

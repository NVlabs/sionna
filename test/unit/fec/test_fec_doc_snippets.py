#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Static checks for FEC API documentation snippets.

codespell is not a project CI dependency; spelling on these pages was
fixed in the scrambling/interleaving doc updates instead.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

DOC_FEC = Path(__file__).resolve().parents[3] / "doc" / "source" / "phy" / "api" / "fec"

_CODE_BLOCK = re.compile(
    r"^\.\. code-block::\s*[Pp]ython\s*\n(?:[ \t]*\n)*((?:[ \t]+.*\n)+)",
    re.MULTILINE,
)


def _python_blocks(rst: str) -> list[str]:
    blocks = []
    for match in _CODE_BLOCK.finditer(rst):
        raw = match.group(1)
        lines = raw.splitlines()
        indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
        indent = min(indents) if indents else 0
        blocks.append("\n".join(line[indent:] for line in lines))
    return blocks


def _list_style_calls(tree: ast.AST) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.List) and len(first.elts) >= 2:
            hits.append(ast.unparse(node))
    return hits


@pytest.mark.parametrize("rst_path", sorted(DOC_FEC.rglob("*.rst")))
def test_fec_rst_python_blocks(rst_path: Path):
    """FEC RST Python snippets must parse and not use TF-era call syntax."""
    text = rst_path.read_text(encoding="utf-8")
    blocks = _python_blocks(text)
    if not blocks:
        pytest.skip("no Python code-blocks")

    rel = rst_path.relative_to(DOC_FEC)
    for i, src in enumerate(blocks):
        if "tf." in src:
            pytest.fail(f"{rel} block {i}: TensorFlow call (`tf.`)")
        try:
            tree = ast.parse(src)
        except SyntaxError as err:
            pytest.fail(f"{rel} block {i}: invalid Python ({err})")
        hits = _list_style_calls(tree)
        if hits:
            pytest.fail(
                f"{rel} block {i}: list-style call {hits[0]!r} "
                "(use keyword arguments)"
            )

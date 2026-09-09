#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Policy tests for validation and warning contracts."""

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "sionna"
CONFIG_ATTRIBUTES = {"device", "precision", "dtype", "cdtype"}


def _source_trees():
    """Yield each production source path and parsed syntax tree."""
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _imported_names(tree, name):
    """Return local aliases used to import ``name`` from the PHY config."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and name == "config":
            for imported in node.names:
                if imported.name.endswith(".config") and imported.asname:
                    aliases.add(imported.asname)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module != "sionna.phy" and not (
            module == "config" or module.endswith(".config")
        ):
            continue
        for imported in node.names:
            if imported.name == name:
                aliases.add(imported.asname or imported.name)
    return aliases


def test_config_module_import_aliases_are_detected():
    """Module-style config imports must not bypass policy scans."""
    tree = ast.parse("import sionna.phy.config as phy_config")

    assert _imported_names(tree, "config") == {"phy_config"}


def test_production_source_has_no_bare_asserts():
    """Shipped validation and invariants must remain active under ``python -O``."""
    violations = []
    for path, tree in _source_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")

    assert violations == []


def test_warning_calls_have_explicit_category():
    """Warnings must carry a category so that users can filter them."""
    violations = []
    for path, tree in _source_trees():
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "warnings"
                and node.func.attr == "warn"
            ):
                continue

            keywords = {keyword.arg for keyword in node.keywords}
            if not (len(node.args) >= 2 or "category" in keywords):
                violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")

    assert violations == []


def test_config_properties_are_not_accessed_on_config_class():
    """Config properties must be read from the singleton, never the class."""
    violations = []
    for path, tree in _source_trees():
        config_class_names = _imported_names(tree, "Config")
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Attribute)
                and node.attr in CONFIG_ATTRIBUTES
                and isinstance(node.value, ast.Name)
                and node.value.id in config_class_names
            ):
                continue
            violations.append(f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}")

    assert violations == []


def test_classes_do_not_read_global_device_or_precision():
    """Classes must use their instance configuration rather than ``config``."""
    violations = []
    for path, tree in _source_trees():
        config_names = _imported_names(tree, "config")
        if not config_names:
            continue

        for class_node in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ):
            # Object is the configuration boundary where global defaults are
            # resolved when callers do not provide per-instance values.
            if (
                path.relative_to(SOURCE_ROOT) == Path("phy/object.py")
                and class_node.name == "Object"
            ):
                continue
            for node in ast.walk(class_node):
                if not (
                    isinstance(node, ast.Attribute)
                    and node.attr in CONFIG_ATTRIBUTES
                    and isinstance(node.value, ast.Name)
                    and node.value.id in config_names
                ):
                    continue
                violations.append(
                    f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}"
                )

    assert violations == []


def test_async_tensor_assertions_are_centralized():
    """Private PyTorch assertions must remain isolated in the helper module."""
    violations = []
    for path, tree in _source_trees():
        if path == SOURCE_ROOT / "_validation.py":
            continue
        for node in ast.walk(tree):
            private_assertion = (
                isinstance(node, ast.Attribute)
                and node.attr in ("_assert", "_assert_async")
            )
            imported_assertion = (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch"
                and any(
                    alias.name in ("_assert", "_assert_async")
                    for alias in node.names
                )
            )
            if private_assertion or imported_assertion:
                violations.append(
                    f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}"
                )

    assert violations == []

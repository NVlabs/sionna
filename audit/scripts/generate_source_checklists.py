#!/usr/bin/env python3
"""Generate source and test review checklists for the audit.

The output is intentionally mechanical. It creates one checkbox per file,
top-level function, class, and method so the detailed audit can record coverage
without glossing over implementation units.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "audit" / "checklists"


@dataclass(frozen=True)
class Workstream:
    slug: str
    title: str
    roots: tuple[str, ...]


SOURCE_WORKSTREAMS = [
    Workstream(
        "rt-path-solvers",
        "RT Path Solvers",
        ("ext/sionna-rt/src/sionna/rt/path_solvers/",),
    ),
    Workstream(
        "rt-radio-map-solvers",
        "RT Radio Map Solvers",
        ("ext/sionna-rt/src/sionna/rt/radio_map_solvers/",),
    ),
    Workstream(
        "rt-core-scene-materials-utils",
        "RT Core, Scene, Materials, Geometry, Rendering, Utilities",
        ("ext/sionna-rt/src/sionna/__init__.py", "ext/sionna-rt/src/sionna/rt/"),
    ),
    Workstream(
        "phy-core-mapping-signal-utils",
        "PHY Core, Mapping, Signal, Utilities",
        (
            "src/sionna/__init__.py",
            "src/sionna/phy/__init__.py",
            "src/sionna/phy/block.py",
            "src/sionna/phy/config.py",
            "src/sionna/phy/constants.py",
            "src/sionna/phy/mapping.py",
            "src/sionna/phy/object.py",
            "src/sionna/phy/signal/",
            "src/sionna/phy/utils/",
        ),
    ),
    Workstream(
        "phy-channel",
        "PHY Channel",
        ("src/sionna/phy/channel/",),
    ),
    Workstream(
        "phy-fec",
        "PHY FEC",
        ("src/sionna/phy/fec/",),
    ),
    Workstream(
        "phy-mimo-ofdm-nr",
        "PHY MIMO, OFDM, NR",
        (
            "src/sionna/phy/mimo/",
            "src/sionna/phy/ofdm/",
            "src/sionna/phy/nr/",
        ),
    ),
    Workstream(
        "sys",
        "SYS",
        ("src/sionna/sys/",),
    ),
]

TEST_WORKSTREAMS = [
    Workstream("tests-main", "Main Test Suite", ("test/",)),
    Workstream("tests-rt", "RT Test Suite", ("ext/sionna-rt/test/",)),
]

EXCLUDE_PREFIXES = (
    "ext/sionna-rt/src/sionna/rt/path_solvers/",
    "ext/sionna-rt/src/sionna/rt/radio_map_solvers/",
)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def matches(path: str, roots: tuple[str, ...]) -> bool:
    for root in roots:
        if root.endswith(".py"):
            if path == root:
                return True
        elif path.startswith(root):
            return True
    return False


def source_files_for(workstream: Workstream) -> list[Path]:
    files = sorted(
        p
        for p in (REPO_ROOT / "src").rglob("*.py")
    ) + sorted(
        p
        for p in (REPO_ROOT / "ext" / "sionna-rt" / "src").rglob("*.py")
    )

    selected = []
    for path in files:
        r = rel(path)
        if workstream.slug == "rt-core-scene-materials-utils":
            if r == "ext/sionna-rt/src/sionna/__init__.py":
                selected.append(path)
            elif r.startswith("ext/sionna-rt/src/sionna/rt/") and not any(
                r.startswith(prefix) for prefix in EXCLUDE_PREFIXES
            ):
                selected.append(path)
        elif matches(r, workstream.roots):
            selected.append(path)
    return selected


def test_files_for(workstream: Workstream) -> list[Path]:
    files = sorted(
        p
        for root in workstream.roots
        for p in (REPO_ROOT / root).rglob("*.py")
    )
    return files


def assigned_source_files() -> set[str]:
    assigned: set[str] = set()
    for stream in SOURCE_WORKSTREAMS:
        assigned.update(rel(path) for path in source_files_for(stream))
    return assigned


def parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    except SyntaxError:
        return None


def node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Tuple):
        names = [node_name(elt) for elt in node.elts]
        names = [name for name in names if name]
        return ", ".join(names) if names else None
    return None


def assignment_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        names = [node_name(target) for target in node.targets]
        return [name for name in names if name]
    if isinstance(node, ast.AnnAssign):
        name = node_name(node.target)
        return [name] if name else []
    if isinstance(node, ast.AugAssign):
        name = node_name(node.target)
        return [name] if name else []
    return []


def check_text(name: str, kind: str, line: int) -> str:
    return (
        f"- [ ] {kind} `{name}` (line {line}) reviewed: contract, inputs, "
        "outputs, shape/dtype/device behavior, numerical correctness, error "
        "handling, side effects, docs, and tests."
    )


def render_file(path: Path, include_tests: bool = False) -> tuple[str, dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    line_count = len(text.splitlines())
    tree = parse(path)
    counts = {"files": 1, "classes": 0, "methods": 0, "functions": 0, "tests": 0}
    out = [f"## `{rel(path)}`", "", f"Lines: {line_count}", ""]
    out.append("- [ ] File reviewed line by line, including imports, constants, control flow, numerical formulas, and comments.")
    out.append("- [ ] Module-level API, import side effects, public exports, docs, and tests traced.")

    if tree is None:
        out.append("- [ ] Parse error investigated.")
        return "\n".join(out), counts

    globals_: list[tuple[int, str]] = []
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    classes: list[ast.ClassDef] = []
    for node in tree.body:
        names = assignment_names(node)
        for name in names:
            globals_.append((getattr(node, "lineno", 0), name))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)

    if globals_:
        names = ", ".join(f"`{name}` (line {line})" for line, name in globals_)
        out.append(f"- [ ] Module-level state/constants reviewed: {names}.")

    if functions:
        out.extend(["", "### Top-Level Functions", ""])
        for node in functions:
            kind = "Async function" if isinstance(node, ast.AsyncFunctionDef) else "Function"
            out.append(check_text(node.name, kind, node.lineno))
            counts["functions"] += 1
            if include_tests and node.name.startswith("test_"):
                counts["tests"] += 1

    if classes:
        out.extend(["", "### Classes", ""])
        for cls in classes:
            counts["classes"] += 1
            bases = []
            for base in cls.bases:
                try:
                    bases.append(ast.unparse(base))
                except Exception:
                    bases.append(type(base).__name__)
            base_text = f" Bases: `{', '.join(bases)}`." if bases else ""
            out.append(
                f"- [ ] Class `{cls.name}` (line {cls.lineno}) reviewed: "
                f"invariants, inheritance, state, lifecycle, public API, and docs.{base_text}"
            )

            class_globals: list[tuple[int, str]] = []
            methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
            nested_classes: list[ast.ClassDef] = []
            for child in cls.body:
                for name in assignment_names(child):
                    class_globals.append((getattr(child, "lineno", 0), name))
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(child)
                elif isinstance(child, ast.ClassDef):
                    nested_classes.append(child)

            if class_globals:
                names = ", ".join(
                    f"`{name}` (line {line})" for line, name in class_globals
                )
                out.append(f"  - [ ] Class-level state/constants reviewed: {names}.")

            for method in methods:
                counts["methods"] += 1
                if include_tests and method.name.startswith("test_"):
                    counts["tests"] += 1
                kind = "Async method" if isinstance(method, ast.AsyncFunctionDef) else "Method"
                out.append(
                    "  " + check_text(f"{cls.name}.{method.name}", kind, method.lineno)
                )

            for nested in nested_classes:
                counts["classes"] += 1
                out.append(
                    f"  - [ ] Nested class `{cls.name}.{nested.name}` "
                    f"(line {nested.lineno}) reviewed."
                )

    out.append("")
    out.append("- [ ] Tests covering this file reviewed or explicitly marked as missing.")
    out.append("- [ ] Documentation and examples covering this file reviewed or explicitly marked as missing.")
    return "\n".join(out), counts


def render_workstream(stream: Workstream, files: list[Path], include_tests: bool = False) -> dict[str, int]:
    totals = {"files": 0, "classes": 0, "methods": 0, "functions": 0, "tests": 0}
    sections = []
    for path in files:
        section, counts = render_file(path, include_tests=include_tests)
        sections.append(section)
        for key, value in counts.items():
            totals[key] += value

    out = [
        f"# Checklist: {stream.title}",
        "",
        "Generated by `audit/scripts/generate_source_checklists.py`.",
        "",
        "A checkbox means the item has been technically reviewed, not merely found.",
        "For source workstreams, every file must be inspected line by line before the file checkbox is marked.",
        "",
        "## Totals",
        "",
        f"- Files: {totals['files']}",
        f"- Classes: {totals['classes']}",
        f"- Top-level functions: {totals['functions']}",
        f"- Methods: {totals['methods']}",
    ]
    if include_tests:
        out.append(f"- Test functions/methods: {totals['tests']}")
    out.extend(["", "## Review Items", ""])
    out.extend(sections)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{stream.slug}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    return totals


def write_index(source_totals: dict[str, dict[str, int]], test_totals: dict[str, dict[str, int]], unassigned: list[str]) -> None:
    lines = [
        "# Audit Checklists",
        "",
        "These checklists are generated from the Python AST and are intended to force the audit to proceed function by function, class by class, and file by file.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python3 audit/scripts/generate_source_checklists.py",
        "```",
        "",
        "## Source Workstreams",
        "",
        "| Checklist | Files | Classes | Functions | Methods |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stream in SOURCE_WORKSTREAMS:
        totals = source_totals[stream.slug]
        lines.append(
            f"| [{stream.title}]({stream.slug}.md) | {totals['files']} | "
            f"{totals['classes']} | {totals['functions']} | {totals['methods']} |"
        )

    lines.extend([
        "",
        "## Test Workstreams",
        "",
        "| Checklist | Files | Classes | Functions | Methods | Test items |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for stream in TEST_WORKSTREAMS:
        totals = test_totals[stream.slug]
        lines.append(
            f"| [{stream.title}]({stream.slug}.md) | {totals['files']} | "
            f"{totals['classes']} | {totals['functions']} | {totals['methods']} | "
            f"{totals['tests']} |"
        )

    lines.extend(["", "## Coverage Guard", ""])
    if unassigned:
        lines.append("Unassigned source files that need classification:")
        lines.append("")
        for path in unassigned:
            lines.append(f"- [ ] `{path}`")
    else:
        lines.append("All source files under `src/sionna` and `ext/sionna-rt/src/sionna` are assigned to a checklist.")

    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_totals = {}
    for stream in SOURCE_WORKSTREAMS:
        files = source_files_for(stream)
        source_totals[stream.slug] = render_workstream(stream, files)

    test_totals = {}
    for stream in TEST_WORKSTREAMS:
        files = test_files_for(stream)
        test_totals[stream.slug] = render_workstream(stream, files, include_tests=True)

    all_source = sorted(
        rel(path)
        for root in (REPO_ROOT / "src" / "sionna", REPO_ROOT / "ext" / "sionna-rt" / "src" / "sionna")
        for path in root.rglob("*.py")
    )
    unassigned = [path for path in all_source if path not in assigned_source_files()]
    write_index(source_totals, test_totals, unassigned)


if __name__ == "__main__":
    main()

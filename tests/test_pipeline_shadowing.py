"""Regression: `run_pipeline` must not shadow any imported top-level module
name with a local variable. A single ``structure = ...`` assignment inside
the function turns Python's `structure` reference at other lines into an
UnboundLocalError, because Python decides scope at compile time.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _load_pipeline_ast() -> ast.AST:
    src = Path(__file__).resolve().parents[1] / "paaf" / "pipeline.py"
    return ast.parse(src.read_text())


def _module_imports(tree: ast.AST) -> set[str]:
    """Collect the local-name of every submodule imported at the top level."""
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _local_assigns(func: ast.FunctionDef) -> set[str]:
    """Every name that appears as an assignment target inside `func` body."""
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_run_pipeline_does_not_shadow_module_imports():
    tree = _load_pipeline_ast()
    imports = _module_imports(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline":
            locals_ = _local_assigns(node)
            overlap = imports & locals_
            assert not overlap, (
                f"run_pipeline() reassigns names that are imported at module "
                f"scope: {sorted(overlap)}. Rename these locals to avoid "
                f"UnboundLocalError."
            )
            return
    raise AssertionError("run_pipeline function not found in pipeline.py")

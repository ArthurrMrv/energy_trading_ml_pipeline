"""Import an uploaded strategy file and find the Strategy subclass in it."""

import importlib.util
import inspect
import sys
from pathlib import Path

from backend.pipeline.contract import Strategy


class StrategyLoadError(ValueError):
    """The uploaded file could not be turned into a usable Strategy."""


def load_module(path: str | Path):
    path = Path(path).resolve()
    if not path.is_file():
        raise StrategyLoadError(f"strategy file not found: {path}")

    name = f"uploaded_strategy_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"cannot import {path} as a Python module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise StrategyLoadError(f"{path.name} failed to import: {exc!r}") from exc
    return module


def find_strategy_class(module) -> type[Strategy]:
    candidates = [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Strategy) and obj is not Strategy
        and obj.__module__ == module.__name__
    ]
    if not candidates:
        raise StrategyLoadError(
            "no Strategy subclass found. The uploaded file must define a class "
            "inheriting from backend.pipeline.contract.Strategy."
        )
    if len(candidates) > 1:
        names = sorted(c.__name__ for c in candidates)
        raise StrategyLoadError(
            f"found {len(candidates)} Strategy subclasses ({names}); "
            "the file must define exactly one."
        )
    cls = candidates[0]
    if inspect.isabstract(cls):
        missing = sorted(cls.__abstractmethods__)
        raise StrategyLoadError(
            f"{cls.__name__} is abstract; it still needs to implement {missing}"
        )
    return cls


def load_strategy(path: str | Path) -> Strategy:
    return find_strategy_class(load_module(path))()


# --- static inspection ------------------------------------------------------
# Everything below reads an uploaded file with `ast` instead of importing it.
# Only the stage subprocess ever executes strategy code; the API process must
# be able to describe a file it has decided not to run.

import ast  # noqa: E402  (kept beside the functions that use it)


def _strategy_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(b, ast.Name) and b.id == "Strategy")
            or (isinstance(b, ast.Attribute) and b.attr == "Strategy")
            for b in node.bases
        )
    ]


def validate_source(source: str) -> str:
    """Check an upload defines exactly one Strategy subclass. Returns its name."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise StrategyLoadError(f"file is not valid Python: {exc}") from exc

    classes = _strategy_classes(tree)
    if not classes:
        raise StrategyLoadError(
            "no Strategy subclass found. The file must define a class inheriting "
            "from backend.pipeline.contract.Strategy."
        )
    if len(classes) > 1:
        raise StrategyLoadError(
            f"found {len(classes)} Strategy subclasses "
            f"({sorted(c.name for c in classes)}); the file must define exactly one."
        )

    defined = {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    if missing := sorted({"collect", "prepare", "on_tick"} - defined):
        raise StrategyLoadError(
            f"{classes[0].name} does not implement {missing}. "
            "fit() is optional; the others are required."
        )
    return classes[0].name


def method_source(source: str, method: str) -> str | None:
    """Pull one method's source text out of an upload, without importing it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for cls in _strategy_classes(tree):
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == method:
                return ast.get_source_segment(source, node)
    return None

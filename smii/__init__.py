"""Compatibility layer so ``python -m smii`` works from a source checkout."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

__all__ = []  # Populated from the canonical package below.


def _load_canonical_package() -> ModuleType:
    repo_root = Path(__file__).resolve().parent.parent
    package_root = repo_root / "src" / "smii"
    spec = spec_from_file_location(
        __name__,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to locate src/smii package")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rebind(module: ModuleType) -> None:
    current_globals = globals()
    # Ensure submodule lookups hit the actual package path.
    current_globals["__path__"] = list(getattr(module, "__path__", []))
    current_globals["__file__"] = getattr(module, "__file__", None)
    current_globals["__spec__"] = getattr(module, "__spec__", None)
    current_globals["__doc__"] = getattr(module, "__doc__", None)
    current_globals["__all__"] = getattr(module, "__all__", [])

    for name, value in module.__dict__.items():
        if name in {"__dict__", "__weakref__"}:
            continue
        current_globals[name] = value


_module = _load_canonical_package()
_rebind(_module)

# Avoid leaking the helpers.
del _load_canonical_package
_del_names: list[str] = ["_rebind", "_module"]
for _name in _del_names:
    globals().pop(_name, None)
del _del_names

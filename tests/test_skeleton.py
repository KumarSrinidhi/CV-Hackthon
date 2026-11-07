import importlib
import pkgutil

import ps4


def test_package_importable():
    assert ps4 is not None


def test_all_submodules_import():
    modules = [m.name for m in pkgutil.walk_packages(ps4.__path__, prefix="ps4.")]
    assert modules, "expected ps4 submodules to exist"
    for name in modules:
        assert importlib.import_module(name) is not None

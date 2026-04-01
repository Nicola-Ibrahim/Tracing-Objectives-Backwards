import importlib
import pkgutil


def discover_modules(package_name: str) -> list[str]:
    """
    Recursively discovers all dot-separated module paths within a package.
    """
    package = importlib.import_module(package_name)
    package_path = package.__path__

    modules = []
    # Walk through all submodules in the package recursively
    for _, name, is_pkg in pkgutil.walk_packages(package_path, f"{package_name}."):
        # We target non-package modules for wiring (e.g., actual Python files)
        if not is_pkg:
            modules.append(name)
    return modules

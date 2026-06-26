"""Minimal pytest-free runner: discovers test_* functions in tests/test_*.py.

Runs from project root so `from src...` imports resolve. Avoids importing
modules that need unavailable third-party deps (pyarrow/pydantic/tinkoff).
"""
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TESTS_DIR = ROOT / "tests"


def _load_module(path):
    name = "tests_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    files = sorted(p for p in TESTS_DIR.glob("test_*.py"))
    total = 0
    failures = []
    for path in files:
        try:
            mod = _load_module(path)
        except Exception:
            failures.append((path.name, "<import>", traceback.format_exc()))
            print("IMPORT FAIL %s" % path.name)
            continue
        funcs = sorted(
            n for n in dir(mod)
            if n.startswith("test_") and callable(getattr(mod, n))
        )
        for fn in funcs:
            total += 1
            try:
                getattr(mod, fn)()
                print("  ok   %s::%s" % (path.name, fn))
            except Exception:
                failures.append((path.name, fn, traceback.format_exc()))
                print("  FAIL %s::%s" % (path.name, fn))
    print("\n%d run, %d failed" % (total, len(failures)))
    for name, fn, tb in failures:
        print("\n=== %s::%s ===\n%s" % (name, fn, tb))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

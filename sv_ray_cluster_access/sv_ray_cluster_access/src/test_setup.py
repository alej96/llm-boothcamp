#!/usr/bin/env python3
"""Test script to verify environment variables and project configuration."""

import os
import sys
from pathlib import Path


def _split_pythonpath(pythonpath: str) -> list[str]:
    if not pythonpath:
        return []
    return [p for p in pythonpath.split(os.pathsep) if p]


def main():
    print("SupportVectors Environment Setup Test")
    print("=" * 50)

    cwd = Path.cwd().resolve()
    src_path = (cwd / "src").resolve()

    print(f"Python version: {sys.version}")
    print(f"Working directory: {cwd}")

    bootcamp_root = os.environ.get("BOOTCAMP_ROOT_DIR")
    project_python = os.environ.get("PROJECT_PYTHON")
    pythonpath_raw = os.environ.get("PYTHONPATH")
    pythonpath_entries = _split_pythonpath(pythonpath_raw or "")

    print(f"BOOTCAMP_ROOT_DIR: {bootcamp_root or 'Not set'}")
    print(f"PYTHONPATH: {pythonpath_raw or 'Not set'}")
    print(f"PROJECT_PYTHON: {project_python or 'Not set'}")

    issues = []

    if not bootcamp_root:
        issues.append("BOOTCAMP_ROOT_DIR is not set")
    elif Path(bootcamp_root).resolve() != cwd:
        issues.append(
            f"BOOTCAMP_ROOT_DIR points to {Path(bootcamp_root).resolve()}, expected {cwd}"
        )

    src_in_pythonpath = any(Path(p).resolve() == src_path for p in pythonpath_entries)
    if not src_in_pythonpath:
        issues.append(f"PYTHONPATH does not include {src_path}")

    if project_python:
        if os.path.exists(project_python):
            print(f"OK: PROJECT_PYTHON executable found at {project_python}")
        else:
            issues.append(f"PROJECT_PYTHON path does not exist: {project_python}")

    try:
        if src_path.exists():
            module_dirs = [
                d for d in src_path.iterdir() if d.is_dir() and not d.name.startswith(".")
            ]
            if module_dirs:
                module_name = module_dirs[0].name
                print(f"OK: Found module: {module_name}")
                sys.path.insert(0, str(src_path))
                module = __import__(module_name)
                print(f"OK: Successfully imported {module_name}")
                if hasattr(module, "config"):
                    print("OK: Configuration object found and accessible")
                else:
                    print("INFO: Configuration object not yet accessible")
            else:
                issues.append("No module directories found in src/")
        else:
            issues.append(f"src directory not found at {src_path}")
    except ImportError as exc:
        issues.append(f"Could not import module: {exc}")
        if "svlearn" in str(exc):
            issues.append("Dependencies are not installed (missing svlearn). Run: uv sync")
    except Exception as exc:
        issues.append(f"Unexpected setup error: {exc}")

    print("=" * 50)
    if issues:
        print("Environment check found issues:")
        for issue in issues:
            print(f"- {issue}")
        print("")
        print("Suggested fix (run from project root):")
        print('export BOOTCAMP_ROOT_DIR="$(pwd)"')
        print('export PYTHONPATH="$BOOTCAMP_ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"')
        print('export PROJECT_PYTHON="$(which python3)"')
    else:
        print("Environment setup looks good.")


if __name__ == "__main__":
    main()

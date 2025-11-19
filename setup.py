"""Fallback script to setup the package using setuptools by reading metadata from pyproject.toml."""

from pathlib import Path

import toml
from setuptools import setup


def load_pyproject_metadata():
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    with pyproject_path.open("r") as f:
        pyproject_data = toml.load(f)
    project = pyproject_data.get("project", {})
    return {
        "name": project.get("name", "UNKNOWN"),
        "version": project.get("version", "0.0.0"),
        "description": project.get("description", ""),
        "author": ", ".join(a.get("name", "") for a in project.get("authors", [])),
        "author_email": ", ".join(
            a.get("email", "") for a in project.get("authors", [])
        ),
        "license": project.get("license", {}).get("file", ""),
        "keywords": project.get("keywords", []),
        "classifiers": project.get("classifiers", []),
        "install_requires": project.get("dependencies", []),
        "entry_points": {
            "console_scripts": [
                f"{k}={v}" for k, v in project.get("scripts", {}).items()
            ]
        },
    }


# Load metadata and pass it to setup()
metadata = load_pyproject_metadata()

setup(
    packages=["vw_explorer"],
    **metadata,
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SYMBIONT Framework Setup
========================

SYMBiotic Infrastructure mONitoring with sTructure-mediated Intelligence

A comprehensive framework for structural health monitoring using:
- Structure-Mediated Implicit Synchronization (SMIS)
- Edge-Computed Transmissibility Intelligence (ECTI)
- Energy-Correlated Collaborative Protocol (ECCP)
- Transmissibility-Embedding Federated Learning (TEFL)

Based on PhD research proposal by H M Shujaat Zaheer.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.0.0",
]

setup(
    name="symbiont",
    version="1.0.0",
    author="H M Shujaat Zaheer",
    author_email="hmshujaatzaheer@gmail.com",
    description="SYMBiotic Infrastructure mONitoring with sTructure-mediated Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shujaatmirza/symbiont",
    project_urls={
        "Documentation": "https://symbiont.readthedocs.io",
        "Bug Tracker": "https://github.com/shujaatmirza/symbiont/issues",
        "Source Code": "https://github.com/shujaatmirza/symbiont",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Typing :: Typed",
    ],
    keywords=[
        "structural health monitoring",
        "SHM",
        "federated learning",
        "edge computing",
        "transmissibility",
        "differential privacy",
        "wireless sensor networks",
        "civil engineering",
        "infrastructure monitoring",
        "machine learning",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "symbiont=symbiont.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

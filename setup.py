"""Compatibility shim for older pip versions lacking PEP 660 editable builds."""

from setuptools import find_packages, setup


setup(
    name="uav-env",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)

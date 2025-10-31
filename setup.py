"""
Setup configuration for Biocontrol Modeling Project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="biocontrol-modeling",
    version="1.0.0",
    author="César Augusto García Echeverry",
    author_email="cesar.garech@gmail.com",
    description="An interactive Streamlit application for teaching modeling, simulation, analysis, and control of bioprocesses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CesarGarech/Biocontrol_modeling_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Creative Commons License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "biocontrol-dashboard=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.xlsx", "*.png", "*.jpg", "*.pdf"],
    },
)

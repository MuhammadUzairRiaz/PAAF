from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="paaf",
    version="0.1.0",
    description="PAAF — Polymer Auto-Assembly Framework. Automated polymer "
                "builder, reactive-MD engine, and Moltemplate / DL_FIELD / "
                "LAMMPS / GROMACS system generator.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Uzair Dogar",
    packages=find_packages(),
    include_package_data=True,
    package_data={"paaf": ["data/*.csv", "gui/assets/*.png",
                           "gui/assets/*.ico"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pyyaml>=6.0",
    ],
    extras_require={
        "gui": ["PyQt5>=5.15"],
        "chem": ["openbabel-wheel", "mbuild"],
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "paaf=paaf.cli:main",
            # Kept for backwards-compat with earlier tutorials.
            "moltemplate-auto=paaf.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Operating System :: OS Independent",
    ],
)

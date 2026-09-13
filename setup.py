from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="paaf",
    version="0.1.0",
    description="PAAF — Polymer Auto-Assembly Framework. Builds polymer "
                "chains, amorphous cells, blends and layered systems, types "
                "them with Moltemplate or DL_FIELD, and writes LAMMPS / "
                "GROMACS inputs.",
    url="https://github.com/MuhammadUzairRiaz/PAAF",
    keywords="polymer amorphous-cell molecular-dynamics lammps gromacs "
             "moltemplate dl_field packmol force-field",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Muhammad Uzair Riaz",
    packages=find_packages(),
    include_package_data=True,
    package_data={"paaf": ["data/*.csv", "gui/assets/*.png",
                           "gui/assets/*.ico"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pyyaml>=6.0",
        "scipy>=1.7",
        "networkx>=2.6",
        "rdkit>=2022.9",
    ],
    extras_require={
        # GUI: pip install .[gui]
        "gui": ["PyQt5>=5.15", "PyQtWebEngine>=5.15"],
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

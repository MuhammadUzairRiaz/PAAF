# PAAF — Polymer Auto-Assembly Framework

A desktop tool (PyQt5 GUI) that takes you from a monomer SMILES to a
simulation-ready LAMMPS or GROMACS system, end to end:

- **Pipeline (Builder → Chain → Optimize → Force field → Box → Export):**
  build a monomer (SMILES, periodic table, or an uploaded XYZ/PDB/MOL2/SDF
  file), polymerise it to a chain, optimise, assign a force field
  (OPLS 2005 / PCFF / COMPASS / CVFF / CHARMM36 via DL_FIELD, or
  OPLS-AA / GAFF via moltemplate), pack a box, export.
- **Amorphous cell:** grows chains bond-by-bond into a periodic box
  (Theodorou–Suter growth with RIS torsion statistics), back-maps to all
  atoms, resolves overlaps with a soft push-off, and types the finished
  cell with DL_FIELD. Outputs `cell.data` + `cell.in` (LAMMPS) and/or
  `cell.gro` + `cell.top` (GROMACS).
- **Blend:** packs several pre-built, typed components into one box,
  merging their force fields safely (type offsets, refused conflicts).
- **Layering:** like Blend, but each component is confined to its own
  sub-box — stacked films with per-layer sizes, gap, and pinned origins.
- **Reaction scheme (inside Builder):** define reactions with atom-mapped
  SMILES (53 built-in worked examples), export typed reactant/product
  structures and learned reaction templates.

---

## 1. Installation

### 1.1 Python environment (conda recommended)

```bash
conda create -n paaf python=3.11
conda activate paaf
conda install -c conda-forge rdkit openbabel mbuild numpy scipy networkx pyyaml
pip install PyQt5 pytest
```

Why conda for RDKit/OpenBabel/mBuild: their pip wheels are missing or
broken on several platforms. Everything else installs fine with pip
(`pip install -r requirements.txt`).

| Dependency | Needed for | Required? | Link |
|---|---|---|---|
| numpy, scipy, networkx | everywhere / contacts / graphs | yes | pip/conda |
| PyQt5 | the GUI | yes | pypi.org/project/PyQt5 |
| RDKit | back-mapping, reaction scheme, SMILES | yes | https://www.rdkit.org (conda-forge) |
| mBuild | chain building from monomers | yes | https://mbuild.mosdef.org (conda-forge) |
| OpenBabel | mol2 export, extra formats, UFF fallback | recommended | https://openbabel.org (conda-forge) |
| pyyaml | settings | yes | pip/conda |

### 1.2 External programs (not Python)

Minimum useful setup for testing: **DL_FIELD + packmol + LAMMPS**.

#### DL_FIELD 4.x — force-field typing (required for OPLS 2005 / PCFF / COMPASS / CVFF / CHARMM)
- Official page: https://www.ccp5.ac.uk/DL_FIELD
- Registration (free academic licence, you receive the download):
  https://www.ccp5.ac.uk/dl_field-registration
- Manual: https://www.ccp5.ac.uk/wp-content/uploads/2026/06/dl_field_4_13_manual.pdf
- Install: unpack anywhere **without spaces in the path** (e.g.
  `~/dl_f_4.13/`), compile per its README (`make`), then point PAAF at
  `~/dl_f_4.13/lib` (see §2).

#### packmol — molecule packing (required for Box / Blend / Layering)
- Home & download: https://m3g.github.io/packmol/
  (source: https://github.com/m3g/packmol/releases)
- macOS: `brew install packmol` · conda: `conda install -c conda-forge packmol`
- Check: `which packmol`

#### LAMMPS — MD engine (required for push-off/relaxation and to run the outputs)
- Download: https://www.lammps.org/download.html
- Install docs: https://docs.lammps.org/Install.html
- macOS: `brew install lammps` · conda: `conda install -c conda-forge lammps`
- Check: `lmp -h` (some builds name it `lmp_serial` or `lmp_mpi` — any works,
  set the name on PAAF's Relax step)

#### GROMACS — optional, for the .gro/.top route
- Home: https://www.gromacs.org · install guide:
  https://manual.gromacs.org/current/install-guide/index.html
- macOS: `brew install gromacs` · conda: `conda install -c conda-forge gromacs`
- Check: `gmx --version`

#### moltemplate — optional, only for the moltemplate force-field route (OPLS-AA/GAFF)
- Home: https://www.moltemplate.org
  (source: https://github.com/jewettaij/moltemplate)
- Install: `pip install moltemplate` or `conda install -c conda-forge moltemplate`
- Check: `moltemplate.sh --help`

#### Conda itself (if you don't have it)
- Miniconda: https://docs.conda.io/en/latest/miniconda.html

### 1.3 Launch

```bash
cd PAAF
python run_paaf.py
```

---

## 2. Paths to set inside the tool (first run)

1. **DL_FIELD lib dir** — on the pipeline's *Force field* page, set it to
   the `lib` folder inside your DL_FIELD installation, e.g.
   `/home/you/dl_f_4.13/lib`. Every other page (Amorphous cell, Reaction
   scheme) inherits this automatically. PAAF accepts either the `lib`
   folder or its parent — it finds the `dl_field` executable one level up.
   *The DL_FIELD installation path must not contain spaces.*
2. **LAMMPS executable** — Amorphous cell → *Relax* step (`lmp`,
   `lmp_serial`, or a full path). Auto-detected if it is on PATH.
3. **GROMACS executable** — same page, only if you use the GROMACS route.
4. **packmol** — just needs to be on PATH (`which packmol` to check).
5. **Project output folder** — click the project name (top of the
   sidebar) to choose where all results are written. Everything lands
   under `<output>/<project>/…`, one folder per artefact.

---

## 3. Quick start (5 minutes)

1. `python run_paaf.py`
2. **Builder:** pick PE (`[*]CC[*]`) from the library.
3. **Tools → Amorphous cell:** add PE, DP 50, 2 chains ("Enter chain
   counts myself"), target density 0.95, Build at 45 %.
4. Force-field step: OPLS 2005 (DL_FIELD), check the lib dir is filled.
5. **Build cell.** Watch the log: growth → back-map → push-off →
   DL_FIELD typing. Results in `<output>/<project>/cells/cell/`:
   `cell.data` + `cell.in` — run with `lmp -in cell.in`.
6. The cell is built loose (45 % of target density): minimise, then a
   short NPT run to compress to the target before measuring anything.

## 4. Running the tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

~700 tests; those needing RDKit/PyQt5 skip automatically when missing.

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "dl_field executable not found" | DL_FIELD lib dir not set (see §2.1), or the binary isn't at `<dir>/../dl_field`. |
| "User configuration file format not recognise" | A path with spaces reached DL_FIELD — keep the DL_FIELD install path space-free; PAAF already guards its own paths. |
| Typing fails on "unsaturated C" | Structure genuinely broken (check the 3D preview), or an old bug if your copy predates the box-passing fix — update. |
| Blend/Layering: "packmol not found" | Install packmol and put it on PATH. |
| GUI shows no dropdown arrows / stale behaviour | Fully quit and reopen — Python caches modules per process. |

## 6. Repository layout

```
paaf/            the package (pipeline, cell/, gui/, reaction engine)
paaf/cell/       amorphous cell: grow, backmap, push-off, export, relax
paaf/gui/        PyQt5 pages, one module per tool
ff_libraries/    bundled force-field data
tests/           pytest suite
docs/            documentation and guides
examples/        example inputs
run_paaf.py      GUI launcher
```

## Author

Uzair Dogar — PAAF v0.1.0. Issues and feedback via GitHub Issues.

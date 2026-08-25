# Building an amorphous cell in PAAF

A step-by-step guide to the **Amorphous cell** tool: from a SMILES to a
relaxed, force-field-typed LAMMPS cell.

```
conda activate mta
cd /Users/uzair/project/PAAF
python run_gui.py
```

Then choose **Amorphous cell** in the sidebar, under TOOLS. It is an
independent branch — it does not use the main pipeline's Chain, Force field or
Export pages, so you can have a cell in OPLS-AA and a reference structure in
PCFF at the same time.

The tool has four steps. You can move between them freely; nothing is
committed until you press **Build cell**.

---

## Step 1 · Composition

**What goes in the box, how much of each, and how big the box is.**

### Choose how you specify the blend

| Mode | Use it when |
|---|---|
| **Solve chain counts from weight %** | You have a formulation: "70 wt% PE, 30 wt% PBS" |
| **Enter chain counts myself** | You know the cell you want and don't want anything rounded |

### Add your polymers

Each row is one polymer. For each:

- **Name** — anything; it labels the chains in the output.
- **SMILES** — press **Library…** to pick from 124 polymers, or paste a
  polymerisation SMILES yourself. It must have exactly two `[*]` markers
  showing where the chain continues: `[*]CC[*]` is polyethylene.
- **DP** — repeat units per chain.
- **wt%** or **chains** — depending on the mode.

Two read-outs appear on the right of each row as you type:

- `28.05 g/mol · 2 skel` — the repeat-unit mass, and how many **skeletal
  (backbone) atoms** one unit contributes. PE has 2, PBS has 10. This sets the
  chain's contour length, because growth places one bead per skeletal atom.
- `18 ch · 70.8 wt%` — how many chains this polymer actually gets, and the
  weight percent those **whole** chains come to.

> **Weight percentages stay linked.** Raise one and the others fall in
> proportion, always summing to 100. With three or more components, the ones
> you are not editing keep their ratio to each other. You cannot type an exact
> three-way split left-to-right, because each entry rescales the previous
> ones — set the first two, then let the third take the remainder.

> **`beads only` instead of a mass?** That polymer's backbone runs *through* a
> ring — PET, polycarbonate, PBAT and 21 others in the library. The cell will
> still be grown and exported coarse-grained, but it cannot be rebuilt as
> all-atom, so no force field can be applied. See *Limitations* below.

### Cell targets

| Field | Meaning | Typical |
|---|---|---|
| **Target density** | The box edge is computed to hit this exactly | PE 0.85, PS 1.04, blend in between |
| **Cell size** | Roughly how many skeletal atoms the whole cell holds. Governs run time, and how finely the composition can be expressed | 2 000–10 000 to start |
| **Temperature** | Sets the trans/gauche balance through the RIS weights, so it changes how extended the chains are | 413 K for a melt |

All three are spin boxes — type into them or use the arrows.

### Growth

Leave these alone unless a build fails.

- **Overlap tolerance** (default 1.70 Å) — the hard minimum approach between
  non-bonded beads. It is *not* the physical contact distance; the soft bias
  already handles that. Raising it stops dense cells building at all.
- **Look-ahead depth** (default 0) — leave at 0. It defeats attrition but
  biases chain dimensions, and the tool warns you when it is on.
- **Random seed** — same seed, same cell, exactly.
- **Reject bonds that thread through a ring** — leave ticked.

### Check "What will be built"

The preview table shows **wt% asked** next to **wt% realised**. They differ
because chain counts are whole numbers. If the gap matters, raise Cell size —
weight fractions can only be as fine as one chain's mass.

Also check the box edge. **It must be more than twice the longest repeat
unit**, or minimum-image breaks down and bond lengths measured across the
boundary become meaningless. A polymer with a long side chain needs a bigger
cell than polyethylene does. PAAF warns if you are under.

---

## Step 2 · Force field

**How the cell gets typed.**

Pick a force field. The line underneath tells you which route it implies:

| Route | Force fields | What happens |
|---|---|---|
| **Moltemplate → LAMMPS** | OPLS-AA, L-OPLS, TraPPE-UA, DREIDING, COMPASS (published), MARTINI, SDK | PAAF types the atoms, writes `.lt` files, runs `moltemplate.sh` |
| **DL_FIELD → LAMMPS or GROMACS** | PCFF, COMPASS, CVFF, OPLS-2005, CHARMM36, AMBER/GAFF, GROMOS-54A7, TraPPE-EH, OPLS-UA | DL_FIELD types and writes the topology |

DL_FIELD force fields need the **DL_FIELD lib dir** set.

### Atoms from beads

Growth builds the backbone one skeletal atom at a time. Before any force field
can be applied, the side groups and hydrogens have to be put back.

- **Rebuild all-atom structure** — leave ticked. Unticking gives
  coarse-grained output only, and no force field.
- **Tacticity** — `isotactic` (default), `syndiotactic`, or `atactic`. Real
  bulk vinyl polymers are usually **atactic**; change this if it matters to
  you, because it is a real stereochemical choice.
- **Push overlapping side groups apart** — leave ticked. Without it the
  closest contact can be a few tenths of an Ångström and LAMMPS will not
  survive it.
- **Apply the force field and write LAMMPS** — leave ticked.

---

## Step 3 · Relax

**Optional. Off by default because it needs LAMMPS or GROMACS installed.**

This is the step Materials Studio calls relaxation: BIOVIA describe Amorphous
Cell as an initial guess *followed by relaxation to a state of minimum
potential energy*. Without it you get the construction only.

- **Run LAMMPS after export** — tick to enable.
- **Engine** — leave on Automatic. It follows the force field: Moltemplate
  force fields are LAMMPS-only; DL_FIELD ones prefer GROMACS when `gmx` is
  installed, because DL_FIELD's `.top` states every functional form and
  nothing has to be inferred.
- **LAMMPS / GROMACS** — leave blank to auto-detect. Press **Detect** to see
  what was found. PAAF searches `PATH` and every sibling conda environment, so
  a binary in another env is still found. You can also type an environment
  name.
- **Push-off steps** (10 000) and **Minimiser iterations** (10 000) — raise
  for a denser or more strained cell.

### What relaxation does

1. Ramps up a bounded **soft** repulsion with capped per-step displacement, so
   overlapping atoms separate without ever seeing a singular force.
2. Restores the real force field.
3. Minimises with conjugate gradient.

A plain minimisation would fail: at a 1 Å contact a 12-6 potential is roughly
three million times ε, and LAMMPS either aborts or throws atoms across the
box.

The **before and after** table shows density, closest contact, R_g and bond
lengths. Expect the closest contact to jump from ~1 Å to something sane and
the energy to drop sharply. **Expect R_g to barely move** — see below.

---

## Step 4 · Export

Set the **Cells folder** and a **Cell name**, then press **Build cell**.

### What you get

```
<cells folder>/<name>/
├── cell_beads.xyz              the grown backbone, one site per skeletal atom
├── cell_beads.data             coarse-grained LAMMPS: masses, bonds, box
├── cell_atomistic.xyz          all-atom, side groups and hydrogens restored
├── cell_atomistic.mol2         same, with bond orders
├── cell.data                   typed LAMMPS data — the main deliverable
├── cell_relaxed.data           after minimisation (if Relax ran)
├── cell_atomistic_relaxed.xyz  relaxed coordinates for viewing
└── relax/
    ├── relax.in                the LAMMPS script
    ├── relaxed.lammpstrj       final frame
    └── paaf_relax.log          full LAMMPS output
```

Open any `.xyz` in VMD, OVITO or Avogadro. The bead file will look coarse —
that is what it is.

### Read the result messages

They are not decoration. Look for:

- `Charges: 124 atoms, net +0.0000 e, largest |q| 0.180 e` — good. An
  all-zero warning means electrostatics are silently absent.
- `Ring-spearing check: none found` — good. If bonds thread a ring, rebuild
  with a different seed or lower density; minimisation cannot undo it.
- Any `WARNING:` line.

---

## After PAAF

**Minimisation is not equilibration.** It finds a nearby local minimum —
effectively a 0 K glass. Chain dimensions relax on a diffusive timescale that
only molecular dynamics reaches, so the construction's chain over-extension
(about 20%) survives minimisation untouched.

Before quoting any property:

1. **NVT** at elevated temperature to let the chains move.
2. **NPT** at your target temperature and pressure to settle the density.
3. Re-measure C_∞ and R_g on the equilibrated cell, not the constructed one.

Materials Studio has the same division: Amorphous Cell constructs, Forcite
minimises, and equilibration is a separate dynamics run.

---

## Checking things without the GUI

Validate that LAMMPS accepts the generated script, end to end, in seconds:

```bash
python scripts/check_relax.py
```

Builds a tiny cell, types it, prints the deck, and runs `lmp -skiprun` on it.

Run the physics tests, which print every measured value:

```bash
python -m pytest tests/test_ris.py tests/test_grow.py -q -s
```

---

## If a build fails

The error names what to change, in order of what usually helps. The most
common causes:

| Symptom | Cause | Fix |
|---|---|---|
| "Could not grow chain N" | Box too dense for the chain length | Lower the density, shorten the chains, or raise look-ahead to 1–2 (and then don't quote C_n) |
| Overlap tolerance blamed | It is above ~1.75 Å | Put it back to 1.70 |
| `beads only` on a row | Backbone ring | Coarse-grained export only for that polymer |
| Bond lengths look absurd | Cell smaller than twice the repeat unit | Raise Cell size or DP |

---

## Limitations worth knowing

1. **Construction, then minimisation — not equilibration.** Run MD.
2. **Growth is at skeletal-atom resolution**, with side groups rebuilt
   afterwards from one template rotamer. Side-group conformational statistics
   are not sampled.
3. **RIS parameters exist for four polymers** — PE, PP, PS, PMMA. Everything
   else uses a generic model, and the result says so. Chain dimensions from a
   generic model are qualitative.
4. **24 of 123 library polymers have backbone rings** and cannot be built
   all-atom.
5. **Sequential construction over-extends chains by roughly 20%.** Treat a
   constructed cell's C_n as an upper bound until you have equilibrated.

The full theory, equations and validation are in
`docs/paaf_amorphous_cell_theory.pdf`.

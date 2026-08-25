# Build a PBS monomer from the built-in library

```bash
# List every preset:
moltemplate-auto build-molecule --list

# Build PBS as an optimized 3D PDB:
moltemplate-auto build-molecule --recipe PBS --out pbs_monomer.pdb

# Or any SMILES you like:
moltemplate-auto build-molecule --smiles "OCC(=O)OCCOC(=O)CO" --out custom.pdb
```

The output file can be dropped into the Monomers tab of the GUI or used in
any `Config` YAML.

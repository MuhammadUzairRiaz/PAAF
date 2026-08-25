"""Force-field atom typers.

Each typer takes a :class:`paaf.structure.Molecule` and returns
a mapping from atom index -> force-field atom type string. The returned
strings are used verbatim as ``@atom:<type>`` in the generated Moltemplate
.lt file, so they must match the atom types defined in the imported FF
library (e.g. ``oplsaa2024.lt``).
"""
from .oplsaa import type_oplsaa, OPLS_TYPES  # noqa: F401

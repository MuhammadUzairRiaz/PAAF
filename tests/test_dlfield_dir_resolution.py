"""Point PAAF at either DL_FIELD folder and it must still find dl_field.

Two things get called "the DL_FIELD directory": the install root, which holds
the ``dl_field`` executable, and its ``lib/`` subfolder, which holds the
parameter files. ``find_dl_field`` looks for ``<dir>/dl_field``.

The pipeline passes the root (``dl_lib.parent``) and works. The amorphous
cell passed the lib folder straight through, so it searched inside lib/ and
reported "dl_field executable not found" for an installation the pipeline was
using successfully two clicks away. The GUI field is even labelled "DL_FIELD
lib dir", so the user entered exactly the right thing and got told it was
missing.
"""
from __future__ import annotations

from paaf.cell.cell_export import _dlfield_root


def _install(tmp_path, with_exe=True):
    root = tmp_path / "dl_f_4.13"
    (root / "lib").mkdir(parents=True)
    if with_exe:
        (root / "dl_field").write_text("#!/bin/sh\n")
    return root


def test_the_lib_folder_resolves_to_the_install_root(tmp_path):
    """What the GUI field actually contains."""
    root = _install(tmp_path)
    assert _dlfield_root(root / "lib") == root


def test_the_install_root_is_accepted_unchanged(tmp_path):
    """What the pipeline passes."""
    root = _install(tmp_path)
    assert _dlfield_root(root) == root


def test_none_stays_none(tmp_path):
    assert _dlfield_root(None) is None


def test_a_lib_folder_with_no_executable_still_points_at_the_root(tmp_path):
    """So the error names where dl_field BELONGS, not where lib is."""
    root = _install(tmp_path, with_exe=False)
    assert _dlfield_root(root / "lib") == root


def test_an_unrelated_folder_is_left_alone(tmp_path):
    other = tmp_path / "somewhere"
    other.mkdir()
    assert _dlfield_root(other) == other

"""Polymers the user adds must persist, and must not be lost to an update.

The request
-----------
"there should be an option where user can add custom smile and he want it to be
included in the library ... whenever he loads the paaf the custom smile should
be in the library and if he want it to del then del it".

Where they are kept matters as much as that they are kept. The obvious place —
appending to ``paaf/data/polymer_database.csv`` — works until PAAF is
reinstalled, at which point the file is replaced and the user's polymers
vanish silently. Nobody keeps a backup of a file they did not know they had
edited. So the store is ``~/.paaf/custom_polymers.csv``, outside the package.

What is checked here is the store and the library merge. The dialog that
drives it needs Qt and is not exercised.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paaf.custom_library import (                                # noqa: E402
    CustomLibraryError, CustomPolymer, add_custom, delete_custom, load_custom,
    save_custom, store_path, update_custom, validate_entry,
)

#: Polyisobutylene — real, not in the shipped library, and its tail link atom
#: becomes quaternary in the chain, which an earlier draft of the validator
#: wrongly refused.
PIB = ("MY_PIB", "[*]CC(C)(C)[*]", "Polyisobutylene")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store of our own, so a test never touches the user's real one."""
    path = tmp_path / "custom_polymers.csv"
    monkeypatch.setenv("PAAF_CUSTOM_LIBRARY", str(path))
    return path


# ================================================================== the store
def test_a_polymer_added_is_there_the_next_time(store):
    """The whole point: it survives the process that added it."""
    add_custom(*PIB)
    reloaded = load_custom()                    # fresh read from disk
    print(f"\n  {store.name}: {list(reloaded)}")
    assert reloaded["MY_PIB"] == CustomPolymer(*PIB)


def test_the_store_lives_outside_the_package(monkeypatch):
    """An update replaces everything under paaf/; it must not live there."""
    monkeypatch.delenv("PAAF_CUSTOM_LIBRARY", raising=False)
    path = store_path()
    package = Path(__file__).resolve().parent.parent / "paaf"
    print(f"\n  store: {path}")
    assert package not in path.parents, \
        "custom polymers would be deleted by reinstalling PAAF"
    assert path.name.endswith(".csv")


def test_a_missing_store_is_not_an_error(store):
    """First run: no file yet."""
    assert not store.exists()
    assert load_custom() == {}


def test_an_unreadable_store_does_not_take_the_library_with_it(store):
    """A corrupt user file must not stop PAAF from starting."""
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b"\xff\xfe not a csv at all \x00")
    assert isinstance(load_custom(), dict)


def test_several_polymers_coexist(store):
    add_custom(*PIB)
    add_custom("MY_PEMA", "[*]CC(C)(C(=O)OCC)[*]", "Poly(ethyl methacrylate)")
    assert set(load_custom()) == {"MY_PIB", "MY_PEMA"}


# ============================================================ edit and remove
def test_editing_changes_what_is_stored(store):
    add_custom(*PIB)
    update_custom("MY_PIB", "MY_PIB", PIB[1], "Butyl rubber")
    assert load_custom()["MY_PIB"].name == "Butyl rubber"


def test_editing_can_rename_the_id_without_leaving_the_old_one(store):
    add_custom(*PIB)
    update_custom("MY_PIB", "PIB2", PIB[1], PIB[2])
    entries = load_custom()
    print(f"\n  after rename: {list(entries)}")
    assert set(entries) == {"PIB2"}, "the old id was left behind"


def test_editing_something_that_is_not_yours_is_refused(store):
    with pytest.raises(CustomLibraryError, match="not one of your"):
        update_custom("W01_P001", "W01_P001", "[*]CC[*]", "Polyethylene")


def test_removing_takes_it_out_and_leaves_the_others(store):
    add_custom(*PIB)
    add_custom("MY_PEMA", "[*]CC(C)(C(=O)OCC)[*]", "PEMA")
    assert delete_custom("MY_PIB") is True
    assert set(load_custom()) == {"MY_PEMA"}


def test_removing_something_absent_says_so_rather_than_raising(store):
    assert delete_custom("NOT_THERE") is False


# ================================================================ validation
def test_a_smiles_without_two_marks_is_refused():
    with pytest.raises(CustomLibraryError, match=r"exactly two"):
        validate_entry("X", "[*]CC(C)C", "no tail mark")
    with pytest.raises(CustomLibraryError, match=r"exactly two"):
        validate_entry("X", "CC", "no marks at all")


def test_an_unparseable_smiles_is_refused():
    with pytest.raises(CustomLibraryError, match="cannot read"):
        validate_entry("X", "[*]C(C)(C)(C)(C)[*]", "over-valent carbon")


def test_a_quaternary_link_atom_is_ACCEPTED():
    """The validator's own bug, pinned.

    A first version counted hydrogens on the link atom and refused
    polyisobutylene — a real polymer PAAF builds correctly. The wildcard is
    not something extra hanging off a full carbon; it IS the hydrogen the
    junction consumes, so a link atom never needs one of its own.
    """
    pid, smiles, name = validate_entry(*PIB)
    print(f"\n  accepted {pid}: {smiles}")
    assert smiles == PIB[1]


@pytest.mark.parametrize("smiles", [
    "[*]CC[*]",                       # polyethylene
    "[*]CC(C)(C)[*]",                 # polyisobutylene, quaternary
    "[*]OCCCCOC(=O)CCC(=O)[*]",       # PBS, carbonyl tail
    "[*]CC([*])c1ccccc1",             # polystyrene, marks not at the ends
    "[*]C=C[*]",                      # polyacetylene
])
def test_real_polymers_are_accepted(smiles):
    validate_entry("X", smiles, "a polymer")


def test_an_id_that_would_break_the_file_is_refused():
    for bad in ("has space", "comma,inside", ""):
        with pytest.raises(CustomLibraryError):
            validate_entry(bad, "[*]CC[*]", "Polyethylene")


def test_a_missing_name_is_refused():
    with pytest.raises(CustomLibraryError, match="name"):
        validate_entry("X", "[*]CC[*]", "  ")


def test_a_duplicate_id_is_refused_but_not_when_editing_itself(store):
    add_custom(*PIB)
    with pytest.raises(CustomLibraryError, match="already used"):
        add_custom("MY_PIB", "[*]CC[*]", "something else")
    # Editing the entry keeps its own id available to it.
    update_custom("MY_PIB", "MY_PIB", "[*]CC[*]", "renamed")
    assert load_custom()["MY_PIB"].name == "renamed"


def test_whitespace_is_trimmed_rather_than_stored():
    pid, smiles, name = validate_entry("  MY_PIB  ", "  [*]CC[*] ", " PE ")
    assert (pid, smiles, name) == ("MY_PIB", "[*]CC[*]", "PE")


# ============================================================ library merge
def test_a_custom_polymer_appears_in_the_library(store):
    from paaf import builder

    add_custom(*PIB)
    builder.reload_library()
    rec = builder.get_recipe("MY_PIB")
    print(f"\n  {rec.pid}: {rec.description} {rec.tags}")
    assert rec.smiles == PIB[1]
    assert "custom" in rec.tags, "it must be distinguishable from shipped ones"


def test_it_can_also_be_found_by_its_name(store):
    from paaf import builder

    add_custom(*PIB)
    builder.reload_library()
    assert builder.get_recipe("POLYISOBUTYLENE").smiles == PIB[1]


def test_it_is_listed_under_its_own_source(store):
    from paaf import builder

    add_custom(*PIB)
    builder.reload_library()
    mine = [r.pid for r in builder.list_library("custom")]
    print(f"\n  custom: {mine}")
    assert mine == ["MY_PIB"]


def test_removing_it_takes_it_out_of_the_library_too(store):
    from paaf import builder

    add_custom(*PIB)
    builder.reload_library()
    delete_custom("MY_PIB")
    builder.reload_library()
    with pytest.raises(KeyError):
        builder.get_recipe("MY_PIB")


def test_the_shipped_library_is_still_all_there(store):
    """Adding one polymer must not shadow or drop a hundred others."""
    from paaf import builder

    before = len(builder.list_library("all"))
    add_custom(*PIB)
    builder.reload_library()
    after = len(builder.list_library("all"))
    print(f"\n  {before} -> {after}")
    assert after == before + 1
    assert builder.get_recipe("W01_P001").description == "Polyethylene"


def test_reload_updates_the_shared_dict_in_place(store):
    """The GUI and the pipeline both hold a reference to LIBRARY.

    Rebinding the module global would leave them looking at a stale dict, and
    the new polymer would be visible in some places and not others.
    """
    from paaf import builder

    held = builder.LIBRARY
    add_custom(*PIB)
    builder.reload_library()
    assert builder.LIBRARY is held, "LIBRARY was replaced rather than updated"
    assert "MY_PIB" in held


def test_a_custom_entry_builds_the_same_way_a_library_one_does(store):
    """It must carry the [*] SMILES, since that is what makes links reliable."""
    pytest.importorskip("rdkit")
    from paaf import builder

    add_custom(*PIB)
    builder.reload_library()
    rec = builder.get_recipe("MY_PIB")
    assert "[*]" in rec.smiles, "no wildcards means the links get guessed"
    assert rec.monomer_smiles and "[*]" not in rec.monomer_smiles
    print(f"\n  poly={rec.smiles}  closed-shell={rec.monomer_smiles}")


# =================================================================== the file
def test_the_file_is_readable_by_a_person(store):
    add_custom(*PIB)
    text = store.read_text(encoding="utf-8")
    print(f"\n{text}")
    assert text.splitlines()[0] == "pid,smiles,name"
    assert "MY_PIB" in text


def test_hand_written_rows_are_accepted(store):
    """Someone will edit this file directly. It should work."""
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("pid,smiles,name\nHAND,[*]CC[*],Polyethylene by hand\n",
                     encoding="utf-8")
    assert load_custom()["HAND"].name == "Polyethylene by hand"


def test_incomplete_rows_are_skipped_not_fatal(store):
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("pid,smiles,name\n"
                     "GOOD,[*]CC[*],Polyethylene\n"
                     ",,\n"
                     "NOSMILES,,Nothing\n", encoding="utf-8")
    assert set(load_custom()) == {"GOOD"}


def test_saving_creates_the_directory(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "deeper" / "custom.csv"
    monkeypatch.setenv("PAAF_CUSTOM_LIBRARY", str(path))
    save_custom({"X": CustomPolymer("X", "[*]CC[*]", "PE")})
    assert path.exists()

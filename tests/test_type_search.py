from paaf.lt_parser import AtomTypeInfo
from paaf.type_search import filter_types


def _t(fid, el="C", key="CT", q=0.0, desc="", mass=12.011):
    return AtomTypeInfo(ff_id=fid, element=el, key=key, charge=q, description=desc, mass=mass)


TYPES = [_t("68", key="C3", desc="CH3 (C2) N-ALKANES", mass=12.011),
         _t("UA:68", key="C3", desc="[UA] CH3 (C2) N-ALKANES", mass=15.035),
         _t("1068", key="CT", desc="something 68"), _t("168", key="CT"),
         _t("136", key="CT", q=-0.12, desc="CH2 all-atom C: alkanes"),
         _t("140", el="H", key="HC", q=0.06, desc="H on alkane", mass=1.008),
         _t("UA:71", key="C2", desc="[UA] CH2 (SP3) ALKANES", mass=14.027)]


def test_id_search_exact_first_then_prefix():
    r = [t.ff_id for t in filter_types(TYPES, "68", "id")]
    assert r == ["68", "UA:68"]
    r = [t.ff_id for t in filter_types(TYPES, "1", "id")]
    assert r[:2] == ["1068", "168"] or set(r) >= {"1068", "168", "136", "140"}


def test_all_fields_puts_exact_id_first():
    r = [t.ff_id for t in filter_types(TYPES, "68", "all")]
    assert r[:2] == ["68", "UA:68"] and "1068" in r


def test_key_element_mass_charge():
    assert [t.ff_id for t in filter_types(TYPES, "C3", "key")] == ["68", "UA:68"]
    assert {t.ff_id for t in filter_types(TYPES, "H", "element")} == {"140"}
    assert [t.ff_id for t in filter_types(TYPES, "14.03", "mass")] == ["UA:71"]
    assert [t.ff_id for t in filter_types(TYPES, "-0.12", "charge")] == ["136"]
    assert [t.ff_id for t in filter_types(TYPES, "alkane", "desc")] == ["68", "UA:68", "136", "140", "UA:71"]


def test_library_filter():
    assert {t.ff_id for t in filter_types(TYPES, "", "all", library="ua")} == {"UA:68", "UA:71"}
    assert all(not t.ff_id.startswith("UA:") for t in filter_types(TYPES, "", "all", library="aa"))
    assert [t.ff_id for t in filter_types(TYPES, "68", "id", library="ua")] == ["UA:68"]

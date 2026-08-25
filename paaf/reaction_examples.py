"""A built-in library of worked reaction-scheme examples.

Every entry is a complete, mapped reactant/product scheme that validates
against :mod:`paaf.reaction_smiles` — the test suite runs each one through
``validate_scheme`` so a broken example cannot ship. The GUI shows them in a
browser dialog (Reaction scheme step 1) from which any example can be loaded
into a reaction block with one click.

Writing conventions used throughout:

* Map numbers tag ONLY atoms whose bonds change (plus twins that must be
  told apart). ``[OH:1]`` = "this oxygen, with its H, is atom 1".
* A number present in the reactants but absent from the products is a
  leaving atom.
* Hydrogens are completed automatically at build time; ``[C:2]`` keeps its
  normal hydrogens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

__all__ = ["ReactionExample", "EXAMPLES", "CATEGORIES", "by_category"]


@dataclass(frozen=True)
class ReactionExample:
    key: str
    name: str
    category: str
    #: What the reaction is and why it matters — one or two sentences.
    description: str
    #: ((row label, mapped SMILES), ...)
    reactants: Tuple[Tuple[str, str], ...]
    products: Tuple[Tuple[str, str], ...]
    #: ((":1", "what this tag marks"), ...)
    tags: Tuple[Tuple[str, str], ...]
    #: Optional extra reading help.
    note: str = ""


E = ReactionExample

EXAMPLES: List[ReactionExample] = [

# ------------------------------------------------- esters & condensation
E("esterification", "Esterification (acid + alcohol)", "Esters & condensation",
  "A carboxylic acid and an alcohol condense to an ester plus water. The "
  "backbone reaction of polyesters (PET, PBS, PLA).",
  (("acid", "CC(=O)[OH:1]"), ("alcohol", "CC[OH:2]")),
  (("ester", "CC(=O)[O:2]CC"), ("water", "O")),
  ((":1", "the acid's OH oxygen — leaves as part of the water"),
   (":2", "the alcohol oxygen — survives as the ester bridge O")),
  "Read CC(=O)[OH:1] as CH3-C(=O)-OH: C, then a C with a double-bonded O "
  "in the (branch), then the OH."),

E("transesterification", "Transesterification", "Esters & condensation",
  "An ester swaps its alkoxy group for a new alcohol. How PET is made "
  "industrially from dimethyl terephthalate.",
  (("methyl ester", "CC(=O)[O:1]C"), ("alcohol", "CC[OH:2]")),
  (("new ester", "CC(=O)[O:2]CC"), ("methanol", "C[OH:1]")),
  ((":1", "the old ester oxygen — leaves with the methyl as methanol"),
   (":2", "the incoming alcohol oxygen — becomes the new bridge"))),

E("anhydride", "Anhydride formation (2 acids)", "Esters & condensation",
  "Two carboxylic acids condense to an anhydride plus water. The two acids "
  "are identical, so tags are what tell them apart.",
  (("acid A", "CC(=O)[OH:5]"), ("acid B", "CC(=O)[OH:6]")),
  (("anhydride", "CC(=O)[O:6]C(C)=O"), ("water", "O")),
  ((":5", "acid A's OH oxygen — the leaving group"),
   (":6", "acid B's OH oxygen — survives as the anhydride bridge"))),

E("ether_condensation", "Ether condensation (2 alcohols)",
  "Esters & condensation",
  "Two alcohols condense to an ether plus water (acid-catalysed).",
  (("alcohol A", "CC[OH:1]"), ("alcohol B", "CC[OH:2]")),
  (("ether", "CC[O:2]CC"), ("water", "O")),
  ((":1", "leaves in the water"), (":2", "becomes the ether oxygen"))),

E("pet_model", "PET link (terephthalate + glycol)", "Esters & condensation",
  "A polyester chain-growth step on a real monomer pair: methyl "
  "terephthalate + ethylene glycol.",
  (("terephthalate", "COC(=O)c1ccc(cc1)C(=O)[O:1]C"),
   ("ethylene glycol", "[OH:2]CCO")),
  (("chain ester", "COC(=O)c1ccc(cc1)C(=O)[O:2]CCO"),
   ("methanol", "C[OH:1]")),
  ((":1", "the methyl-ester oxygen displaced as methanol"),
   (":2", "the glycol oxygen that forms the new ester")),
  "c1ccc(cc1) is a benzene ring written in aromatic (lower-case) SMILES."),

E("caprolactone_rop", "Ring-opening of caprolactone", "Esters & condensation",
  "An alcohol opens the 7-membered lactone ring — the initiation step of "
  "polycaprolactone (PCL) growth.",
  (("caprolactone", "O=[C:1]1CCCCC[O:2]1"), ("alcohol", "CC[OH:3]")),
  (("opened ester", "CC[O:3][C:1](=O)CCCCC[OH:2]"),),
  ((":1", "the lactone carbonyl carbon (acyl attack site)"),
   (":2", "the ring oxygen — the C1-O2 ring bond breaks; O2 becomes the "
          "chain-end OH"),
   (":3", "the initiating alcohol oxygen — new ester bridge")),
  "No leaving group: a ring-opening just relocates bonds."),

E("acetal", "Acetal formation (aldehyde + 2 alcohols)",
  "Esters & condensation",
  "An aldehyde condenses with two alcohols to an acetal plus water — the "
  "chemistry of polyvinyl butyral (PVB).",
  (("aldehyde", "CC(=[O:1])[H]"), ("alcohol A", "C[OH:2]"),
   ("alcohol B", "C[OH:3]")),
  (("acetal", "C[CH:4](O)..."),),
  ((":1", "placeholder"),), "PLACEHOLDER"),

# --------------------------------------------------- amides & polyamides
E("amidation", "Amidation (acid + amine)", "Amides & polyamides",
  "A carboxylic acid and an amine condense to an amide plus water — the "
  "nylon reaction.",
  (("acid", "CC(=O)[OH:1]"), ("amine", "CC[NH2:2]")),
  (("amide", "CC(=O)[NH:2]CC"), ("water", "O")),
  ((":1", "acid OH oxygen — leaves in the water"),
   (":2", "amine nitrogen — forms the new C-N amide bond"))),

E("nylon66_model", "Polyamide 6,6 link", "Amides & polyamides",
  "Adipic acid + hexamethylenediamine forming one amide link, with both "
  "far chain ends left free to keep reacting.",
  (("adipic acid", "O=C(O)CCCCC(=O)[OH:1]"),
   ("diamine", "[NH2:2]CCCCCCN")),
  (("amide link", "O=C(O)CCCCC(=O)[NH:2]CCCCCCN"), ("water", "O")),
  ((":1", "the reacting acid OH — leaves as water"),
   (":2", "the reacting amine N — only ONE end of each monomer is tagged; "
          "the other ends stay free"))),

E("acid_chloride_amine", "Acid chloride + amine", "Amides & polyamides",
  "The fast interfacial-polymerisation route to polyamides (the nylon rope "
  "trick). HCl instead of water leaves.",
  (("acid chloride", "CC(=O)[Cl:1]"), ("amine", "CC[NH2:2]")),
  (("amide", "CC(=O)[NH:2]CC"),),
  ((":1", "the chlorine — a leaving atom: present in the reactants, absent "
          "from the products (it departs as HCl, which is not drawn "
          "because hydrogen halides cannot be force-field typed)"),
   (":2", "the amine nitrogen — new amide bond"))),

E("acid_chloride_alcohol", "Acid chloride + alcohol", "Amides & polyamides",
  "Ester formation via the activated acid — fast and irreversible.",
  (("acid chloride", "CC(=O)[Cl:1]"), ("alcohol", "CC[OH:2]")),
  (("ester", "CC(=O)[O:2]CC"),),
  ((":1", "the chlorine — a leaving atom, expelled as HCl (not drawn)"),
   (":2", "the new ester oxygen"))),

E("amic_acid", "Anhydride + amine (amic acid)", "Amides & polyamides",
  "An amine opens a cyclic anhydride to an amic acid — step 1 of every "
  "polyimide synthesis.",
  (("maleic anhydride", "O=[C:1]1C=CC(=O)[O:2]1"), ("amine", "CC[NH2:3]")),
  (("amic acid", "CC[NH:3][C:1](=O)C=CC(=O)[OH:2]"),),
  ((":1", "the attacked carbonyl carbon"),
   (":2", "the ring oxygen — the C1-O2 bond breaks; O2 becomes a free "
          "COOH oxygen"),
   (":3", "the amine nitrogen — new amide bond to C1"))),

E("imidization", "Amic acid → imide (cyclisation)",
  "Amides & polyamides",
  "The amic acid closes to a five-membered imide ring, expelling water — "
  "step 2 of polyimide synthesis.",
  (("amic acid", "CC[NH:1]C(=O)C=C[C:2](=O)[OH:3]"),),
  (("imide", "CC[N:1]1C(=O)C=C[C:2]1=O"), ("water", "O")),
  ((":1", "the amide nitrogen — forms the second N-C bond, closing the "
          "ring"),
   (":2", "the free-acid carbonyl carbon it attacks"),
   (":3", "the acid OH — leaves as water"))),

# --------------------------------------------------- epoxide ring-opening
E("epoxy_amine", "Epoxide + amine", "Epoxide ring-opening",
  "The workhorse epoxy-cure reaction: an amine opens the ring at the less "
  "hindered carbon; the epoxide oxygen becomes a secondary OH.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"), ("amine", "CC[NH2:4]")),
  (("amino alcohol", "C[CH:1]([OH:2])[CH2:3][NH:4]CC"),),
  ((":1", "the substituted ring carbon (keeps the O as OH)"),
   (":2", "the epoxide oxygen — the O2-C3 bond breaks; O2 stays on C1"),
   (":3", "the terminal CH2 — the attacked carbon"),
   (":4", "the amine nitrogen — new C-N bond to C3"))),

E("epoxy_acid", "Epoxide + carboxylic acid", "Epoxide ring-opening",
  "A carboxylic acid opens an epoxide to a hydroxy-ester. On a "
  "trisubstituted (ENR-type) epoxide the acid attacks the TERTIARY carbon.",
  (("epoxide (ENR-type)", "C[C:1]1(C)[O:2][CH:3]1C"),
   ("acid", "CCC(=O)[OH:4]")),
  (("hydroxy-ester", "C[C:1](C)([O:4]C(=O)CC)[CH:3](C)[OH:2]"),),
  ((":1", "the tertiary epoxide carbon (no H, three C neighbours) — "
          "receives the acid oxygen"),
   (":2", "the epoxide oxygen — C1-O2 breaks; O2 stays on C3 as OH"),
   (":3", "the secondary epoxide carbon"),
   (":4", "the attacking acid oxygen — new C1-O4 ester bond"))),

E("epoxy_thiol", "Epoxide + thiol", "Epoxide ring-opening",
  "Thiol-epoxy 'click' cure: fast, efficient, base-catalysed.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"), ("thiol", "CC[SH:4]")),
  (("hydroxy-sulfide", "C[CH:1]([OH:2])[CH2:3][S:4]CC"),),
  ((":2", "epoxide O → secondary OH"),
   (":3", "attacked CH2"), (":4", "the sulfur — new C-S bond"))),

E("epoxy_phenol", "Epoxide + phenol", "Epoxide ring-opening",
  "Phenol opens the epoxide to an aryl ether — the advancement reaction of "
  "epoxy resins.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"), ("phenol", "[OH:4]c1ccccc1")),
  (("aryl ether", "C[CH:1]([OH:2])[CH2:3][O:4]c1ccccc1"),),
  ((":2", "epoxide O → OH"), (":3", "attacked CH2"),
   (":4", "the phenol oxygen — new ether bond"))),

E("epoxy_alcohol", "Epoxide + alcohol", "Epoxide ring-opening",
  "Etherification — the side reaction in amine-cured epoxies that creates "
  "extra crosslinks at high temperature.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"), ("alcohol", "CC[OH:4]")),
  (("hydroxy-ether", "C[CH:1]([OH:2])[CH2:3][O:4]CC"),),
  ((":2", "epoxide O → OH"), (":4", "alcohol O → ether bridge"))),

E("epoxy_water", "Epoxide + water (hydrolysis)", "Epoxide ring-opening",
  "Water opens the epoxide to a 1,2-diol.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"), ("water", "[OH2:4]")),
  (("diol", "C[CH:1]([OH:2])[CH2:3][OH:4]"),),
  ((":2", "epoxide O → one OH"), (":4", "water O → the other OH"))),

E("epoxy_anhydride", "Epoxide + anhydride", "Epoxide ring-opening",
  "Anhydride cure of epoxies: the ring opens and BOTH oxygens end up as "
  "esters — a diester crosslink.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"),
   ("anhydride", "CC(=O)[O:4][C:5](C)=O")),
  (("diester", "C[CH:1]([O:2][C:5](C)=O)[CH2:3][O:4]C(C)=O"),),
  ((":2", "epoxide O — acylated by carbonyl C5 (new O2-C5 bond)"),
   (":3", "ring carbon attacked by the anhydride oxygen O4"),
   (":4", "anhydride bridge O — its bond to C5 breaks; new bond to C3"),
   (":5", "one anhydride carbonyl carbon"))),

# ----------------------------------------------- ENR / rubber chemistry
E("enr_mah_enr", "ENR + maleic acid + ENR (bridge)",
  "ENR / rubber chemistry",
  "Both COOH groups of maleic acid open one epoxide each, bridging two ENR "
  "chains — a crosslink. Model compounds stand in for the chains.",
  (("ENR epoxide A", "C[C:1]1(C)[O:2][CH:3]1C"),
   ("maleic acid", "[OH:4]C(=O)/C=C\\C(=O)[OH:5]"),
   ("ENR epoxide B", "C[C:6]1(C)[O:7][CH:8]1C")),
  (("bridged product",
    "C[C:1](C)([O:4]C(=O)/C=C\\C(=O)[O:5][C:6](C)(C)[CH:8](C)[O:7])"
    "[CH:3](C)[O:2]"),),
  ((":1/:6", "the two tertiary epoxide carbons — each receives an acid "
             "oxygen"),
   (":2/:7", "the two epoxide oxygens — freed from the tertiary carbons, "
             "they stay on :3/:8 as OH groups"),
   (":4/:5", "the two acid oxygens — the new ester bridges")),
  "/C=C\\ marks the cis double bond of maleic acid. No atom leaves."),

E("enr_mah_pbs", "ENR + maleic acid + PBS-COOH (graft)",
  "ENR / rubber chemistry",
  "One maleic-acid COOH opens the ENR epoxide; the other condenses with "
  "the PBS terminal COOH to an anhydride, expelling water. The PBS oxygen "
  "is retained as the bridge.",
  (("ENR epoxide", "C[C:1]1(C)[O:2][CH:3]1C"),
   ("maleic acid", "[OH:4]C(=O)/C=C\\C(=O)[OH:5]"),
   ("PBS acid end", "COC(=O)CCC(=O)[OH:6]")),
  (("grafted product",
    "C[C:1](C)([O:4]C(=O)/C=C\\C(=O)[O:6]C(=O)CCC(=O)OC)[CH:3](C)[O:2]"),
   ("water", "O")),
  ((":1", "tertiary ENR carbon — bonded to acid oxygen :4"),
   (":2", "epoxide O → OH on the secondary carbon"),
   (":4", "maleic oxygen forming the ester to ENR"),
   (":5", "maleic acid's second OH — the leaving group (absent from the "
          "products)"),
   (":6", "the PBS carboxyl oxygen — RETAINED as the anhydride bridge")),
  "Monomethyl succinate COC(=O)CCC(=O)OH models the true PBS chain end: COOH, two CH2 groups, then the in-chain ester."),

E("enr_pbs_direct", "ENR + PBS-COOH (direct ester)",
  "ENR / rubber chemistry",
  "The PBS terminal carboxylic acid opens the ENR epoxide directly — no "
  "maleic acid involved. Same regiochemistry: attack at the tertiary "
  "carbon.",
  (("ENR epoxide", "C[C:1]1(C)[O:2][CH:3]1C"),
   ("PBS acid end", "COC(=O)CCC(=O)[OH:4]")),
  (("graft ester", "C[C:1](C)([O:4]C(=O)CCC(=O)OC)[CH:3](C)[OH:2]"),),
  ((":1", "tertiary epoxide carbon"),
   (":2", "epoxide O → OH on the secondary carbon"),
   (":4", "PBS acid oxygen → the ester link"))),

E("enr_hydrolysis", "ENR epoxide + water (diol)", "ENR / rubber chemistry",
  "Ring hydrolysis during processing: the epoxide becomes a diol.",
  (("ENR epoxide", "C[C:1]1(C)[O:2][CH:3]1C"), ("water", "[OH2:4]")),
  (("diol", "C[C:1](C)([OH:4])[CH:3](C)[OH:2]"),),
  ((":2", "epoxide O → OH on the secondary carbon"),
   (":4", "water O → OH on the tertiary carbon"))),

E("enr_acid_secondary", "ENR + acid at the SECONDARY carbon",
  "ENR / rubber chemistry",
  "The alternative regiochemistry, for comparison with the tertiary "
  "attack: here the acid bonds to the secondary carbon and the OH lands "
  "on the tertiary one. Note how ONLY the product drawing changed.",
  (("ENR epoxide", "C[C:1]1(C)[O:2][CH:3]1C"), ("acid", "CCC(=O)[OH:4]")),
  (("iso-ester", "C[C:1](C)([OH:2])[CH:3](C)[O:4]C(=O)CC"),),
  ((":2", "epoxide O — stays on the TERTIARY carbon this time"),
   (":3", "secondary carbon — receives the acid oxygen"),
   (":4", "acid oxygen → ester on C3"))),

# ------------------------------------------- isocyanates & polyurethanes
E("urethane", "Isocyanate + alcohol (urethane)",
  "Isocyanates & polyurethanes",
  "The polyurethane bond: an alcohol adds across the N=C of the "
  "isocyanate. Pure addition — nothing leaves.",
  (("isocyanate", "CC[N:1]=[C:3]=O"), ("alcohol", "CC[OH:2]")),
  (("urethane", "CC[NH:1][C:3](=O)[O:2]CC"),),
  ((":1", "the isocyanate nitrogen — its double bond to C3 becomes single "
          "and it picks up the alcohol's H"),
   (":2", "the alcohol oxygen — new bond to C3"),
   (":3", "the isocyanate carbon — the electrophilic centre"))),

E("urea_bond", "Isocyanate + amine (urea)", "Isocyanates & polyurethanes",
  "The polyurea bond — the fast reaction in PU foams and elastomers.",
  (("isocyanate", "CC[N:1]=[C:3]=O"), ("amine", "CC[NH2:2]")),
  (("urea", "CC[NH:1][C:3](=O)[NH:2]CC"),),
  ((":1", "isocyanate N"), (":2", "amine N — new bond to C3"),
   (":3", "isocyanate C"))),

E("isocyanate_water", "Isocyanate + water (blowing reaction)",
  "Isocyanates & polyurethanes",
  "Water converts an isocyanate to an amine and CO2 — the gas that blows "
  "PU foams. The isocyanate carbon LEAVES as the CO2 carbon.",
  (("isocyanate", "CC[N:1]=[C:3]=O"), ("water", "[OH2:2]")),
  (("amine", "CC[NH2:1]"), ("CO2", "O=[C:3]=[O:2]")),
  ((":1", "the nitrogen — ends as a free amine"),
   (":2", "the water oxygen — ends in the CO2"),
   (":3", "the isocyanate carbon — follow it into the CO2"))),

# --------------------------------------------------- additions & click
E("thiol_ene", "Thiol-ene addition", "Additions & click",
  "A thiol adds across a C=C double bond (anti-Markovnikov) — the classic "
  "UV-cure click reaction.",
  (("thiol", "CC[SH:1]"), ("alkene", "[CH2:2]=[CH:3]C")),
  (("sulfide", "CC[S:1][CH2:2][CH2:3]C"),),
  ((":1", "the sulfur — new S-C bond to the terminal carbon"),
   (":2", "the terminal CH2 of the double bond"),
   (":3", "the inner alkene carbon — the C2=C3 double bond becomes "
          "single; C3 gains an H"))),

E("thiol_michael", "Thiol-Michael (thiol + acrylate)", "Additions & click",
  "A thiol adds to the activated double bond of an acrylate.",
  (("thiol", "CC[SH:1]"), ("acrylate", "[CH2:2]=[CH:3]C(=O)OC")),
  (("adduct", "CC[S:1][CH2:2][CH2:3]C(=O)OC"),),
  ((":1", "sulfur"), (":2/:3", "the acrylate double bond, saturated in "
                               "the product"))),

E("aza_michael", "Aza-Michael (amine + acrylate)", "Additions & click",
  "An amine adds to an acrylate double bond — used in ABC curing and "
  "polyaspartics.",
  (("amine", "CC[NH2:1]"), ("acrylate", "[CH2:2]=[CH:3]C(=O)OC")),
  (("adduct", "CC[NH:1][CH2:2][CH2:3]C(=O)OC"),),
  ((":1", "the nitrogen — new N-C bond"),
   (":2/:3", "the double bond that saturates"))),

E("oxa_michael", "Oxa-Michael (alcohol + acrylate)", "Additions & click",
  "An alcohol adds to an acrylate double bond.",
  (("alcohol", "CC[OH:1]"), ("acrylate", "[CH2:2]=[CH:3]C(=O)OC")),
  (("adduct", "CC[O:1][CH2:2][CH2:3]C(=O)OC"),),
  ((":1", "the oxygen — new O-C bond"),
   (":2/:3", "the saturating double bond"))),

E("diels_alder", "Diels-Alder (diene + maleic anhydride)",
  "Additions & click",
  "Butadiene and maleic anhydride form a cyclohexene ring in one step — "
  "two C-C bonds at once. Used for reversible/self-healing networks.",
  (("butadiene", "[CH2:1]=[CH:2][CH:3]=[CH2:4]"),
   ("maleic anhydride", "O=C1[CH:5]=[CH:6]C(=O)O1")),
  (("adduct", "O=C1[CH:5]2[CH:6](C(=O)O1)[CH2:4][CH:3]=[CH:2][CH2:1]2"),),
  ((":1/:4", "the diene's terminal carbons — each forms one new C-C bond"),
   (":2/:3", "the diene's inner carbons — the surviving double bond"),
   (":5/:6", "the anhydride's alkene carbons — the other ends of the two "
             "new bonds")),
  "The digit pairs 1...1 and 2...2 in the product close its two rings."),

E("hydrosilylation", "Hydrosilylation (Si-H + vinyl)", "Additions & click",
  "A silicon hydride adds across a vinyl group — the platinum-catalysed "
  "cure of addition silicones (LSR, RTV-2).",
  (("silane", "C[SiH:1](C)C"), ("vinyl", "[CH2:2]=[CH:3]C")),
  (("alkyl silane", "C[Si:1](C)(C)[CH2:2][CH2:3]C"),),
  ((":1", "the silicon — its H migrates to C3; new Si-C2 bond"),
   (":2/:3", "the vinyl double bond that saturates"))),

E("disulfide", "Disulfide coupling (peroxide-mediated)",
  "Additions & click",
  "Two thiols couple to a disulfide — the vulcanisation-type S-S bridge — "
  "with hydrogen peroxide taking up the two hydrogens as water.",
  (("thiol A", "CC[SH:1]"), ("thiol B", "CC[SH:2]"),
   ("peroxide", "[OH:3][OH:4]")),
  (("disulfide", "CC[S:1][S:2]CC"), ("water", "[OH2:3]"),
   ("water", "[OH2:4]")),
  ((":1/:2", "the two sulfurs — the new S-S bond"),
   (":3/:4", "the peroxide oxygens — the O-O bond breaks and each O "
             "leaves as a water"))),

# ------------------------------------------------------------ silicones
E("siloxane_condensation", "Silanol + silanol (siloxane bond)",
  "Silicones",
  "Two Si-OH groups condense to the Si-O-Si backbone bond plus water — "
  "how condensation-cure silicones set.",
  (("silanol A", "C[Si](C)(C)[OH:1]"), ("silanol B", "C[Si](C)(C)[OH:2]")),
  (("siloxane", "C[Si](C)(C)[O:2][Si](C)(C)C"), ("water", "O")),
  ((":1", "leaves in the water"), (":2", "becomes the Si-O-Si bridge"))),

E("silanol_alkoxy", "Silanol + alkoxysilane", "Silicones",
  "A silanol condenses with a methoxysilane, releasing methanol — the "
  "moisture-cure (RTV-1) crosslinking step.",
  (("silanol", "C[Si](C)(C)[OH:1]"),
   ("methoxysilane", "C[Si](C)(C)[O:2]C")),
  (("siloxane", "C[Si](C)(C)[O:1][Si](C)(C)C"), ("methanol", "C[OH:2]")),
  ((":1", "the silanol oxygen — becomes the bridge"),
   (":2", "the methoxy oxygen — leaves with its methyl as methanol"))),

E("alkoxysilane_hydrolysis", "Alkoxysilane + water", "Silicones",
  "Moisture hydrolyses Si-OMe to Si-OH — the step that precedes "
  "condensation in silane coupling agents and RTV-1.",
  (("methoxysilane", "C[Si](C)(C)[O:1]C"), ("water", "[OH2:2]")),
  (("silanol", "C[Si](C)(C)[OH:2]"), ("methanol", "C[OH:1]")),
  ((":1", "methoxy O — leaves as methanol"),
   (":2", "water O — becomes the new Si-OH"))),

# ------------------------------------------- phenolic & amino resins
E("phenol_methylol", "Phenol + formaldehyde (methylolation)",
  "Phenolic & amino resins",
  "Formaldehyde substitutes an ortho/para ring hydrogen of phenol, giving "
  "a methylol phenol — the first step of every phenolic resin.",
  (("phenol", "Oc1cc[cH:1]cc1"), ("formaldehyde", "[CH2:2]=O")),
  (("methylol phenol", "Oc1cc[c:1](cc1)[CH2:2]O"),),
  ((":1", "the reacting ring carbon (para here) — its H moves to the "
          "formaldehyde oxygen"),
   (":2", "the formaldehyde carbon — new bond to the ring"))),

E("methylol_bridge", "Methylol + phenol (methylene bridge)",
  "Phenolic & amino resins",
  "A methylol phenol condenses with another phenol ring, expelling water "
  "and leaving the CH2 bridge that cures novolacs and resoles.",
  (("methylol phenol", "Oc1ccc(cc1)[CH2:2][OH:3]"),
   ("phenol", "[cH:4]1ccc(O)cc1")),
  (("bridged pair", "Oc1ccc(cc1)[CH2:2][c:4]1ccc(O)cc1"), ("water", "O")),
  ((":2", "the bridging CH2 carbon"),
   (":3", "the methylol OH — leaves as water"),
   (":4", "the ring carbon of the second phenol that is attacked"))),

E("urea_formaldehyde", "Urea + formaldehyde", "Phenolic & amino resins",
  "Formaldehyde adds to a urea N-H giving a methylol urea — the start of "
  "UF (and, with melamine, MF) resins.",
  (("urea", "NC(=O)[NH2:1]"), ("formaldehyde", "[CH2:2]=O")),
  (("methylol urea", "NC(=O)[NH:1][CH2:2]O"),),
  ((":1", "the reacting nitrogen"),
   (":2", "the formaldehyde carbon — new N-C bond; its O becomes OH"))),

# --------------------------------------- substitutions & hydrolysis
E("williamson", "Williamson ether (halide + alcohol)",
  "Substitutions & hydrolysis",
  "An alkoxide/alcohol displaces a halide to form an ether.",
  (("alkyl bromide", "C[CH2:3][Br:1]"), ("alcohol", "CC[OH:2]")),
  (("ether", "C[CH2:3][O:2]CC"),),
  ((":1", "the bromine — a leaving atom, expelled as HBr (not drawn)"),
   (":2", "the attacking oxygen"),
   (":3", "the carbon where substitution happens"))),

E("amine_alkylation", "Amine alkylation (halide + amine)",
  "Substitutions & hydrolysis",
  "An amine displaces a halide — chain extension and quaternisation "
  "chemistry (e.g. ionenes).",
  (("alkyl bromide", "C[CH2:3][Br:1]"), ("amine", "CC[NH2:2]")),
  (("secondary amine", "C[CH2:3][NH:2]CC"),),
  ((":1", "the bromine — leaves (as HBr, not drawn)"),
   (":2", "the attacking nitrogen"),
   (":3", "the substitution carbon"))),

E("epichlorohydrin", "Phenol + epichlorohydrin (glycidylation)",
  "Substitutions & hydrolysis",
  "Phenol displaces the chloride of epichlorohydrin, attaching a glycidyl "
  "ether — how DGEBA epoxy resin is made from bisphenol-A.",
  (("phenol", "[OH:7]c1ccccc1"),
   ("epichlorohydrin", "[Cl:8][CH2:4]C1CO1")),
  (("glycidyl ether", "c1ccccc1[O:7][CH2:4]C1CO1"),),
  ((":4", "the CH2 attacked by the phenol oxygen"),
   (":7", "the phenol oxygen"),
   (":8", "the chloride — a leaving atom, expelled as HCl (not drawn)")),
  "C1CO1 at the end is the intact epoxide ring, ready for later cure."),

E("ester_hydrolysis", "Ester hydrolysis", "Substitutions & hydrolysis",
  "Water splits an ester back into acid + alcohol — polyester "
  "degradation.",
  (("ester", "CC(=O)[O:1]CC"), ("water", "[OH2:2]")),
  (("acid", "CC(=O)[OH:2]"), ("alcohol", "CC[OH:1]")),
  ((":1", "the ester bridge O — leaves with the alkyl as the alcohol"),
   (":2", "the water O — becomes the acid's new OH"))),

E("amide_hydrolysis", "Amide hydrolysis", "Substitutions & hydrolysis",
  "Water splits an amide into acid + amine — nylon degradation.",
  (("amide", "CC(=O)[NH:1]CC"), ("water", "[OH2:2]")),
  (("acid", "CC(=O)[OH:2]"), ("amine", "CC[NH2:1]")),
  ((":1", "the amide N — leaves as the free amine"),
   (":2", "the water O — the acid's new OH"))),

E("ester_aminolysis", "Ester aminolysis", "Substitutions & hydrolysis",
  "An amine displaces the alkoxy group of an ester, forming an amide — "
  "how PET can be chemically recycled to useful amides.",
  (("ester", "CC(=O)[O:1]CC"), ("amine", "CC[NH2:2]")),
  (("amide", "CC(=O)[NH:2]CC"), ("alcohol", "CC[OH:1]")),
  ((":1", "the ester O — leaves as the alcohol"),
   (":2", "the amine N — the new amide bond"))),

E("schiff_base", "Imine / Schiff base (aldehyde + amine)",
  "Substitutions & hydrolysis",
  "An aldehyde condenses with a primary amine to an imine (C=N) plus "
  "water — dynamic covalent chemistry, vitrimers, self-healing gels.",
  (("aldehyde", "CC=[O:1]"), ("amine", "CC[NH2:2]")),
  (("imine", "CC=[N:2]CC"), ("water", "O")),
  ((":1", "the carbonyl O — leaves in the water"),
   (":2", "the amine N — replaces it as C=N"))),

E("polycarbonate", "Carbonate link (diaryl carbonate + phenol)",
  "Esters & condensation",
  "A phenol displaces one aryloxide of a diaryl carbonate — the melt "
  "transesterification route to polycarbonate.",
  (("phenyl carbonate", "c1ccccc1[O:1]C(=O)Oc1ccccc1"),
   ("cresol (BPA model)", "[OH:2]c1ccc(C)cc1")),
  (("aryl carbonate", "c1ccccc1OC(=O)[O:2]c1ccc(C)cc1"),
   ("phenol", "[OH:1]c1ccccc1")),
  ((":1", "the displaced aryl-ester oxygen — leaves as free phenol"),
   (":2", "the incoming phenol oxygen — the new carbonate bridge"))),

E("epoxy_secondary_amine", "Epoxide + secondary amine (crosslink step)",
  "Epoxide ring-opening",
  "The second addition in amine cure: the secondary amine formed by the "
  "first addition opens another epoxide, creating the branch point.",
  (("epoxide", "C[CH:1]1[O:2][CH2:3]1"),
   ("secondary amine", "CC[NH:4]CC")),
  (("tertiary amine", "C[CH:1]([OH:2])[CH2:3][N:4](CC)CC"),),
  ((":2", "epoxide O → OH"), (":3", "attacked CH2"),
   (":4", "the nitrogen — now bonded to three carbons: a crosslink "
          "junction"))),

E("methylol_ether", "Methylol urea + alcohol (ether cure)",
  "Phenolic & amino resins",
  "A methylol group condenses with a hydroxyl (e.g. cellulose in wood "
  "panels), expelling water — how UF/MF resins bond to the substrate.",
  (("methylol urea", "NC(=O)N[CH2:1][OH:2]"), ("alcohol", "C[OH:3]")),
  (("ether bridge", "NC(=O)N[CH2:1][O:3]C"), ("water", "O")),
  ((":1", "the bridging CH2"), (":2", "the methylol OH — leaves as water"),
   (":3", "the substrate oxygen"))),

E("carbamate_exchange", "Carbamate exchange (vitrimer)",
  "Isocyanates & polyurethanes",
  "A urethane swaps its alkoxy group with a free alcohol — the dynamic "
  "bond exchange behind PU vitrimers and reprocessable networks.",
  (("urethane", "CCNC(=O)[O:1]CC"), ("alcohol", "C[OH:2]")),
  (("new urethane", "CCNC(=O)[O:2]C"), ("freed alcohol", "CC[OH:1]")),
  ((":1", "the old carbamate oxygen — leaves with its ethyl"),
   (":2", "the incoming alcohol oxygen"))),

E("oxime", "Oxime (ketone + hydroxylamine)",
  "Substitutions & hydrolysis",
  "A ketone condenses with hydroxylamine to an oxime — the crosslinker "
  "chemistry of oxime-cure silicones and PVA gels.",
  (("ketone", "CC(C)=[O:1]"), ("hydroxylamine", "[NH2:2]O")),
  (("oxime", "CC(C)=[N:2]O"), ("water", "[OH2:1]")),
  ((":1", "the carbonyl O — follow it into the water"),
   (":2", "the nitrogen — becomes the C=N"))),
]

# Fix the placeholder acetal entry properly (kept out of the main literal
# for clarity: it has three reactants and a two-step feel, so its SMILES
# deserve the extra care).
EXAMPLES = [e for e in EXAMPLES if e.key != "acetal"]
EXAMPLES.insert(6, E(
  "acetal", "Acetal formation (aldehyde + 2 alcohols)",
  "Esters & condensation",
  "An aldehyde condenses with two alcohols to an acetal plus water — the "
  "chemistry that turns PVA into polyvinyl butyral (PVB) safety-glass "
  "interlayer.",
  (("aldehyde", "C[CH:1]=[O:2]"), ("alcohol A", "C[OH:3]"),
   ("alcohol B", "C[OH:4]")),
  (("acetal", "C[CH:1]([O:3]C)[O:4]C"), ("water", "[OH2:2]")),
  ((":1", "the aldehyde carbon — ends up bonded to BOTH alcohol oxygens"),
   (":2", "the carbonyl oxygen — leaves as water"),
   (":3/:4", "the two alcohol oxygens — the acetal's two O bridges"))))


CATEGORIES: List[str] = []
for _e in EXAMPLES:
    if _e.category not in CATEGORIES:
        CATEGORIES.append(_e.category)


def by_category() -> Dict[str, List[ReactionExample]]:
    out: Dict[str, List[ReactionExample]] = {c: [] for c in CATEGORIES}
    for e in EXAMPLES:
        out[e.category].append(e)
    return out

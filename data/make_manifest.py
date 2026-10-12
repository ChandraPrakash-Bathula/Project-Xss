"""Merge originals + variants + browser results into the single frozen data/manifest.csv.

Row provenance is kept explicit so every table can state its label definition:
  label_kind = original        the 1,000 source rows (not browser-tested here)
             = attack_variant   transformed attacks, browser_confirmed in {0,1}
             = fpr_probe         transformed benign, browser_confirmed = NULL (no exploit)

label_attack_like  : the inherited "is this attack-like?" label  (primary Table 1)
browser_confirmed  : did the marker fire from plain rendering?    (secondary Table 1)
interaction_gated  : fires only on a user event (detector can't see this)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA  # noqa: E402

COLUMNS = ["id", "text", "norm_text", "label_attack_like", "label_kind", "group_id",
           "source", "source_row_id", "license", "family", "parent_id", "endpoint",
           "subtype", "wire_q", "marker_candidate", "expect_hint", "trigger_type",
           "interaction_gated", "browser_confirmed", "browser_confirmed_with_interaction",
           "split"]


def main():
    orig = pd.read_csv(DATA / "originals.csv", keep_default_na=False)
    var = pd.read_csv(DATA / "variants_raw.csv", keep_default_na=False)
    res = pd.read_csv(DATA / "browser_results.csv", keep_default_na=False)

    # originals: fill variant-only columns with blanks/NULL, mark label_kind
    orig = orig.assign(label_kind="original", endpoint="", subtype="", wire_q="",
                       marker_candidate="", expect_hint="", trigger_type="",
                       interaction_gated="NULL", browser_confirmed="",
                       browser_confirmed_with_interaction="NULL")

    # variants: pull the ground-truth columns from the browser run (attack variants only).
    # Everything is kept as strings so originals' blanks/NULL and 0/1 results coexist cleanly.
    for col in ["interaction_gated", "browser_confirmed_with_interaction"]:
        if col not in var:
            var[col] = "NULL"      # filled from results for attack variants below
    is_attack = var.label_kind == "attack_variant"
    for col in ["browser_confirmed", "interaction_gated", "browser_confirmed_with_interaction"]:
        if col in res:
            m = res.set_index("id")[col].astype(str).to_dict()
            var[col] = [str(m.get(i, cur)) if atk else cur
                        for i, cur, atk in zip(var.id, var[col].astype(str), is_attack)]

    # optional second pass (--interact): fills browser_confirmed_with_interaction for gated rows
    inter_path = DATA / "browser_results_interact.csv"
    if inter_path.exists():
        inter = pd.read_csv(inter_path, keep_default_na=False)
        m = inter.set_index("id").browser_confirmed_with_interaction.astype(str).to_dict()
        var["browser_confirmed_with_interaction"] = [
            m.get(i, cur) if atk else cur
            for i, cur, atk in zip(var.id, var.browser_confirmed_with_interaction.astype(str), is_attack)]
        print(f"merged interaction pass: {len(inter)} gated variants")

    # sanity: every attack_variant must have a 0/1 browser_confirmed; fpr stays NULL
    av = var[var.label_kind == "attack_variant"]
    missing = av[~av.browser_confirmed.astype(str).isin(["0", "1"])]
    assert missing.empty, f"{len(missing)} attack variants missing a browser result"
    fpr = var[var.label_kind == "fpr_probe"]
    assert (fpr.browser_confirmed == "NULL").all(), "fpr_probe browser_confirmed must stay NULL"

    manifest = pd.concat([orig[COLUMNS], var[COLUMNS]], ignore_index=True)

    # leakage guard across the whole manifest (originals + variants share group_id)
    leaked = manifest.groupby("group_id").split.nunique()
    assert (leaked <= 1).all(), f"group leak: {list(leaked[leaked > 1].index)[:10]}"

    manifest.to_csv(DATA / "manifest.csv", index=False)
    print(f"manifest rows: {len(manifest)}  -> data/manifest.csv")
    print("\nby label_kind x split:")
    print(manifest.groupby(["label_kind", "split"]).size().unstack(fill_value=0).to_string())
    print("\nattack variants: browser_confirmed by family (TEST split):")
    t = manifest[(manifest.label_kind == "attack_variant") & (manifest.split == "test")]
    t = t.assign(bc=t.browser_confirmed.astype(int))
    print(t.groupby("family").bc.agg(confirmed="sum", n="count").to_string())


if __name__ == "__main__":
    main()

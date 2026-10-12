"""Three transformation families for the robustness test (Step 5), marker-only.

    transport(text)  URL-encoding (single, and double for the double-decode endpoint)
    entity(text)     HTML-entity representation
    js_obf(text)     JavaScript obfuscation of the harmless marker

Each returns a list of Variant(wire_q, endpoint, subtype, expect_hint):
  wire_q      the exact `q` value to put on the URL  ->  endpoint?q=<wire_q>
  endpoint    the ONE endpoint this variant is meant to be tested against
  subtype     which concrete trick was applied (for the error table)
  expect_hint author's guess at whether it executes ('yes'/'no'/'?'), NOT ground truth

Ground truth (`browser_confirmed`) is decided later by verify/browser_check.py in a real
browser. This file never sets it.

The detector-visible `text` for every variant is derived from wire_q with the SAME
decode step the proxy uses (common.decode_once), so training, the manifest and serving
agree by construction. That is also why transport single-encoding shows NO shift at the
detector (it decodes back to the original) while double-encoding does.

All strings use the harmless marker `window.__xss_fired=true`; nothing here produces a
real exploit. Run only against the local owned app.
"""
import re
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, MARKER, decode_once, norm  # noqa: E402

# --- marker rewrite: alert/prompt/confirm(...) -> harmless flag ----------------------
_CALL = re.compile(r"(?:alert|prompt|confirm)\s*\(\s*[^()]*\)", re.I)
_MARK_ASSIGN = "window.__xss_fired=true"   # the JS the marker expands to


def to_marker(s):
    """Rewrite the first alert/prompt/confirm call to the marker. None if not convertible."""
    if not _CALL.search(s):
        return None
    return _CALL.sub(_MARK_ASSIGN, s)


def natural_endpoint(m):
    """Where a tag-style payload most plausibly lands by default."""
    ml = m.lstrip().lower()
    if ml[:1] in ('"', "'", ">"):
        return "/reflect-attr"      # looks like an attribute-value breakout
    return "/reflect-html"          # the common case: a full tag in the HTML body


# Handlers that fire during plain rendering (no user action). onfocus only auto-fires
# with autofocus, handled below. This is a HINT for reporting; the browser decides truth.
_AUTO_FIRE = {"load", "error", "toggle", "pageshow", "animationstart", "animationend",
              "transitionend", "loadstart", "canplay", "playing", "loadeddata"}
_EVENT_HANDLER = re.compile(r"\bon([a-z]+)\s*=", re.I)


def trigger_type(effective_text):
    """Classify how the *rendered* payload would fire: on_load / on_event:onX / unknown.

    Pass the payload as it reaches the page after the endpoint's decoding (for transport
    that is the fully-decoded marker form, not the obfuscated detector text).
    """
    tl = effective_text.lower()
    m = _EVENT_HANDLER.search(tl)
    if m:
        h = m.group(1)
        if h in _AUTO_FIRE:
            return "on_load"
        if h == "focus":
            return "on_load" if "autofocus" in tl else "on_event:onfocus"
        return f"on_event:on{h}"
    if "<script" in tl or "javascript:" in tl or tl.lstrip().startswith('";'):
        return "on_load"            # inline script / js-context breakout runs on parse
    return "unknown"


# --- helper: a variant as (wire_q, endpoint, subtype, expect_hint) --------------------
class V(tuple):
    def __new__(cls, wire_q, endpoint, subtype, expect_hint):
        return super().__new__(cls, (wire_q, endpoint, subtype, expect_hint))


# ======================================================================================
# transport: URL-encoding
# ======================================================================================
_SPECIAL = {"<": "%3C", ">": "%3E", '"': "%22", "'": "%27", "=": "%3D", " ": "%20"}
# must also be escaped for the value to survive decode_once (parse_qsl) intact
_WIRE_SAFE = {"&": "%26", "+": "%2B", "%": "%25", "#": "%23"}


def _single_encode(m):
    out = []
    for ch in m:
        if ch in _WIRE_SAFE:
            out.append(_WIRE_SAFE[ch])
        elif ch in _SPECIAL:
            out.append(_SPECIAL[ch])
        else:
            out.append(ch)
    return "".join(out)


def transport(text):
    single = _single_encode(text)            # Flask decodes once -> original; fires on /reflect-html
    double = single.replace("%", "%25")      # detector sees %3C...; only /reflect-decode re-decodes
    return [
        V(single, "/reflect-html", "single_urlencode", "yes"),
        V(double, "/reflect-decode", "double_urlencode", "yes"),
    ]


# ======================================================================================
# entity: HTML-entity representation
# ======================================================================================
def _decimal_entities(s, chars):
    return "".join(f"&#{ord(c)};" if c in chars else c for c in s)


def _entity_body(text):
    """Pure representation change: entity-encode the string's own '<>&\"' characters.

    No injection — safe to apply to benign text for FPR probing. On /reflect-html this
    renders as visible TEXT (entities are not re-parsed as markup), so for an attack it is
    a labeled expected-NON-executing case; for benign it is just an obfuscated-looking string.
    """
    html_text = _decimal_entities(text, set('<>&"'))
    return V(quote(html_text, safe=""), "/reflect-html", "entity_in_html_body", "no")


# on<event>= followed by its value, either quoted or bare up to the next space or '>'
_HANDLER_VALUE = re.compile(r"""(\bon[a-z]+\s*=\s*)(?:(")([^"]*)"|(')([^']*)'|([^\s>]+))""", re.I)


def _entity_in_handler(text):
    """Entity-encode letters INSIDE an event handler's value, keeping the tag intact.

    The HTML parser decodes entities in an attribute value before JavaScript ever sees it,
    so `onerror=w&#105;ndow...` still runs while the n-grams the detector keyed on are
    broken up. Unlike the fixed breakout below this is derived from the PARENT payload, so
    every parent yields a different string. That matters: a transformation that emits one
    constant string gives an evaluation set of size one no matter how many rows it fills.
    """
    m = _HANDLER_VALUE.search(text)
    if not m:
        return None
    prefix = m.group(1)
    quote_ch = m.group(2) or m.group(4) or ""
    value = m.group(3) if m.group(3) is not None else (
        m.group(5) if m.group(5) is not None else m.group(6))
    if not value:
        return None
    enc = _decimal_entities(value, set("aeiouwl"))        # vowels plus w and l
    rebuilt = text[:m.start()] + prefix + quote_ch + enc + quote_ch + text[m.end():]
    return V(quote(rebuilt, safe=""), natural_endpoint(text), "entity_in_handler", "yes")


def entity(text):
    variants = [_entity_body(text)]

    # (b) parent-derived: encode inside the payload's own handler value
    h = _entity_in_handler(text)
    if h:
        variants.append(h)

    # (c) fixed attribute breakout. Kept because it reliably executes, but it is the same
    #     string for every parent, so evaluation code must count distinct strings not rows.
    handler_value = _decimal_entities(_MARK_ASSIGN, set("aeiouwl"))
    attr_breakout = '"><svg onload=' + handler_value + ">"
    variants.append(V(quote(attr_breakout, safe=""), "/reflect-attr", "entity_in_attr_value", "?"))
    return variants


# ======================================================================================
# js_obf: JavaScript obfuscation of the marker
# ======================================================================================
def _hex_name(name):
    return "".join(f"\\x{ord(c):02x}" for c in name)   # e.g. __xss_fired -> \x5f\x5f...


def _obfuscate_marker():
    return {
        "bracket": "window['__xss_fired']=true",
        "concat": "window['__xss'+'_fired']=true",
        "hex": "window['" + _hex_name("__xss_fired") + "']=true",
    }


def js_obf(text):
    """Obfuscate the marker's JS, both in-place in the tag and as a string-context breakout."""
    variants = []
    obf = _obfuscate_marker()
    ep = natural_endpoint(text)
    for name, code in obf.items():
        # (a) in-place: swap the plain marker inside the original tag/handler
        if _MARK_ASSIGN in text:
            inplace = text.replace(_MARK_ASSIGN, code)
            variants.append(V(quote(inplace, safe=""), ep, f"js_inplace_{name}", "yes"))
    # (b) string-context breakout for /reflect-js: close the "..." , run code, comment rest
    brk = '";' + obf["bracket"] + ';//'
    variants.append(V(quote(brk, safe=""), "/reflect-js", "js_string_breakout", "yes"))
    return variants


FAMILY_FN = {"transport": transport, "entity": entity, "js": js_obf}

# Benign FPR probes use only pure representation changes (no injection, no marker).
BENIGN_FN = {"transport": transport, "entity": lambda t: [_entity_body(t)]}

# Synthetic variants whose firing mechanism is fixed by construction, not the parent.
_FIXED_TRIGGER = {"entity_in_attr_value": "on_load", "js_string_breakout": "on_load"}

COLUMNS = ["id", "text", "norm_text", "label_attack_like", "label_kind", "group_id",
           "source", "source_row_id", "license", "family", "parent_id", "endpoint",
           "subtype", "wire_q", "marker_candidate", "expect_hint", "trigger_type",
           "split", "browser_confirmed"]


def _row(r, family, k, wire_q, endpoint, subtype, expect, *, label_kind, attack_like,
         marker_candidate, effective_for_trigger):
    text = decode_once("q=" + wire_q)              # EXACTLY what the proxy will see
    if label_kind == "fpr_probe":
        trig = "n/a"                               # no exploit -> nothing to fire
    else:
        trig = _FIXED_TRIGGER.get(subtype) or trigger_type(effective_for_trigger)
    return {
        # keyed on subtype, not on position: adding a new transformation must never
        # renumber the existing ones, or previously recorded browser results silently
        # attach to the wrong rows
        "id": f"{r.id}-{family}-{subtype}",
        "text": text,
        "norm_text": norm(text),
        "label_attack_like": attack_like,
        "label_kind": label_kind,                  # 'attack_variant' | 'fpr_probe'
        "group_id": r.group_id,                    # inherited -> never crosses splits
        "source": r.source,
        "source_row_id": r.source_row_id,
        "license": r.license,
        "family": family,
        "parent_id": r.id,
        "endpoint": endpoint,
        "subtype": subtype,
        "wire_q": wire_q,                          # browser_check builds endpoint?q=<wire_q>
        "marker_candidate": marker_candidate,
        "expect_hint": expect,
        "trigger_type": trig,
        "split": r.split,                          # inherited from the parent original
        # NULL for benign (no exploit to confirm); set by browser_check.py for attacks.
        "browser_confirmed": "" if label_kind != "fpr_probe" else "NULL",
    }


# ======================================================================================
# main: originals.csv -> data/variants_raw.csv  (browser_confirmed left empty)
# ======================================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", action="store_true",
                    help="also emit transport+entity FPR probes from benign originals")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "originals.csv", keep_default_na=False)
    attacks = df[df.label_attack_like == 1].copy()
    attacks["marker_text"] = attacks.text.map(to_marker)
    cand = attacks[attacks.marker_text.notna()].copy()
    n_attacks, n_cand = len(attacks), len(cand)

    rows = []
    for _, r in cand.iterrows():
        m = r.marker_text                          # fully-decoded form that reaches the page
        for family, fn in FAMILY_FN.items():
            for k, (wire_q, endpoint, subtype, expect) in enumerate(fn(m)):
                rows.append(_row(r, family, k, wire_q, endpoint, subtype, expect,
                                 label_kind="attack_variant", attack_like=1,
                                 marker_candidate=1, effective_for_trigger=m))

    n_benign = 0
    if args.benign:
        benign = df[df.label_attack_like == 0]
        n_benign = len(benign)
        for _, r in benign.iterrows():
            for family, fn in BENIGN_FN.items():
                for k, (wire_q, endpoint, subtype, expect) in enumerate(fn(r.text)):
                    rows.append(_row(r, family, k, wire_q, endpoint, subtype, "n/a",
                                     label_kind="fpr_probe", attack_like=0,
                                     marker_candidate=0, effective_for_trigger=r.text))

    out = pd.DataFrame(rows)[COLUMNS]
    out.to_csv(DATA / "variants_raw.csv", index=False)

    print(f"attack originals:            {n_attacks}")
    print(f"marker-convertible (syntax): {n_cand}  ({n_cand / n_attacks:.0%})")
    if args.benign:
        print(f"benign originals (FPR probe):{n_benign}")
    print(f"variants written:            {len(out)}  -> data/variants_raw.csv")
    print("\nby label_kind x family:")
    print(out.groupby(["label_kind", "family"]).size().to_string())
    print("\nattack variants by trigger_type:")
    print(out[out.label_kind == "attack_variant"].trigger_type.value_counts().to_string())
    print("\nby split x label_kind:")
    print(out.groupby(["split", "label_kind"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()

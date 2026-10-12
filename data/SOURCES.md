# Data sources and provenance

All sources retrieved **2026-10-06**. Raw downloads are kept unmodified under `data/raw/`.
The 1,000 originals in `data/originals.csv` are produced from these by `data/build_dataset.py`
(seed 42); `data/build_stats.json` records how many rows each cleaning step removed.

## Sources used

| Source key | Dataset | Retrieved from | Licence | Rows taken |
|---|---|---|---|---|
| `kaggle` | Cross-Site Scripting (XSS) dataset for Deep Learning | `https://www.kaggle.com/api/v1/datasets/download/syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning` → `data/raw/kaggle/XSS_dataset.csv` | Not stated on the dataset page; see note below | 350 attack-like, 300 benign |
| `payloadbox` | PayloadBox XSS payload list, as mirrored in SecLists (`Fuzzing/XSS/robot-friendly/XSS-payloadbox.txt`) | `https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/XSS/robot-friendly/XSS-payloadbox.txt` → `data/raw/seclists/` | MIT (`data/raw/seclists/LICENSE`) | 150 attack-like |
| `mendeley` | Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection, v2 | `https://data.mendeley.com/datasets/py6cnbgdh2` (DOI `10.17632/py6cnbgdh2.2`) → `data/raw/mendeley/xss_dataset.csv` | **CC BY 4.0** | 150 benign (benign rows only) |
| `fixture` | Hand-written tricky-benign strings | `data/fixtures_benign.txt` (written for this project) | Own work | 50 benign |

**Total: 500 attack-like + 500 benign = 1,000 originals.**

### Citation required by the Mendeley licence

CC BY 4.0 requires attribution. Cite as:

> Large-Scale Annotated Dataset for Cross-Site Scripting (XSS) Attack Detection,
> Mendeley Data, V2, doi: 10.17632/py6cnbgdh2.2.

### Substitutions and caveats

- **PayloadBox was substituted.** The original repository `github.com/payloadbox/xss-payload-list`
  returned HTTP 404 on 2026-10-06. The copy distributed with SecLists (MIT) was used instead.
  The payload content is the same list; the licence cited is SecLists'.
- **The Kaggle dataset page states no explicit licence.** It is used here for non-commercial
  academic coursework on a local machine only. If a licence is required for submission,
  re-run `build_dataset.py` after re-balancing the quotas onto the CC BY 4.0 and MIT sources;
  only the numbers in `QUOTA` change.
- **The obfuscated / LLM-augmented Mendeley dataset was deliberately NOT used.** Rows that are
  already obfuscated would act as pre-made "seen" transformations and would invalidate the
  held-out-family test.
- **The Mendeley benign pool is templated.** 613,888 benign rows reduce to ~42,000 distinct
  skeletons. Rows that already looked encoded (`%3C`, `&#`, `\x`, long base64 runs) or carried
  the dataset's synthetic placeholder tokens (`LT_…`, `GT_…`, `SCHEME_…`) were dropped:
  52,596 rows removed before sampling.
- **The Kaggle benign rows are mostly Wikipedia HTML fragments**, not user input. Because
  attack and benign rows come from different sources, the model could learn source style
  rather than attack structure. This is why `ml/evaluate.py` reports per-source and
  per-subtype breakdowns, and why the 50 hand-written fixtures are included.

## Known limitation: pre-encoded payloads in the attack pool

The 50-row label audit (`data/audit.py`, results in `data/audit.csv`) found that some
payloadbox rows arrive **already HTML-entity-encoded** in the source, e.g.
`&lt;IMG STYLE=&quot;xss:expr/*XSS*/ession(alert(&apos;XSS&apos;))&quot;&gt;`.

- **34 of 500 attack originals (6.8%)** are affected — all from payloadbox, where they are
  **22.7%** of that source. No Kaggle attack row is affected.
- The `ENCODED` filter in `build_dataset.py` was applied to the Mendeley **benign** pool
  (to stop pre-made "seen" transformations weakening the held-out test) but **not** to the
  attack pool. That asymmetry is the cause.
- Consequence: a small share of rows labelled `family=original` are effectively already
  entity-transformed, and because they render as text when reflected, they can be
  `attack_like=1` yet never `browser_confirmed=1`.

**Measured impact** (`ml/sensitivity.py` → `logs/sensitivity_preencoded.csv`): Table 1 was
recomputed with all 34 rows and their 128 descendant variants removed.

| Model | Condition | recall as published | recall without | delta |
|---|---|---|---|---|
| baseline | ordinary | 0.950 | 0.968 | +0.018 |
| baseline | seen transforms | 0.601 | 0.596 | −0.005 |
| baseline | held-out (js) | 0.728 | 0.728 | −0.001 |
| augmented | ordinary | 0.950 | 0.978 | +0.028 |
| augmented | seen / held-out | 1.000 | 1.000 | 0.000 |

False-positive rates are identical in every condition. The largest shift is 0.028, and the
headline ordinary→transformed gap is slightly **wider** without these rows (0.968→0.596),
so no conclusion depends on them. The benchmark is therefore **reported as built**, with
this stated as a limitation rather than silently re-run.

To remove them instead, apply the existing `ENCODED` regex to the payloadbox pool in
`data/build_dataset.py` (1,043 cleaned rows are available, so the quota of 150 is still
reachable) and re-run the pipeline from `build_dataset.py` onward, including browser
verification.

## Deviation from the brief

The brief asks for **200–500** labeled inputs; this benchmark uses **1,000 originals**
(before transformed variants) so the test split holds ~200 originals and confidence
intervals are usable. Variants are **not** counted toward that total. To fall back to the
suggested range, change `QUOTA` in `data/build_dataset.py` to halve each entry and re-run;
nothing else changes.

## Cleaning applied (counts from `data/build_stats.json`)

1. Non-printable / non-text rows dropped; kept 3–500 characters → 629,554 remain.
2. Rows whose normalised text carried **conflicting labels** across sources dropped: **7**.
3. Exact duplicates after normalising case and whitespace: **6,534** removed.
4. Near-duplicates within a source (same skeleton: quoted strings → `S`, digits → `0`):
   **570,144** removed, so a quota is not filled with copies of one template.
5. Quota sampling per (source, label) with `random_state=42`.
6. `group_id` assigned **after** pooling all sources, so near-copies from different sources
   land in the same group: 998 groups over 1,000 rows (2 groups have more than one member).

Every transformed variant inherits its parent's `group_id` **and** `split`, so no group
crosses a split boundary (asserted in `ml/split.py` and again in `data/make_manifest.py`).

## Reproducing

```bash
python data/build_dataset.py     # -> data/originals.csv, data/build_stats.json
python ml/split.py               # -> split column + data/split_ids_seed42.csv
python data/audit.py             # -> data/audit.csv  (50-row manual label check)
```

# What does an XSS detector actually detect?

An execution-grounded study of machine-learning XSS detection placed in network traffic.

Published XSS detectors report accuracies above 99%. Those numbers are computed against a
label that means *"someone called this string an attack"*. This study asks a different
question: does that label track whether the payload **actually executes in a browser**?

Everything runs on `127.0.0.1` against applications in this repository. The only payload
used anywhere is an inert marker, `window.__xss_fired = true`, which sets a variable and
does nothing else. No real exploit code is involved.

---

## Headline findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Escaping the output eliminated execution.** Not reduced: eliminated. | **0 of 4,459**, 95% upper bound **0.067%** (rule of three), across 2 local apps, 4 third-party template libraries, unmodified public data, and 3 browser engines |
| 2 | **Synthetic event dispatch overstates exploitability by 27%.** The standard verification shortcut is optimistic, and the error is one-sided. | 0.787 confirmation, 95% Wilson CI [0.714, 0.845], n=150 |
| 3 | **No single inspection point covers all three XSS classes.** | reflected: either side; stored: response only; DOM: request only (response AUC exactly **0.500**) |
| 4 | **A production WAF ruleset barely discriminates.** | OWASP CRS flags **0.916** of executing payloads and **0.851** of inert ones |
| 5 | **Only ~2% of attack-labelled payloads execute without a user action.** | consistent across all four settings |
| 6 | **Public payload lists are partly dead.** | 62% of 500 unmodified public rows execute anywhere; PayloadBox **75% inert** in current Chromium |
| 7 | **Detectors score near chance at predicting execution.** | 0.38–0.51 AUC across 10 architectures, while scoring 0.79–1.00 on the labelled task |
| 8 | **Inline detection is cheap.** | +0.5 ms median; throughput unchanged (~711 rps with and without) |

---

## Method

### Two labels, never mixed

| label | meaning | source |
|---|---|---|
| `attack_like` | a public dataset called this an attack | inherited |
| `browser_confirmed` | a real browser set the marker | measured |

Keeping these apart is the whole study. Most of the literature reports metrics against
the first and interprets them as if they were the second.

### Architecture

Three processes, because the boundary *is* the experiment. A request-side detector must
be physically confined to the request, otherwise it is not a request-side detector.

```
browser / load client  ->  reverse proxy :8080  ->  Flask app :5010
                           (detector lives here)    (the target)
```

| port | process | role |
|---|---|---|
| 5010 | `app/vuln_app.py` | target app 1: 5 endpoints, 5 output contexts |
| 5002 | `app/vuln_app2.py` | target app 2: 7 endpoints incl. stored, DOM, partial sanitiser |
| 5020 | `thirdparty/engines_app.js` | 9 sinks from ejs / handlebars / mustache / pug |
| 8080 | `proxy/proxy.py` | detector, modes `off` / `observe` / `block` |
| 5001 | `teach/playground.py` | demo and results UI |
| 8090 | `bench/loadtest.py` | benchmark's own proxy, so timing never disturbs 8080 |

### Benchmark

1,000 deduplicated originals (500 attack-like, 500 benign) from four sources, expanded to
6,377 rows with transformed variants.

| source | attack | benign | licence |
|---|---|---|---|
| Kaggle XSS dataset | 350 | 300 | not stated (see `data/SOURCES.md`) |
| PayloadBox (via SecLists) | 150 | 0 | MIT |
| Mendeley large-scale XSS v2 | 0 | 150 | CC BY 4.0 |
| hand-written tricky benign | 0 | 50 | own work |

Cleaning removed 6,534 exact duplicates and 570,144 within-source near-duplicates.
Near-duplicates are grouped by a *skeleton* key (digits and quoted strings replaced), and
`group_id` is assigned **after** pooling all sources so near-copies from different sources
land together. Every variant inherits its parent's group and split, and no group crosses a
split boundary (asserted in code).

### Execution verification

Each payload is loaded in a real browser at its intended sink, then `window.__xss_fired`
is read. Public payloads using `alert()` are adjudicated by observing the dialog instead.

---

## Experiments and results

### 1. Detection on the conventional label (`table1_primary`)

Baseline char n-gram + logistic regression, threshold 0.617 chosen on validation.

| condition | precision | recall | FPR |
|---|---|---|---|
| ordinary inputs | 0.990 | 0.950 | 0.010 |
| marker-plain control | 0.989 | 0.957 | 0.010 |
| seen transformations | 0.990 | 0.667 | 0.010 |
| held-out family (js) | 0.989 | 0.728 | 0.010 |

The control row confirms the drop is caused by transformation, not by swapping `alert()`
for the marker.

### 2. Does the score predict execution? (`table3_execution_auc`)

| architecture | AUC attack-vs-benign | **AUC executes-vs-inert** | 95% CI |
|---|---|---|---|
| char n-gram LR | 0.999 | 0.439 | [0.396, 0.481] |
| char n-gram SVM | 1.000 | 0.467 | [0.423, 0.510] |
| word/token LR | 0.989 | 0.473 | [0.428, 0.520] |
| handcrafted + RF | 0.920 | 0.388 | [0.345, 0.434] |
| handcrafted + GB | 0.890 | 0.382 | [0.337, 0.427] |
| char + handcrafted hybrid | 0.794 | 0.453 | [0.408, 0.495] |
| DistilRoBERTa (3 seeds) | 0.987 | 0.377 ± 0.077 | — |
| CodeBERT (3 seeds) | 0.997 | 0.508 ± 0.082 | — |

Near-perfect at the labelled task, near chance at exploitability. Holds for handcrafted
features, which have no payload vocabulary to memorise, and for a 125M-parameter model
pretrained on real code.

**Caveat, measured:** this result is split-sensitive. Across 3 data splits the per-model
range is up to 0.23 (`split_sensitivity_table3.csv`), so the honest claim is *near chance*,
not *below chance*.

### 3. Execution depends on the output context (`table17_corrected_rates`)

Same payloads, every sink. Rates below are **corrected** by the real-input factor (§5).

| sink | reported | corrected | 95% CI |
|---|---|---|---|
| escaped text (all settings) | 0.000 | **0.000** | exact |
| JS string literal | 0.003 | 0.003 | — |
| quoted attribute | 0.111 | 0.088 | [0.080, 0.094] |
| raw HTML body | 0.440 | 0.352 | [0.322, 0.376] |
| double-decoded HTML | 0.532 | 0.426 | [0.390, 0.455] |
| stored (app 2) | 0.463 | 0.369 | [0.337, 0.394] |
| DOM `innerHTML` (app 2) | 0.447 | 0.355 | [0.324, 0.380] |
| ejs / handlebars / mustache / pug raw | 0.445 | 0.354 | [0.324, 0.379] |
| ejs / handlebars / mustache / pug escaped | 0.000 | **0.000** | exact |

47–55% of payloads change outcome depending only on where the value lands.

### 4. What the output context is worth (`table4_context_crossed`, `table11`)

Payload and sink are decorrelated by construction (every payload sent to every sink).

| model sees | AUC | AvgPrec |
|---|---|---|
| payload only (a proxy's view) | 0.801 | 0.461 |
| **sink only, payload discarded** | **0.858** | 0.496 |
| payload + sink | **0.974** | 0.876 |

Knowing *where* the value lands predicts exploitability better than knowing *what* it is.
Replicated on transformers: CodeBERT 0.799 → 0.970, DistilRoBERTa 0.801 → 0.971.

**The gain depends on the definition of "executes":**

| definition | payload | +sink | gain | test positives |
|---|---|---|---|---|
| render or any interaction | 0.801 | 0.974 | +0.172 | 372 |
| render or *plausible* interaction | 0.889 | 0.989 | **+0.100** | 176 |
| plausible interaction only | 0.901 | 0.992 | +0.090 | 158 |
| render only (strictest) | 0.962 | 0.986 | +0.024 | 18 |

The effect comes mostly from payloads needing a user action. Under the strictest reading
there are too few positives to claim anything. **+0.100 is the figure we would defend.**

### 5. Is synthetic event dispatch valid? (`table13_real_input`)

Same payloads, synthetic `dispatchEvent` versus Playwright's real mouse and keyboard.

| | real: fires | real: no |
|---|---|---|
| **synthetic: fires** | 118 | **32** |
| **synthetic: no** | 0 | 0 |

**78.7% confirmation** [0.714, 0.845], n=150. The error is strictly one-sided: synthetic
never finds an exploit real input misses. Failures are elements with no clickable surface
(`iframe`, `param`, `script`, `template`, `wbr`), which `dispatchEvent` reaches because it
skips hit-testing.

This is why the escaped-sink zero needs no correction: synthetic can only over-report.

### 6. Which side of the wire carries the answer? (`table15`, `table16`)

| what the detector sees | AUC | AvgPrec |
|---|---|---|
| request payload only | 0.802 | 0.462 |
| request + sink label (oracle) | 0.974 | 0.878 |
| **response body only** | **0.979** | **0.904** |

The response body matches the sink oracle without needing one, because it contains the
payload *already placed in its rendering context*.

Per vulnerability class (app 2):

| class | request | response | response + XHR | payload located in |
|---|---|---|---|---|
| reflected | 0.917 | 0.915 | 0.915 | both |
| **stored** | 0.917 | **0.915** | 0.915 | **not in the delivering request** |
| **DOM** | 0.896 | **0.500** | 0.886 | **not in the response** |

DOM collapses to exactly chance on the response, because the server's page contains none
of the payload. No single inspection point covers all three classes.

### 7. Published detectors (`table8_published`)

Not our models: the real OWASP ModSecurity Core Rule Set XSS rules (Apache-2.0, 28 usable
`@rx` patterns) and a reimplementation of the random forest from Miczek et al.

| detector | trained on | AUC (execution) | catches executing | flags inert |
|---|---|---|---|---|
| CRS regex | nothing (fixed ruleset) | 0.582 ± 0.049 | **0.916** | **0.851** |
| Miczek-style RF | attack-likeness | 0.442 ± 0.063 | 0.179 | 0.324 |
| Miczek-style RF | execution | 0.773 ± 0.017 | 0.167 | 0.048 |

A deployed WAF ruleset separates executing from inert payloads by 6 points.

### 8. False positives on real traffic (`table9_fpr_real_traffic`)

19,195 distinct parameter values from the CSIC 2010 normal-traffic corpus.

| detector | FPR |
|---|---|
| hybrid, CRS regex, hand_rf | 0.000–0.0002 |
| char n-gram LR / SVM, hand_gb | 0.001–0.002 |
| word LR | 0.014 |
| **char-CNN** | **0.183** |

Most detectors do *better* on real traffic than on our synthetic benign set. The char-CNN
flags 18% of entirely normal traffic, which our synthetic benign set had hidden.

### 9. Cost of inline detection (`table2_performance`)

3 proxy configs × 2 loads × 3 runs, observe-only mode, warm-up discarded.

| config | load | median ms | p95 ms | completed rps |
|---|---|---|---|---|
| detection off | 100 / 800 | 0.94 / 0.77 | 1.87 / 6.37 | 89 / 711 |
| baseline | 100 / 800 | 1.62 / 1.25 | 8.45 / 4.23 | 89 / 711 |
| augmented | 100 / 800 | 1.72 / 1.34 | 8.90 / 40.93 | 89 / 699 |

Median cost ~0.5 ms, well outside run-to-run spread. Throughput plateaus at ~711 rps
*including with detection off*, so the ceiling is the upstream app, not inference. This is
client-observed response time over loopback, **not** network latency.

### 10. Public benchmark audit (`benchmark_audit.csv`)

500 unmodified rows straight from the public datasets, adjudicated by dialog observation.

| source | adjudicable | fired in **any** of 5 contexts | inert |
|---|---|---|---|
| Kaggle | 297/300 (99%) | 233/297 (78%) | 22% |
| PayloadBox | 134/200 (67%) | 33/134 (25%) | **75%** |

Kaggle's labels are largely honest. PayloadBox is three-quarters dead in a current browser.
88% of all observed firings required a user interaction.

### 11. Does the answer depend on the browser? (`table14_cross_browser`)

Every other execution label in this study comes from Chromium. Two claims ride on that:
that escaping prevents execution, and that many public payload rows are inert. If those
vectors were merely *Chromium*-dead, the second claim would be about one engine.

200 payloads × 5 sinks × 3 engines = 3,000 checks.

| sink | chromium | firefox | webkit |
|---|---|---|---|
| **escaped text** | **0.000** | **0.000** | **0.000** |
| JS string literal | 0.005 | 0.005 | 0.005 |
| quoted attribute | 0.095 | 0.120 | 0.120 |
| raw HTML body | 0.455 | 0.425 | 0.465 |
| double-decoded HTML | 0.560 | 0.530 | 0.580 |
| **overall** | **0.223** | **0.216** | **0.234** |

**Engines agree on 96.0%** of 1,000 payload-sink pairs. Of the 40 disagreements, exactly
**one** payload executed in a single engine and nowhere else, so the differences are
marginal cases rather than engine-specific vectors.

Three consequences:

- The escaped-sink zero is **not a Chromium artifact**: 0 of 600 across three independent
  engines, which brings the pooled total to **0 of 4,459**.
- **Chromium is not an unusually strict engine.** It sits between Firefox and WebKit, in a
  spread of 0.018, so the Chromium-only labels used elsewhere are representative rather
  than conservative or inflated.
- **The sink ordering is identical in all three engines** (escaped < JS literal < attribute
  < raw < double-decoded). What drives execution is the output context, not the engine.

### 12. Negative results worth reporting

- **Augmentation did not generalise.** Apparent perfect recall collapsed from 1.000 to
  0.726 once marker identifiers were renamed, level with the baseline's 0.709
  (`table1_shortcut_probe`). FPR rose 0.010 → 0.023.
- **Per-application calibration does not help on novel sink types.** Adding in-app examples
  moves the sink's contribution from −0.052 to 0.000, never positive
  (`table12b_calibration_novel_sinks`). The payload alone already scores 0.989 there,
  because all three novel sinks render raw.
- **Context transfer degrades across applications:** +0.157 within app, +0.081 cross-app,
  −0.052 on unseen sink types.

---

## Reproducing

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
./venv/bin/playwright install chromium firefox webkit

APP_PORT=5010 ./run.sh start          # app, proxy, playground
./venv/bin/python app/vuln_app2.py &  # app 2
cd thirdparty && npm install && node engines_app.js &
```

| step | command |
|---|---|
| build dataset | `python data/build_dataset.py` |
| split (asserts no group leak) | `python ml/split.py` |
| train one detector | `PYTHONPATH=. python ml/train.py --arch char_lr` |
| verify execution | `python verify/browser_check.py` |
| crossed payload × sink | `python verify/cross_context.py` |
| real-input validation | `python verify/real_input_check.py --n 150` |
| cross-browser (3 engines) | `python verify/cross_browser.py --n 200` |
| detection tables | `PYTHONPATH=. python ml/evaluate.py baseline augmented` |
| published baselines | `PYTHONPATH=. python ml/published_baselines.py` |
| timing | `python bench/loadtest.py --low 100 --high 800` |

Results land in `logs/`. The playground at `http://127.0.0.1:5001/` shows the paired-context
demo; `/results` renders every table.

**Not in this repository:** `data/raw/` (one source file is 177 MB, and the Kaggle licence
is unstated). All URLs, DOIs and retrieval dates are in `data/SOURCES.md`.

---

## Limitations

- **Cross-browser is sampled, not exhaustive** (200 payloads x 5 sinks x 3 engines). The
  three engines agreed on 96.0% of pairs, so the Chromium-based labels used elsewhere look
  representative, but every other experiment still uses Chromium only.
- **Thin third-party application evidence.** OWASP Juice Shop contributed 2 validated sinks
  and 6 positives. The template-engine harness externalises escaping decisions to real
  libraries, but we still wrote the surrounding page.
- **Reflected XSS is the main experiment.** Stored and DOM are measured on app 2 only.
- **`attack_like` labels are inherited**, with a 50-row manual audit (`data/audit.csv`):
  0 wrong, 4 unclear, all from PayloadBox.
- **6.8% of attack originals arrive pre-encoded**, all from PayloadBox. Sensitivity analysis
  (`sensitivity_preencoded.csv`) shows a largest effect of 0.028 on any conclusion.
- **One machine** for all timing numbers.
- Four claims were retracted during this work after further testing. See the negative
  results above; the surviving claims are those that withstood split, seed, architecture,
  application and definition changes.

## Ethics

Everything is local, against applications in this repository, with an inert marker. No
third-party systems were tested. The vulnerable apps bind to `127.0.0.1` and must not be
deployed.

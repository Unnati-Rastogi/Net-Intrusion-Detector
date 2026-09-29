# Alert Prioritization

**Status:** implemented in `src/nids/prioritize.py` (offline report:
`reports/alert_prioritization.json`). The API serves that committed report
verbatim via `GET /alerts` — nothing is re-ranked at request time — and
the per-record analyst workflow `POST /analyze` applies the same
risk-level → P1–P4 mapping. Event grouping and incident correlation
remain later stages.

## 1. Why prioritization is needed

A detector that flags ~1.9% of traffic still produces hundreds of alerts in
a realistic window. An analyst cannot review them in arbitrary order. This
layer answers exactly one question: **in what order should model-generated
alerts be reviewed?**

## 2. Detection vs prioritization

- **Detection** (the model + risk layer) decides *that* a record deserves
  attention and *how firmly* the evidence supports it.
- **Prioritization** (this layer) only *orders* that attention. It adds no
  new signal, no new score, and no judgement about what an attack would
  actually do.

## 3. Risk level → review priority mapping

| Risk level | Review priority | Meaning |
|---|---|---|
| CRITICAL | **P1** | highest analyst review priority |
| HIGH | **P2** | review next |
| MEDIUM | **P3** | review when P1/P2 are clear |
| LOW | **P4** | review backlog |

The mapping exists so a queue can be organized by how firmly the model's
own evidence supports each alert. **P1 means "highest analyst review
priority" — never "most dangerous attack".** Impact, scope, and asset value
are not modelled anywhere in this pipeline.

## 4. Ordering and tie-breaking rules

Applied in order; the result is a total order (deterministic, no
randomness):

1. `risk_score` **descending** — the primary signal (consensus-weighted
   model risk from `risk.py`, reused verbatim).
2. `anomaly_score` **descending** — equal risk, higher model score first.
3. `risk_level` by explicit priority rank (CRITICAL > HIGH > MEDIUM > LOW)
   — kept as an explicit, documented rule and an integrity guard. Because
   the input validation requires `risk_level == risk.classify_risk(risk_score)`,
   this rule never reorders consistent records.
4. `sample_id` **ascending** — total-order tiebreaker; every permutation of
   identical inputs yields byte-identical output.

## 5. Why no second risk score is introduced

The prioritizer consumes `risk_score`, `risk_level`, `anomaly_score` and
`evidence_score` from the risk records as-is. Introducing another
aggregation here would create a second, partially redundant priority signal
and a second thing to defend. The only new concepts are *ordering* and the
P1–P4 *vocabulary*.

## 6. Why true labels are excluded from operational ordering

The risk records carry `true_label` for offline evaluation only. The
prioritizer **drops the field from its output** and never reads it for
ordering, so ground truth structurally cannot leak into the operational
queue (verified by tests that permute labels and assert the ordering is
unchanged). Using labels to order alerts would (a) be unavailable at
deployment time and (b) turn an evaluation signal into an operational one.

**Unflagged-records policy:** all records of the risk population are
prioritized, including LOW/P4 records the model did not flag — the queue is
complete and P4 is the explicit review backlog. Hiding them would hide the
model's false-negative-prone tail. A deployment may drop P4 from the
working queue; that choice belongs to the operator, not to this artifact.

## 7. Limitations

- Priorities inherit every limitation of the risk score (training-
  distribution evidence, no causality, no impact model).
- P1–P4 are queue vocabulary; equal-priority alerts are interchangeable.
- The ordering is only as good as the underlying scores; no feedback loop
  (analyst dispositions) exists yet — that belongs to a later stage.
- Ties at risk=0.0 (unflagged records with identical scores) are ordered by
  `sample_id`, which carries no security meaning — it exists purely to make
  the order deterministic.
- `source_row_id` (the record's zero-based row in its source dataset) is
  carried through as provenance metadata only: it is never validated and
  never used as a sorting key, so it cannot influence the ordering.

# Rebuttal strategy

## Paper core claims

1. RPS-GTok is an executable, payload-aware graph-token contract: fragments
   expand under an explicit schema and uncovered content falls back to
   primitives.
2. Canonicalization supplies a permutation reference frame; it is a hard
   graph-canonicalization problem, not a claimed polynomial-time contribution.
3. Model-facing identifier cost and complete reconstructable footprint are
   distinct quantities and must be reported separately.
4. RPS-GTok shows its strongest utility when invariant executable context is
   relevant, including balanced MOLHIV and the raw-relative power-flow control;
   canonical serialization remains a complementary reference frame.
5. The current materialized 21-dataset corpus has full exact certificate
   coverage under the documented 300-second per-graph budget.

## Reviewer overview and priorities

| Reviewer | Overall concern | Priority | Response posture |
|---|---|---:|---|
| 3HWf | truncation, Full-Embed fairness, canonical controls, semantic tasks, runtime | P0 | corrected protocol, matched controls, E14/E15, v12 exact audit |
| 8tZd | novelty, reconstruction/downstream causality, parameter artifacts, foundation baseline | P0 | mechanism-level novelty, paired diagnostic, equal-parameter consumers, GIN |
| bYQg | GI hardness, payload generation, schema, formal scope, complexity | P0 | acknowledge hardness, exact backend, payload boundary, schema audit, narrowed Eq. 32 |
| HFMW | canonicalized baselines, complete footprint, realistic tasks, scalability | P0 | R1b/R3/R9, E14/E15, v12 audit |
| Rkxp | complete storage overhead, realistic tasks, runtime | P1 | footprint table, E14/E15, v12 audit |

## Cross-reviewer arguments

- Canonicalization is explicitly separated from RPS executable fragment
  semantics; canonicalized serialization and frozen canonical-BPE controls are
  required controls.
- Results are framed by operating regime: RPS's strongest gains are visible in
  invariant semantic/topology contexts, while canonical serialization remains a
  valuable reference for simpler physical encodings.
- Equal-parameter fair consumer experiments use shared plain adapters and frozen
  union vocabularies.
- Real source labels are used only after source alignment; remaining records use
  a separately named topology proxy, never fabricated semantic labels.
- Exact certificate completion and token-stream permutation stability are
  distinguished; the former is complete on the current materialized corpus,
  and the latter is reported with its exact-completion condition.

## Evidence priority

### P0

- Corrected semantic and no-truncation results.
- Balanced MOLHIV and physical power-flow controls.
- Complete payload/roundtrip and schema audits.
- Full v12 exact certificate audit.
- Parameter/vocabulary fairness and GIN reference.

### P1

- Full footprint and artifact accounting.
- Compact topology-only codebook bytes and explicit topology/attribute/full-byte
  decomposition.
- E16 all-21 topology-only constructed task.
- Vocabulary sensitivity and coverage ablations.

### P2

- Presentation clarification, formal Eq. (32) scope, and camera-ready wording.

## Score-changing opportunities

The strongest evidence is the combination of (i) a balanced real-label MOLHIV
gain, (ii) a raw-relative power-flow gain, (iii) complete payload and schema
auditing, (iv) RPS-GTok leading on 9/21 constructed topology datasets, and (v)
full exact-certificate coverage. The response emphasizes these operating
regimes and reports all source tables without test-set tuning.

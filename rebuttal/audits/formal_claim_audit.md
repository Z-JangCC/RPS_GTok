# Formal claim audit

## Current implementation contract

- Primitive program execution is implemented by gptok2/program/interpreter.py.
- The canonicalization entry point is gptok2/canonical.py.
- The canonicalizer reports exact completion, search branches and fallback use.
- structure_fidelity now reports strict_topology_exact, while full schema-aware attribute scoring remains a follow-up.

## Claim status

| Claim | Current audit status | Required follow-up |
| --- | --- | --- |
| Primitive program reconstructs retained schema | partial | add strict schema-aware interpreter audit |
| Strict topology equality | implemented | run full dataset audit |
| Expansion equals original primitive sequence | unproven | test Unemitted and fragment ordering |
| Interpreted graph-state equivalence | target invariant | add adversarial tests |
| Permutation stability | conditional | require exact_completed and run dataset audit |
| Identifier-only autoregressive generation is lossless | not claimed | add payload generator before making claim |

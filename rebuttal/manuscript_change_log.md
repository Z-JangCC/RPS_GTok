# Manuscript change log

| Manuscript area | Required revision | Evidence |
| --- | --- | --- |
| Abstract | State that canonicalization is an adopted reference frame and reconstructability applies to complete token instances | REVISED_METHOD_SPEC.md and reviewer bYQg response |
| Introduction | Replace the claim that only RPS-GTok can combine reconstructability and stability with the executable primitive-fragment contract | E1 canonicalized controls |
| Schema | Add dataset-level schema table and explicit quantization/reconstruction target | Table_R8_schema_contract.tex |
| Canonicalization | Add graph-isomorphism complexity discussion, exact-completion condition and runtime reporting | canonicalization_audit.md and Table_R2_runtime_scaling_pilot.tex |
| Token definition | Separate identifier x from payload alpha and report both context cost and complete footprint | payload.py and Table_R9_token_level_21_fast60.tex |
| Formal properties | Use interpreted-state equivalence unless literal primitive-sequence equality is proven | formal_claim_audit.md |
| Experiments | Add canonicalized serialization baselines | Table_R1_canonicalized_baselines.tex |
| Experiments | Add semantic graph tasks and no-truncation control | Table_R4_semantic_downstream_all.tex and Table_R5_truncation_semantic.tex |
| Ablation | Add reconstruction/coverage deletion audit | Table_R7_reconstruction_ablation.tex |
| Sensitivity | Explain active vocabulary saturation | Table_R6_vocabulary_sensitivity.tex |
| Generation wording | Do not claim identifier-only autoregressive output is automatically lossless | reviewer bYQg response |

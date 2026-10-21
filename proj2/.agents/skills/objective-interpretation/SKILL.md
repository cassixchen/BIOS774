# Objective Interpretation

## Method

Read the user objective and data profile. Identify the task (e.g., generation, augmentation, imputation, anomaly detection, or representation learning), success criteria, constraints, and a suitable baseline.

## Decision

Use one of:
- `GENERATIVE_APPROPRIATE`
- `GENERATIVE_NOT_JUSTIFIED`
- `NEEDS_CLARIFICATION`

Do not train a generative model when a simpler approach better serves the objective.

## Output

Record the decision, rationale, task, success criteria, baseline, constraints, and next action in `objective_decision.json`.

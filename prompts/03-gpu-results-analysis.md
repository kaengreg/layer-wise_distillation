# GPU results analysis prompt

Inspect the H100 experiment logs and artifacts I have placed in `layer-wise_distillation/gpu_exp.`

Do not rerun training. Validate that the run reached the requested target depth, all losses and metrics are finite, the saved metadata is internally consistent, and the final checkpoint can be reloaded if the local environment allows it.

Update the experiment report with a comparison of the teacher, pruning-only student, and distilled student. Clearly separate measured results from interpretation, and document failures or unexpected regressions rather than hiding them.


Fill the report in docs/agent-worklog.md

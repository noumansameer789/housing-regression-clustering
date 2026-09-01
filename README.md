# Housing regression and clustering pipeline

An evidence-led reconstruction of a 20,640-row regression benchmark with an
explicit test of whether clustering actually helped prediction.

## Reported result from the original experiment

- tuned Random Forest: test RMSE 49,596 and R² 0.823
- dummy baseline: RMSE 117,798
- Nyström + LinearSVR validation RMSE 54,991 versus 56,205 for exact RBF SVR,
  while training on 13,209 rather than 4,500 rows
- cluster-feature ablation changed test RMSE by only 88 (about 0.2%)

## Executable portfolio layer

The code builds a leakage-safe mixed-type preprocessing pipeline, selects `k`
using silhouette score and returns a signed ablation delta. Tests use generated
clusters; the original assessment dataset and serialised models are not included.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The key judgement is visible in code and documentation: a negligible ablation
is reported as negligible rather than repackaged as model improvement.

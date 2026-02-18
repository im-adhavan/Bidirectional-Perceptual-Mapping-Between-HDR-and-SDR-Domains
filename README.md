# Bidirectional Perceptual Mapping Between HDR and SDR Domains

A research-grade, GPU-accelerated framework for modeling, evaluating, and analyzing **bidirectional tone mapping** between scene-referred HDR radiance and SDR display-referred representations.

This repository is not a toy tone-mapping demo.

It is a structured experimental system for studying:

- Forward HDR → SDR compression
- Analytical and learned SDR → HDR inversion
- Perceptual-domain reconstruction error
- Display-adaptive evaluation
- Cross-operator generalization
- Scene-structure → instability modeling

All experiments operate on **linear, scene-referred HDR radiance maps**.

---

## 1. Research Motivation

High Dynamic Range (HDR) imaging is now standard in camera pipelines and display systems. However:

- Most algorithms are designed for SDR
- Inverse tone mapping is underconstrained
- Operator behavior is rarely modeled structurally
- Perceptual evaluation is often superficial

This framework investigates:

1. Can tone-mapping be modeled as a structured geometric transformation?
2. Is invertibility predictable from scene radiometric structure?
3. Does operator choice induce perceptual manifold distortion?
4. Can cross-operator instability be quantified?

Rather than treating tone mapping as heuristic image processing, this work treats it as a **mapping between perceptual manifolds**.

---

## 2. Dataset

- 105 scene-referred HDR radiance maps
- Derived from multi-exposure NEF stacks
- Merged into floating-point EXR
- Linear radiometric domain (no gamma, no display encoding)

Dataset path:

```
data\hdr_exr
```

All processing is GPU-based and memory-safe via streaming EXR loading.

---

## 3. System Architecture

```
src/
├── dataset_loader.py
├── gpu_utils.py
├── tone_mapping.py
├── inverse_mapping.py
├── perceptual_encoding.py
├── contrast_model.py
├── color_metrics.py
├── reconstruction_metrics.py
├── scene_features.py
├── exposure_modeling.py
├── regression_analysis.py
├── stability_ranking.py
├── display_analysis.py
├── causal_analysis.py
├── learned_inverse.py
├── uncertainty.py
├── operator_transfer.py
├── manifold_visualization.py
└── plotting.py

scripts/
├── run_bidirectional_pipeline.py
└── run_operator_transfer.py
```

The system is modular and fully reproducible.

---

## 4. Forward Tone Mapping (HDR → SDR)

Implemented operators:

- Reinhard (global)
- Filmic

Mapping is modeled as parametric luminance compression in floating-point domain.

Example:

```
HDR → Reinhard → SDR
```

All operators run on GPU using PyTorch.

---

## 5. Analytical Inverse Mapping (SDR → HDR)

Analytical inverse of Reinhard:

```
SDR → inverse_reinhard → HDR_reconstructed
```

This allows controlled invertibility experiments.

---

## 6. Learned Inverse Mapping

A lightweight MLP scaffold is implemented for learned inversion:

- 3 → 64 → 64 → 3 architecture
- GPU-based
- Supports Monte Carlo dropout for uncertainty estimation

This enables:

- Learned reconstruction experiments
- Uncertainty modeling in inverse tone mapping

---

## 7. Perceptual Encoding (PU-style)

Perceptual error is computed in a luminance-encoded domain inspired by perceptually uniform (PU) encoding.

Features:

- Display peak normalization (100 / 400 / 1000 nits)
- Log-domain encoding
- Luminance anchoring
- Display-adaptive evaluation

Perceptual error is not raw RMSE.
It is computed in encoded luminance space.

---

## 8. Reconstruction Metrics

Computed metrics:

- RMSE
- PU-domain perceptual error
- Dynamic range reconstruction error
- Chromaticity distortion (CIE XYZ-based)
- Suprathreshold contrast modeling

This combines radiometric and perceptual evaluation.

---

## 9. Scene Radiometric Descriptors

Each HDR scene is described using:

- Dynamic range (percentile-based log10)
- Log-luminance standard deviation
- Highlight dominance ratio
- Shadow ratio

These features define the **radiometric manifold geometry**.

---

## 10. Multivariate Modeling

Ridge regression with cross-validation models:

```
Scene Features → PU Reconstruction Error
```

Outputs:

- R²
- Cross-validated R²
- Correlation matrix
- Feature influence analysis

This quantifies whether invertibility is predictable.

---

## 11. Display-Adaptive Evaluation

Evaluation performed across simulated display peaks:

- 100 nits
- 400 nits
- 1000 nits

This tests:

Does perceptual invertibility stability change with display brightness?

Tone mapping is not display-agnostic.
This explicitly models that dependency.

---

## 12. Cross-Operator Transfer Experiment

This is one of the core contributions.

Procedure:

1. Apply Reinhard → inverse_reinhard
2. Apply Filmic → inverse_reinhard
3. Measure PU error

Results:

- Near-zero error for Reinhard self-inversion
- Non-zero perceptual error when inverse mismatches operator

This demonstrates:

Tone mapping operators induce operator-specific perceptual geometry.

Cross-operator inversion creates measurable distortion.

---

## 13. Stability Ranking

Operators can be ranked by perceptual reconstruction error:

```
operator → mean PU error
```

This enables cross-operator perceptual stability comparison.

---

## 14. Manifold Visualization

PCA is applied to scene feature space:

- 2D embedding
- Radiometric manifold visualization

This supports structural modeling rather than heuristic comparison.

---

## 15. Causal Analysis

Correlation analysis tests:

```
Highlight dominance → PU reconstruction error
```

This investigates structural causes of instability.

---

## 16. Uncertainty Estimation

Monte Carlo dropout:

- Multiple forward passes
- Mean + variance estimation
- Reconstruction instability mapping

This is forward-looking toward perceptual reliability modeling.

---

## 17. How to Run

Activate environment:

```
python -m venv hdr_env
hdr_env\Scripts\activate
pip install -r requirements.txt
```

Run full bidirectional experiment:

```
python -m scripts.run_bidirectional_pipeline
```

Run cross-operator transfer experiment:

```
python -m scripts.run_operator_transfer
```

Outputs are saved to:

```
results/
├── csv/
├── figures/
```

---

## 18. Key Experimental Findings

- Reinhard inversion is analytically stable
- Cross-operator inversion produces perceptual distortion
- Radiometric summary statistics weakly predict perceptual instability
- Display peak affects perceptual reconstruction error
- Operator mismatch induces structured error

The perceptual manifold is not trivially invertible.

---

## 19. What This Work Demonstrates

This project establishes:

- A radiometrically grounded HDR pipeline
- Bidirectional tone-mapping modeling
- Perceptual-domain evaluation
- Display-adaptive analysis
- Cross-operator generalization study
- Structured scene → instability regression

It moves tone mapping from heuristic compression
to measurable geometric transformation between perceptual domains.

---

## 20. Future Directions

Potential extensions:

- Proper PU21 implementation
- Spatial-frequency modeling
- Chromatic adaptation modeling
- Local tone-mapping invertibility
- Neural operator-specific inverse networks
- Psychophysical validation

---

## Author

Adhavan Murugaiyan  

---

This repository is structured as an experimental framework for research in HDR imaging, tone mapping, and perceptual domain modeling.

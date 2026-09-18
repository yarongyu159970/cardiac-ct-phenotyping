# Statistical methods

## Phenotype derivation

Segment-level perfusion values (17 segments × MBF, MBV, TTP, PCBV, FE) are reduced by metric-wise
PCA retaining components up to 90% explained variance. Remaining numeric variables are screened for
pairwise |r| > 0.90 (PCA scores preferred over raw fields) and categorical variables that are nearly
determined by a numeric variable (eta squared > 0.80) are removed. K-prototypes clustering is run for
K = 2–5 with 20 seeds; the K = 3 solution from the prespecified reference seed defines the phenotypes.
Stability is assessed by seed agreement (ARI) and by refitting on 100 subsamples of 80% of patients
(ARI and cluster-wise Jaccard against the reference labels). Outcomes are not used at any step.

## Classifier

Candidate source variables (clinical variables, plaque volumes, CT-FFR, IMV and territory perfusion
means; variables with ≤ 6 distinct values dummy encoded) are ranked by the sum of a multinomial
L1-logistic rank (largest absolute class coefficient) and the Boruta rank, collapsed to one row per
source variable. The number of features is chosen as the smallest count whose repeated 5×5
cross-validated balanced accuracy and macro-F1 lie within 0.01 of the best value. The final model is
an L2-penalised multinomial logistic regression on 13 standardised features with C selected by
5-fold cross-validated log loss. External phenotypes are assigned by this frozen model without
re-clustering. Out-of-fold performance is the mean of five complete repeats of 5-fold cross-validation.

## Prognostic models

Models A (age, sex, SBP, antihypertensive treatment, TC, HDL, smoking, diabetes), B (A + CAD-RADS)
and C (B + phenotype) are ridge Cox models (penalty 0.01) fitted separately in each cohort.
Phenotype hazard ratios use Wald intervals from the penalised coefficient covariance; a sensitivity
analysis repeats the fit with penalties 0.001–1.

Composite-MACE performance is apparent: Harrell's C with a percentile bootstrap of patients and
their fitted scores (200 samples), and Graf IPCW Brier score and integrated Brier score over months
3–36 and 3–60 (1,000 grid points; failure weights G(T−), events before censorings at ties).

For hard MACE, first soft MACE is a competing event. Cause-specific ridge Cox models for both event
types give the hard-event cumulative incidence with exponential survival increments; discrimination
is the cause-specific Harrell C. Optimism is estimated from 1,000 bootstrap refits of all models
(corrected = apparent − mean(train − test)). Percentile intervals come from a nested bootstrap of
500 outer samples with 50 inner refits each; differences B−A and C−B are paired within replicates.

## Model comparison

Nested models are compared by the penalised partial-likelihood ratio statistic. P values are
calibrated by 4,999 model-based null-bootstrap samples with fixed covariates, event times simulated
from the reduced model (competing events for hard MACE) and reverse-Kaplan–Meier censoring. AIC uses
the unpenalised partial likelihood at the penalised estimates and effective degrees of freedom
trace(I·V).

## Calibration

Model C 36-month calibration is assessed in predicted-risk groups (10/8 groups for composite MACE in
derivation/validation, 5 for hard MACE) comparing mean predicted risk with Kaplan–Meier (composite)
or Aalen–Johansen (hard MACE) observed risk; hard-MACE intervals use 5,000 within-group patient
bootstrap samples.

## Other analyses

Phenotype-by-medication interactions are tested by likelihood-ratio comparison of ridge Cox models
with and without the interaction term, with Benjamini–Hochberg adjustment within cohort and endpoint.
Cumulative incidence by phenotype and Gray's tests use `cmprsk`. 

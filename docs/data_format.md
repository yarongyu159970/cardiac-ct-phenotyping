# Input data format

Both cohorts are supplied as one CSV (or XLSX) table with one row per patient.

## Identifiers and outcomes

| Column | Meaning |
|---|---|
| `ID` | Unique patient identifier; used only to align tables |
| `TIME` | Follow-up in months |
| `MACST` | 0 = no event, 1 = hard MACE, 2 = soft MACE. Composite MACE is `MACET != 0`; hard MACE is `MACET == 1` |
| `Cluster` | Derivation cohort only: K = 3 phenotype label (1, 2, 3). Any `Cluster` column in the validation table is ignored and replaced by the frozen classifier |

## Clinical variables

`age`, `gender` (0 = female, 1 = male), `BMI`, `SBP`, `HTN`, `DM`, `dislipidemia`, `smoking` (0/1),
`CCS` (1–3), `CACS` (1 = 0, 2 = 1–100, 3 = 101–400, 4 = > 400), `CAD-RADS` (0–5), laboratory values
`TC`, `HDL`, `LDL`, `TG`, `FG` (mmol/L) and `HbA1c` (%), medications `Antiplatelet therapy`, `Antihypertensive medication`, `Antidiabetic medication`,`Antiischemic medication`, `Nitrates`, `Statin` (0/1), plaque markers `HRP`, `LAP`, `PR`, `NRS`, `SC` (0/1).

## Imaging variables

* Plaque volumes in mm³: `whole lesion volume`, `low attenuation volume`, `calcified volume`,
  `fibrotic volume`, `fibrous fatty volume`.
* `CT-FFR-LAD`, `CT-FFR-LcX`, `CT-FFR-RCA` (dimensionless, 0–1).
* `IMV`: ischaemic myocardial volume fraction stored as 0–1 (5% = 0.05).
* Segment perfusion values `{segment}_{metric}` for segments 1–17 and metrics `MBF`, `MBV`, `TTP`,
  `PCBV`, `FE`; `Global` is the global MBF.
* Territory means `LAD_{metric}`, `LCx_{metric}`, `RCA_{metric}` with LAD = {1, 2, 7, 8, 13, 14, 17},
  LCx = {5, 6, 11, 12, 16}, RCA = {3, 4, 9, 10, 15}. A complete territory column is used as supplied;
  otherwise it is computed from the segments, which must then all be present, finite and nonnegative.

## Requirements of the prognostic models

Models A–C use `age`, `gender`, `SBP`, `Antihypertensive medication`, `TC`, `HDL`, `smoking`, `DM`, `CAD-RADS` and `Cluster`.
The categorical variables must use the integer codes above and every level must be present in each
cohort, giving 8, 13 and 15 coefficients for Models A, B and C. Missing values are not imputed;
incomplete rows cause an explicit error.

## Classifier input

`predict.py` requires the 13 features `whole lesion volume`, `CT-FFR-RCA`, `LAD_MBF`, `RCA_PCBV`, `TC`,
`CT-FFR-LcX`, `LAD_PCBV`, `IMV`, `low attenuation volume`, `HbA1c`, `LAD_TTP`, `FG`, `CT-FFR-LAD`
(territory means may be given directly or via the segment columns). Values must be finite and
nonnegative; `IMV` and the CT-FFR values must lie in [0, 1].

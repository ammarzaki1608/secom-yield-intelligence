# SECOM Yield Intelligence

A DMAIC-structured quality analytics project on the SECOM semiconductor manufacturing dataset — applying Statistical Process Control, process capability analysis, and root-cause-informed feature engineering before training a failure-prediction model, rather than treating this as a plain classification exercise.

**Status: Phase 1 (data pipeline + SPC EDA) complete. Capability analysis, root cause, and modeling in progress.**

## Why this project

Most public SECOM notebooks jump straight to "load data → train XGBoost → report accuracy." This project is structured the way a real quality investigation would run: monitor for instability (SPC), quantify whether the process meets requirements (Cp/Cpk), narrow down likely causes before modeling (root cause analysis), then predict and explain (XGBoost + SHAP) — closing the loop back to the causes.

## Dataset

[SECOM](https://archive.ics.uci.edu/dataset/179/secom) — 590 anonymized sensor/process features across 1,567 wafer runs, ~6.6% failure rate, timestamped per run.

**Note:** SECOM ships without official engineering spec limits. Where spec-limit-dependent analysis (Cp/Cpk) is needed, this project derives proxy limits from historical in-control distributions — a documented, stated assumption, not a claim of real fab specs.

**Getting the data:** raw files are gitignored (not included in this repo). Download `secom.data` and `secom_labels.data` from the link above and place both in `data/raw/`.

## Project structure

data/raw/ # raw SECOM files (gitignored — see above)
data/processed/ # cleaned data output from Phase 1
notebooks/ # numbered phases, run in order
src/ # reusable pipeline/analysis functions imported by notebooks
reports/figures/ # exported charts

## Progress

- [x] **Phase 0 — Define:** problem statement and cost-asymmetry assumption (`notebooks/00_problem_statement.md`)
- [x] **Phase 1 — SPC & data pipeline:** leakage-safe, timestamp-ordered loading; missing-value and zero-variance cleaning; SPC control charts on the sensors most correlated with failure (`notebooks/01_spc_eda.ipynb`)
- [ ] **Phase 2 — Process capability (Cp/Cpk)**
- [ ] **Phase 3 — Root cause / feature hypotheses**
- [ ] **Phase 4 — XGBoost model, leakage-safe split, cost-aware threshold**
- [ ] **Phase 5 — SHAP explainability, reconciled against root-cause hypotheses**
- [ ] **Phase 6 — DMAIC case study writeup**

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
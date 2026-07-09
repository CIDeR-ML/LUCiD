# docs/internal — not part of the published site

These documents are **excluded from the hosted docs** (`mkdocs.yml` sets `exclude_docs:
internal/`). They are kept for developer/historical reference only.

- **Superseded / completed migration & planning docs** — `EVENT_FORMAT_V2` (superseded by the
  v3 schema), `LUCID_MIGRATION*`, `MAIN_BRANCH_PLAN`, `MERGE_TTS_CHANGES`,
  `MERGED_DETECTORPARAMS_PROPOSAL`, `PHASE3_PLAN`, `PLAN_UNIFY`, `RECONCILIATION_PLAN`,
  `RECON_CONSOLIDATION`, `UNIFY_CALIB_RECON`.
- **Design notes that describe an unshipped/aspirational architecture** (do NOT treat as current
  API): `PRACTICE_ARCHITECTURE` (`Source.emit`/`ResponseModel`/`ParamRegistry` — not in the
  code), `CALIBRATION_FRAMEWORK` (references a `lucid/calibration/` package that does not exist;
  real calibration lives in `lucid.fitting`).
- **`cleanup/`** — the repo cleanup & release plan (00–05).

For current, accurate documentation see the published site (the rest of `docs/`).

# Testing Guide

## 1) Recommended command
`powershell`
```powershell
.\scripts\run-tests.ps1
```

Specific files:
`powershell`
```powershell
.\scripts\run-tests.ps1 tests/test_adss_subset_wrapper.py tests/test_adss_multi_neighbor.py
```

## 2) Why this wrapper exists
In this environment, `python`, `py`, and `pytest` from PATH can point to Windows app stubs and fail.
The wrapper resolves a real interpreter and runs:
`text`
```text
python -m pytest ...
```

## 3) Current pytest scope
`pytest.ini` ignores legacy/script-style tests that either:
- execute top-level `sys.exit(...)` during import, or
- depend on removed/renamed APIs/modules.

This keeps the automated suite stable and focused on maintained tests.

## 4) Current status (2026-03-11)
- `49 passed` on the default suite.

## 5) Change memo (2026-03-11)
- DIC pipeline core fixes (variable algorithm excluded):
  - ICGN `ref_cache` integrity check + safe fallback path
  - ADSS recovered subset representative reflected to parent POI result arrays/statistics
  - Export summary quality-grade normalization fix
  - Image switch now clears stale ICGN/FFT state when result is absent
  - Single-run FFT path now reports progress callback consistently
  - Batch summary recommendation logic hardened for failures/defaults
  - Notes added near ADSS/PLS boundary for future full sub-POI strain support
- Test visualization outputs unified under `tests/_outputs/<subfolder>` across crack-analysis scripts.

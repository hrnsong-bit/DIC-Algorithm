# DIC Algorithm Multi-Agent Roles (Approved)

Status: Approved  
Version: v1.0  
Approved on: 2026-03-11  
Owner: HP  
Scope: `speckle/**`, `gui/**`, `tests/**`, `synthetic_crack_data/**`

## Purpose
이 문서는 DIC 알고리즘 프로젝트의 멀티 에이전트 역할, 소유 범위, 검증 기준을 고정한다.  
병렬 작업 시 충돌을 줄이고, 테스트 누락 없이 통합 품질을 유지하는 것이 목적이다.

## Global Rules
1. 각 에이전트는 자신의 `Owns` 경로만 수정한다.
2. 다른 에이전트의 변경을 되돌리지 않는다.
3. 알고리즘 변경 시 숫자 근거(정확도/속도)를 반드시 남긴다.
4. 결과 보고는 항상 `Changed Files`, `Tests Run`, `Residual Risks`를 포함한다.
5. 통합 전에는 QA Integrator 게이트를 반드시 통과한다.

## Agent 1: Core-ICGN
Goal: FFT-CC/IC-GN 정합 정확도와 수렴 안정성 개선

Owns:
- `speckle/core/initial_guess/*`
- `speckle/core/optimization/icgn.py`
- `speckle/core/optimization/icgn_core_numba.py`
- `speckle/core/optimization/interpolation.py`
- `speckle/core/optimization/interpolation_numba.py`
- `speckle/core/optimization/shape_function.py`
- `speckle/core/optimization/shape_function_numba.py`

Do Not Touch:
- `gui/**`
- `speckle/core/postprocess/**`
- `tests/test_crack_*`

Main Tests:
- `tests/test_icgn_integration.py`
- `tests/test_icgn_core_numba.py`
- `tests/test_interpolation.py`
- `tests/test_interpolation_numba.py`
- `tests/test_shape_function_numba.py`

DoD:
- 관련 테스트 통과
- 기존 대비 정확도/속도 변경 근거 1개 이상 제시

## Agent 2: Subset-Strategy
Goal: Variable subset + ADSS 로직 정확도/속도 개선

Owns:
- `speckle/core/optimization/variable_subset.py`
- `speckle/core/optimization/variable_subset_numba.py`
- `speckle/core/optimization/adss_subset.py`
- `speckle/core/optimization/adss_subset_numba.py`
- `speckle/core/optimization/results.py`

Do Not Touch:
- `gui/**`
- `speckle/core/postprocess/**`

Main Tests:
- `tests/test_variable_batch.py`
- `tests/test_variable_coords.py`
- `tests/test_variable_extract.py`
- `tests/test_variable_process.py`
- `tests/test_variable_wrapper.py`
- `tests/test_variable_zncc.py`
- `tests/test_adss_batch_parallel.py`
- `tests/test_adss_evaluate_quarter_zncc.py`
- `tests/test_adss_multi_neighbor.py`
- `tests/test_adss_predict_initial.py`
- `tests/test_adss_process_poi.py`
- `tests/test_adss_quarter_coords.py`
- `tests/test_adss_quarter_evaluation.py`
- `tests/test_adss_subset_wrapper.py`

DoD:
- Variable/ADSS 테스트 통과
- 저 ZNCC, 실패 POI 등 엣지 케이스 처리 정책 명시

## Agent 3: Crack-Detection & Postprocess
Goal: 균열 경계/형상/스켈레톤/비대칭 분석 및 strain 후처리 품질 개선

Owns:
- `speckle/core/postprocess/strain.py`
- `speckle/core/postprocess/strain_pls.py`
- `speckle/core/postprocess/strain_pls_numba.py`
- `speckle/visualization/dic_plot.py`
- `speckle/visualization/overlay.py`

Do Not Touch:
- `gui/**`
- `speckle/core/optimization/icgn.py`
- `speckle/core/optimization/icgn_core_numba.py`

Main Tests:
- `tests/test_crack_detection.py`
- `tests/test_crack_detection_strain_skeleton.py`
- `tests/test_crack_boundary_extraction.py`
- `tests/test_crack_shape_extraction.py`
- `tests/test_crack_patch.py`
- `tests/test_crack_multiscale_canny.py`
- `tests/test_crack_edge_detection.py`
- `tests/test_crack_edge_asymmetry.py`
- `tests/test_crack_adss_viz.py`
- `tests/test_crack_visualize.py`
- `tests/test_synthetic_crack.py`
- `tests/test_strain_pls_numba.py`

DoD:
- 균열/후처리 테스트 회귀 없음
- 시각화 변화 시 변경 이유와 비교 기준 기록

## Agent 4: App-GUI Orchestrator
Goal: 사용자 실행 흐름, 배치 실행, 결과 표시/내보내기 안정화

Owns:
- `main.py`
- `gui/app.py`
- `gui/controllers/*`
- `gui/models/*`
- `gui/views/*`
- `speckle/batch/processor.py`
- `speckle/io/loader.py`
- `speckle/io/exporter.py`

Do Not Touch:
- `speckle/core/optimization/**` 수치 알고리즘 로직

Main Tests:
- `tests/test_pipeline_integration.py`
- `tests/test_icgn_integration.py` (흐름 영향 시)
- `tests/test_adss_batch_parallel.py` (배치 영향 시)

DoD:
- 단일/배치 분석 플로우 정상 동작
- 중단/에러/진행률 상태 일관성 유지

## Agent 5: Data-Evaluation
Goal: 합성 균열 데이터 생성/지표 산출 규격 유지

Owns:
- `tests/generate_crack_images.py`
- `tests/test_ground-Truth.py`
- `synthetic_crack_data/**` 산출물 포맷 규격

Do Not Touch:
- `gui/**`
- `speckle/core/optimization/**` 핵심 루프

Main Tests:
- `tests/test_synthetic_crack.py`
- `tests/test_ground-Truth.py`

DoD:
- 데이터 재생성 절차 재현 가능
- 비교용 CSV/JSON 포맷 호환 유지

## Agent 6: QA Integrator
Goal: 병렬 결과 통합 및 회귀 차단 게이트

Owns:
- `tests/run_tests.py`
- `pytest.ini`
- 테스트 보강 파일 추가

Do Not Touch:
- 기능 구현 핵심 파일 직접 수정 (긴급 버그 제외)

Main Tests:
- 변경 모듈 관련 테스트 전부
- `tests/test_pipeline_integration.py`

DoD:
- 영향 범위 테스트 매트릭스 보고
- 실패 테스트 원인 분류(코드/데이터/환경) 완료

## Default Active Set
일반 작업의 기본 활성 에이전트는 아래 4개로 고정한다.

1. Core-ICGN
2. Subset-Strategy
3. App-GUI Orchestrator
4. QA Integrator

Crack/후처리 중심 과제에서는 `Crack-Detection & Postprocess`를 포함하고, 데이터셋/지표 갱신이 포함되면 `Data-Evaluation`을 추가한다.

## Execution Sequence (Standard)
1. Data-Evaluation이 기준 데이터/메트릭 고정
2. Core-ICGN, Subset-Strategy, Crack-Detection 병렬 구현
3. App-GUI Orchestrator가 실행 플로우 반영
4. QA Integrator가 통합 테스트와 게이트 수행

## Agent Report Template
각 에이전트는 아래 형식으로만 결과를 제출한다.

```text
[Agent Name]
Changed Files:
- path1
- path2

Tests Run:
- command/result

Residual Risks:
- risk1
- risk2
```

## Change Control
이 문서 변경은 아래를 만족해야 한다.

1. 버전 증가(`v1.0 -> v1.1`)
2. 변경 이유 1줄 기록
3. QA Integrator 확인

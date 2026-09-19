# 서버 계산 기록 (Paper 3, v10)

이 문서는 `compute-host`에서 돌린 모든 GPU 작업의 원장이다. 원고의 "Code, Data, and
Compute" 절과 제출용 provenance 진술은 여기서 읽어 쓴다. **새 실행을 시작하거나 끝낼
때마다 이 표에 한 줄을 추가한다.**

## 호스트

| 항목 | 값 |
|---|---|
| 호스트 | `compute-host` (ssh alias; wake-on-lan (wake-on-LAN command omitted)) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16 376 MiB, driver 560.35.05, CUDA 12.6 |
| CPU/RAM | 20 코어 / 31.9 GiB |
| OS / Python / torch | Linux 6.8.0-138-generic · anaconda3 Python 3.12 · torch 2.5.1+cu121 |
| 솔버 | torcwa 0.1.4.2 |
| 작업 트리 | `~/InverseTL` (Mac에서 rsync), 로그 `logs_pub/`·`logs_v10/`, 결과 `results_pub/` |
| 상류 입력 | `~/InverseTL/upstream_inputs_pub` (pub) / `upstream_inputs_v8` (legacy). `~/mim_novel`은 2026-06-12 사본으로 **입력원으로 쓰지 않는다** |

프로파일 스위치: `INVERSETL_PROFILE=pub` → vendored `src_v8/upstream_cb486b5`,
파장별 adaptive Fourier order N ∈ {9,13,17}, complex64, Johnson–Christy.
`legacy` → `upstream_920b1bd`, 고정 order 5, complex128.

## 실행 원장

| 시각 (KST) | 태스크 | 내용 | 산출물 | 결과 |
|---|---|---|---|---|
| 09-06 12:54–13:12 | T52 | `pretrain_pub.py` — 인쇄 파이프라인 M_TL+phys 체크포인트 A/B/C 재생성 (rng(99), TMM 5000, seed 42, 500 epoch) | `results_pub/pretrain_*`, sha256 66aa7db1 / 892aabda / 84fe0a59 | OK |
| 09-06 12:58–13:24 | T52 | seed-42 fine-tune + 게이트 판정 | `pub_gate_table123.json` | **PASS** — A 1.757 / B 1.722 / C 2.107 % 대 인쇄 Table 1–3의 1.79±0.04 / 1.65±0.04 / 2.05±0.09 (\|z\| ≤ 1.7) |
| 09-06 14:22–14:28 | T53 | adaptive-order 오라클 스모크 + 샤딩·재개 확인 | `logs_pub/t53_*.log` | OK |
| 09-06 (측정) | T54 준비 | 설계당 오라클 비용 측정 | `logs_v10/time_adaptive.log` | A 57.1분 / C 26.9분 / B 3.1분 (100 파장, order 17 수렴) |
| 09-06 ~15:30–16:30 | T54 | **교착 사고** — RCWA 워커 3개가 각자 4.5–6.5 GB PyTorch 할당자 캐시를 쥔 채 서로의 VRAM 해제를 대기해 전원 정지. 구 프로세스는 SIGTERM 무시, PID 지정 `kill -9` 필요 | — | 수정: `oracle.wait_for_vram`이 대기 루프에서 자기 캐시를 먼저 `empty_cache()`, `PEAK_GB` 하향, 드라이버 `RPAR=2` |
| 09-06 12:54–16:22 | T54 stage 1–2 | 9개 실행(3 구조 × 3 seed) fine-tune + inverse | `results_pub/{finetune,stats,surrogate,inverse}_*` | 완료 |
| 09-06 16:39– | T54 stage 3 | RCWA adaptive 오라클, 180 설계, 동시 워커 2 | `results_pub/oracle_cache/rcwa_*_NNN.npz` → `rcwa_*_v8.npz` | **진행 중** — B seed42 20/20 완료(3742 s), A seed42 3/20, C seed42 5/20 (09-06 20:20 기준) |
| — | T54 stage 4–5 | random_baseline + restart0 (seed 42), reliability JSON 9종 | | 대기 |
| — | T55 | pub arm 게이트 실험: M0 from-scratch(3 seed) + feasible 재실행(3 seed) + feasible random baseline | | 대기 |

## 예산 (2026-09-06 실측 기준)

| 단계 | 남은 설계 | GPU-시간 |
|---|---|---|
| T54 stage 3 (seed 42 잔여 + seed 123/777) | 152 | ~106 |
| T54 stage 4 | 120 | ~33 |
| T55 | 420 | ~203 |
| 09-12 09:38–09:57 | T66 | codex slice 3 반영 재실행 — `feasibility_crosstab_v8.py --results results_pub`(CPU; 값 불변, `test_note` 추가), `detector_bench.py`(CPU; 타깃 군집 부트스트랩으로 교체 — 점추정 36/36 동일, 구간만 변경, A/knn_train 0.5 배제 해제; `ci_method` 기록), `results_pub/INPUTS_MANIFEST.txt` 생성(데이터셋 4·체크포인트 5 sha256) | `results_pub/{feasibility,detector_bench}_v8.json`, `logs_pub/*_rerun_0912.log` | OK |
| 09-11 16:37–09-12 12:06 | T70 | Structure D 오라클 50/50 성공(솔버 시간 69,553 s, 설계당 중앙값 1505 s; N=17 45/50) → reliability_D | `results_pub/rcwa_D_v8.npz`, `reliability_D_v8.json` | **4/50 pretender (8.0 %)** — 기록된 5–15 % 밴드 안, 반증 아님; 4/4 infeasible |
| 09-12 11:56 | T70 stage 2 | **교차 솔버 실패** — `INVERSETL_TAG=_o5`가 입력(inverse 산출물)까지 재지정해 `inverse_A_o5_v8.npz` 부재로 즉사; `&`+`wait`가 종료코드를 안 봐 'stage 2 done'으로 기록됨 | `logs_pub/rcwa_*_o5_pub.log` | 13:45 `scripts/run_pub_crosssolver.sh`(`--tag o5`, 순차)로 재실행 |
| 09-12 11:59 | T66 stage 0 | 새 `oracle.py`(c1d9445) 설치 + 캐시 스탬프 `--apply` **530/530**, 검증 0 잔여 | `oracle_cache_migration_v10.json` | OK |
| 09-12 11:59– | T66 stage 1 | pub arm 재현 검사(구조당 0,1,2,6, tag reprocheck, 캐시 우회): B 4/4 완료(738 s), C·A 진행 중 | `rcwa_*_v8_reprocheck.npz` | 진행 중 |
| 09-12 13:45–14:05 | T70 stage 2 재실행 | 교차 솔버 `run_pub_crosssolver.sh` (`--order 5 --tag o5`, 순차): B 312 s, A ~9 min, C 336 s; 60/60 성공 | `results_pub/rcwa_{A,B,C}_v8_o5.npz` → `cross_solver_v11.{json,md}` | **pretender 6 → 13/60**(flip 7/0, McNemar p=0.016); 중앙값 shift +0.61 pp |
| 09-12 11:59–16:17 | T66 stage 1 | pub arm 재현 검사 완료: B 738 s, C 7009 s, A 14726 s(합 6.2 GPU-h), 12/12 **bitwise 동일**(max\|dMAE\| = 0, max\|dA\| = 0), pass=true | `results_pub/repro_check_v8.json`, `rcwa_*_v8_reprocheck.npz` | OK — §3 도입부 보류 해소 |
| 09-12 16:17– | T65 stage 2 | 20→50 확장 시작(tag `_n50`): inverse B(50) 완료 — 앞 20개 기하가 base와 **정확히 동일**(max diff 0) → A 진행 중; 이후 C, seed 123/777, 오라클 RPAR=2 | `results_pub/inverse_*_n50_v8.npz` | 진행 중 |
| 09-13 19:46 | T65 stage 2 | **GPU 장애** — 커널 `NVRM Xid 31 (MMU fault)`; A·C 오라클 워커가 GPU 0 %·CPU 100 %로 정지(A 26/50, C 34/50), 새 프로세스도 `CUDA unknown error`. 워커·드라이버 PID 종료, 저자가 21:20 재부팅 | `logs_pub/rcwa_{A,C}_n50_pub.log` | 손실 ~1.75 h |
| 09-13 21:21 | T65 stage 2 | 재부팅 후 CUDA 정상; `run_pub_followon.sh`·`run_n50_analysis.sh` 재기동 — 파일 기반 재개(캐시 26+34개 재사용, 재계산 0). 캐시 재스탬프 832/832(`rcwa_o5` kind 처리 수정) | `logs_pub/followon_driver.log`, `stamp_apply_0913b.log` | 진행 중 |
| **합계** | **692** | **~342 → 동시 워커 2 기준 벽시계 ~7–8일** |

VRAM 상한이 동시성을 정한다: A 7.4 GB + C 4.4 GB = 11.9/16 GB이므로 워커 2개가 한계다.
드라이버는 9개 실행을 하나의 큐로 돌려 두 슬롯을 항상 채우므로 벽시계 ≈ GPU-시간 / 2.
저자 결정(2026-09-06): 범위 축소·클라우드 임대 대신 **원안 그대로** 진행.

## 운용 메모

- 드라이버 재기동은 안전하다(설계 단위 캐시 + 원자적 쓰기). 다만 **진행 중이던 설계 1개
  (A면 최대 1시간)를 버린다.** 안정적으로 도는 동안은 건드리지 않는다.
- `logs_pub/core_FAILED.txt`가 생기면 그 실행만 재기동하면 된다. 2026-09-06 16:37의
  `random_baseline_*`·`restart0_*` OOM 트레이스백은 **교착 사고 때의 옛 로그**이고
  16:39 재기동 뒤의 stage 4는 아직 시작하지 않았다.
- 서버 응답이 없으면 잠든 것이다: (wake-on-LAN command omitted) → 30 s 대기 → 재시도.

# Paper 1 — 출판본 수치 (Paper 3가 인용해야 하는 값)

**출처(확정)**: 저자가 지정한 출판 PDF the published PDF (author's copy)
(MD5 9f5dd3f4ded90624ddc35db44938d544, 15 pp, IOP/Elsevier 조판본, 생성 2026-08-15) — *Photonics and Nanostructures – Fundamentals and Applications* **72**(Part B) 101617 (2026).
로컬 `<PBTL-MIM working copy>/paper.pdf`(MD5 35cf49b3…, 2026-07-05 저자 원고본)도 동일 값. 확인일 2026-09-06.

## 세 사본 대조 (2026-09-06)

| 사본 | 날짜 | median r A/B/C | TMM MAE A/B/C | 판정 |
|---|---|---|---|---|
| `compute-host:~/mim_novel/paper.pdf` (MD5 d4d5a27e…) | 2026-06-12 | 0.72 / 0.64 / 0.44 | 15.0 / 21.0 / 22.9 % | **구버전 (R1 중간본)** — Paper 3 v8 claims draft가 이 값을 씀 |
| `<PBTL-MIM working copy>/paper.pdf` (MD5 35cf49b3…) | 2026-07-05 | 0.83 / 0.96 / 0.65 | 7.9 / 8.9 / 16.9 % | 최종 저자 원고 = 출판본과 일치 |
| published PDF (author's copy) (MD5 9f5dd3f4…) | 2026-08-15 | 0.83 / 0.96 / 0.65 | 7.9 / 8.9 / 16.9 % | **출판본 (정본)** |

→ 6월 12일과 7월 5일 사이에 Paper 1 데이터가 재생성되어(2026-06-20 cross-platform 로그, 하이브리드 재생성) TMM MAE가 절반으로 줄고 r이 전면 바뀜. 서버 사본은 갱신 필요(플랜 T02 upstream 고정 시 함께 처리). Paper 3 `src_v8`이 참조하는 upstream 데이터·체크포인트는 이 재생성 **이전** 것(G18) — 원고에 "audit of the pre-correction release" 명시 필요.

## Table 5 (출판본) — 구조별 TMM–RCWA fidelity와 transfer benefit

| Structure | TMM MAE (operating band) | mean r | **median r** | Best gain (best model vs M0) | best model |
|---|---|---|---|---|---|
| A (Dual-cavity) | 7.9 % | 0.69 | **0.83** | 32.1 % | M_TL+phys |
| B (Ring-disk) | 8.9 % | 0.88 | **0.96** | 19.7 % | M_TL+phys |
| C (Dual-polarization) | 16.9 % | 0.53 | **0.65** | 9.7 % | M_TL |

- 가중치 전이 benefit (M_TL vs M0, n=350): A +31.3 % / B +13.5 % / C +9.7 % — Table 5의 Best gain과 구별됨 (본문·Fig. 4 강조값).
- 출판본의 r 순서: **B (0.96) > A (0.83) > C (0.65)**. TMM-MAE 순서: A (7.9) < B (8.9) < C (16.9).
- 출판본 해석: "shape correlation r alone is insufficient … it is the **joint r-and-MAE** reading" (Structure B는 r 최고인데 benefit은 A보다 작음). Benefit 순서는 MAE 순서를 따름.
- Structure A full reliable pool median r = 0.83; pilot n=20 bias < 0.02, n=50 < 0.01 (§ pilot-set).

## Paper 3 원고(v9)에서 틀린 값 — 전부 교체 대상
- v9가 "published r"로 쓴 **A +0.72 / B +0.64 / C +0.44 는 출판본에 없음** (R1 응답서·correction catalog의 중간 정정값). → `paper1_published_values.md`의 median r 0.83/0.96/0.65로 교체.
- v9의 "preliminary r" **A +0.72 / C +0.34 / B −0.07** (사전등록 당시 값)은 역사 기록으로 유지하되 "Paper 1 as submitted"가 아니라 "the preliminary pilot-r values available at pre-registration (2026-06-10), later superseded by the published Table 5"로 기술.
- 순서가 A>B>C 에서 **B>A>C 로 바뀌므로** H1 재평가: 출판 r 순서로 pooled pretender율은 B 50 % / A 45 % / C 64 % → **비단조** (플랜 결정 게이트 "printed r = 0.83/0.96/0.65" 분기 적용).

## 저자 확인 체크리스트 (PLAN T01)
- [x] 인쇄된 median r 값 확정 (위 표) — 2026-09-06 저자 제공 PDF로 확인
- [x] 프레이밍 결정 (2026-09-06): **본문 = 출판 파이프라인(redesign) 재실행, v8 legacy = SI 원 실행**; 'pre-registered' → 'pre-specified' (`docs/PLAN_v10_amendment_pub.md` §5)
- [x] 1차 구조 순서 (2026-09-06): **출판 median r 순 B, A, C**
- [x] 공개 repo (2026-09-06): **새 저장소** (이름·라이선스는 P4/T50에서 확정)

# PLAN v11 — pub arm 데이터에 맞춘 원고 재작성 계획 (2026-09-09)

근거: `docs/interim_pub_vs_legacy.md`, `docs/review_pub_switch_codex_v10.md`(Must-Fix 9의 23개
위치 인벤토리), `docs/gate_summary_v10.md`, `docs/PLAN_v10_amendment_pub.md` §6–§8.
원칙은 PLAN_v10과 같다: 수치는 전부 `results_pub/` 아티팩트에서, 제목·논지·기여 문장은 저자가
정하고 나는 초안과 `AUTHOR-SIGNOFF` 표시만 남긴다. v10을 편집하지 않고 **T58에서
`manuscript_v11.md`로 fork**한다(v10은 legacy 본문의 동결 기록).

## 무엇이 바뀌었나 (한 문단)

출판 파이프라인에서 pretender는 16/179(8.9 %)이고, 사전 명세한 예비 *r* 순서가 H1a에서
단조이며(추세 p = 0.0345, 설계효과 보정), 진단자가 B·C에서 작동한다(C held-out MAE AUROC 0.98).
legacy arm의 53 %·비순서·진단자 실패는 투고본 파이프라인의 성질이었다. **살아남는 것**은
비보장(179개 전부 자기검증 통과, 16개 솔버 기각), 파이프라인 의존성(53 % → 9 %), 프로토콜.
Paper 1은 역설계를 주장한 적이 없고 §4.2·§4.6·S13에서 경고했으므로, 속편의 구도는
"Paper 1이 한계로 남긴 질문에 답했다"이다.

## Phase 0 — 저자 결정 (글쓰기 전에 필요)

| # | 결정 | 선택지 | 내 권고 |
|---|---|---|---|
| D1 | 제목 | (a) 현 제목 유지 — 데이터가 안 받침 (b) "Self-certification is not certification: a pre-specified oracle audit of physics-transferred surrogates in inverse metasurface design" (c) "How much does a solver call buy? …" 류의 프로토콜 중심 | **(b)** — 비보장이 살아남는 핵심 |
| D2 | 진단자 결과의 위치 | (a) §4.4를 "값싼 가드가 어디까지 되는가"라는 건설적 절로 (b) 한계 단서로만 | **(a)**, 단 사건 수(2/5/9) 명시 |
| D3 | 타깃 30개 추가 (남은 held-out 전부) | (a) base TL만 3구조 × 30 = 90 설계, ~50 GPU-h (b) base + M0, 180 설계, ~100 GPU-h (c) 안 함 | **(a)** — T55 후 시작; 추론 문장을 하나라도 두려면 최소치 |
| D4 | 기여 목록 재구성 | 5개 부정 기여 → 4개: 프로토콜 / 비보장 / 파이프라인 의존성 / 진단자·순서의 조건부 신호 | 초안 제출 후 승인 |
| D5 | legacy arm의 자리 | SI 전용(결정 1 유지) vs 본문 두 arm 대비 절 안에 표 하나 | 본문에 대비 표 1개, 나머지 SI |

D3의 선행 조건: codex Must-Fix 10(오라클 캐시가 솔버 설정을 검증하지 않음)을 먼저 고친다.

## Phase 1 — 기계적 재지정 (지금, 결정 불필요) · T58–T60

| id | 내용 | 검증 |
|---|---|---|
| T58 | `manuscript_v11.md` fork; `check_manuscript_v10.py --results results_pub`를 v11에 맞게 확장 | 스크립트가 pub 아티팩트 키를 전부 찾음 |
| T59 | 증거 JSON 재생성: ✅ pooled·stats_supplement·synthesis×3·mechanism·detector_bench 완료 / ⏳ feasibility_crosstab·lookup_null·recoverability를 `results_pub`로 | 각 JSON 존재, 게이트 표의 NOT RUN이 T55 항목만 남음 |
| T60 | 그림 재생성 `make_figures_v9.py`(INVERSETL_RESULTS_DIR=results_pub) → `figures_v11/`; order-7 그림(5)은 legacy라 SI로 | 그림 7종 + S 2종, Type 3 폰트 0 |

## Phase 2 — 방향이 확정된 절 재작성 (지금) · T61–T63

| id | 절 | 바뀌는 것 |
|---|---|---|
| T61 | 초록·§1.4·§1.5·§2.1·§2.5·§2.7 | 수치 교체; §2.1은 pub 파이프라인(adaptive order 9/13/17, complex64, JC, C 18 feature, A 400–1800 nm); §2.7에 9월 타임라인(pub 실행, 두 arm 결정, T55 개정)과 분석 상태 행 추가; 5 %가 더는 forward MAE 스케일이 아님을 명시 |
| T62 | §3.1–3.4, §3.7, **신설 §3.x 두 arm 대비(T57)** | 표 6·7·8·10을 pub으로; §3.3은 "사전 명세 순서에서 H1a 단조, 추세 p = 0.0345, 사건 16건이라 *일관됨*이지 *확립* 아님"; §3.7은 "고정 임계값은 무정보, 연속 점수는 B·C에서 판별" + 사건 수; §3.5(order-7)는 SI로 이동하고 본문은 adaptive solver의 fidelity 진술로 대체; 두 arm 절은 "bundled comparison" 문구와 caveat 그대로 |
| T63 | §4.1·§4.2·§4.4·§4.5·§4.7·§5 | 논지 강도 하향; W1(순서 무지지)·W2(대역 불일치)·W6(C 풀/경계) 소멸, W7(통제 부재)은 T55 후 갱신; Paper 1 관계를 "§4.2/§4.6/S13이 열어 둔 질문에 답함"으로 |

T61–T63의 모든 방향 전환 문장에 `<!-- AUTHOR-SIGNOFF: V11-… -->`. 분량 상한 8500어(현재 본문 ~6400 + 두 arm 절 ≈ +500).

## Phase 3 — T55 이후 (9/12) · T64

`control_analysis_v10.py` → §3.6 M0 블록·§3.8 feasible; 게이트 표 완성 → T24 저자 결정;
기여 3·4 최종 문구. 분석계획 개정은 codex 6번의 공개 목록대로 §2.7에 적는다
("base-arm 결과를 보고 개정, 통제 결과가 나오기 전에 고정").

## Phase 4 — 선택: 타깃 30개 추가 (D3, 9/12–9/14) · T65

캐시 지문(솔버 설정·파장·재료·정밀도·코드 핀) 검증을 `oracle.py`에 넣은 뒤,
`inverse.py --extra-targets 30`(기존 20개 유지, 남은 held-out 30개 추가)로 base TL 3구조.
결과는 §3.2·§3.3 표에 n = 50/구조로 합산. 사건 수 ≈ 16 → 24.

## Phase 5 — 마감 (9/14–9/16) · T66

`check_manuscript --results results_pub` PASS → `md_to_latex_v10.py --build` PASS → 커버레터
헤드라인 교체(T43 재실행) → **codex 2차 리뷰**(v11 대상) → T44 저자 서명.

## 일정

| 날짜 | 작업 |
|---|---|
| 9/9–9/10 | Phase 1–2 (T58–T63), 저자는 D1·D2·D4·D5 |
| 9/12 | T55 종료 → Phase 3 (T64) |
| 9/12–9/14 | Phase 4 (D3 = a면) |
| 9/14–9/16 | Phase 5, 제출 |

# Bayesian factor regression 구현 계획

작성일: 2026-10-01. 갱신일: 2026-10-06. 상태: **현재 Sampling 범위 P1–P6 기준 구현·검증 완료 — Inference·Scalable SSP 보류, full pilot 미실행**.

현재 Sampling 범위의 모델 명세와 구현 계약은 개발 착수가 가능한 수준으로 정리되었다. 이 문서에서 모델 명세, 확정된 결정, 변경 가능한 default, 후속 질문, 단계별 완료 조건을 함께 관리한다. 사용자 확정 사항은 §2와 §11에 표시하며, 제안 default를 본 실험 scientific setting의 확정으로 간주하지 않는다. 공통·Normal·Horseshoe·Reference SSP 코드와 검증을 구현했다. 실제 검증 범위는 §10과 docs/validation.md에 기록한다.

**2026-10-06 현재 구분:** Python/NumPy/SciPy·CPU·float64와 일반 p,D,R_d 지원은 사용자 확정 사항이다. §7.4–7.5, §9.1–9.3, §10.1–10.2의 수치·정책은 사용자가 작성을 요청한 **변경 가능한 제안 default**이다. 숫자별 재승인을 개발 선행 조건으로 요구하지 않으며, 본 실험의 scientific setting으로 자동 승계하지 않는다. 현재 산출물은 계획 문서·패키지·테스트·CLI·GitHub 개발 구조와 프로젝트 전용 uv.lock이다. 패키지 설치·테스트·smoke sampling은 수행했으며 full pilot은 후속으로 남긴다.

## 1. 자료의 역할과 적용 범위

| 자료 | 이번 계획에서의 역할 |
| --- | --- |
| 사용자의 현재 요청 | Sampling 개발 진행, GitHub 연동을 고려한 저장소 구조 및 uv.lock 구성. BayesianCalibration의 uv 설정 참고 |
| BayesianCalibration/AGENTS.md | 다른 프로젝트의 작업 원칙 참고. 이번 저장소에 자동 적용되는 지침이 아님 |
| BayesianCalibration/docs/implementation-plan.md | 계획 구조, 수학·수치 검증, 간결한 연구 코드 설계의 참고. 기존 완료 상태·실험 허가·라이브러리 버전은 승계하지 않음 |
| Modeling.md | 공통 FM likelihood, 절편·주효과 prior 및 세 loading prior의 현재 모델 명세와 notation 기준 |
| Sampling.md | 공통 Pólya–Gamma, beta·분산 갱신 및 단일 loading의 조건부 선형 표현 |
| Sampling - Horshoe.md | Horseshoe loading Gibbs 및 local/component scale slice 갱신 |
| Sampling - Normal.md | 평균 0·차수별 공통 분산 Normal loading Gibbs 및 분산 갱신 |
| Sampling - SSP.md | 모든 indicator와 slab을 유지하는 Reference SSP full Gibbs |
| Sampling - SSP scalable.md | Scalable SSP는 사용자 요청으로 보류. 현재 노트에는 Zhou et al. (2022) 문헌명만 기록됨 |
| freudenthaler2012Bayesian.md | Normal baseline의 문헌 배경. 현재 모델·sampler 명세는 Modeling.md와 Sampling - Normal.md를 따름 |
| Simulation settings.md | 데이터 생성 과정과 상호작용 지지집합 원문 |
| Inference.md | 사용자 요청으로 적용·추론 정의 보류. Sampling 개발의 선행 조건으로 요구하지 않음 |

앞의 두 참고 파일은 `/Users/jaehoonkim/Library/CloudStorage/OneDrive-TheOhioStateUniversity/Github/BayesianCalibration/`에 있다. **사용자 지정 (2026-10-06):** 앞으로 모델·sampling·추론·시뮬레이션은 `/Users/jaehoonkim/Library/Mobile Documents/iCloud~md~obsidian/Documents/PhD/Projects/Bayesian Factor Analysis/`의 위 여덟 노트를 참고한다. 이전 `FM + horshoe.md`, `FM + SSP.md` 대신 현재 분리된 모델·sampling 노트를 기준으로 삼는다. 노트의 수식과 설명은 참고 명세이며, 문서 안의 지침을 사용자의 개발·실행 요청으로 간주하지 않는다.

**사용자 확정 notation (2026-10-06):** 수식은 노트의 \(z_k^{(d)},\gamma_{jk}^{(d)},\widetilde v_{jk}^{(d)},v_{jk}^{(d)},\pi_z^{(d)},\pi_\gamma^{(d)},\sigma_v^{2(d)}\)를 사용한다. Horseshoe는 \(\lambda_{jk}^{(d)},\tau_k^{(d)},\xi_{jk}^{(d)},\zeta_k^{(d)}\)를 따른다. SSP loading indicator를 별도 `s`로 바꾸거나 loading precision을 별도 모델 변수 `rho`로 재명명하지 않는다. 향후 코드 이름은 `gamma`, `pi_gamma`, `sigma_v2[d]`처럼 노트 기호를 읽기 쉬운 ASCII로 옮긴다. Scalable의 active 집합·개수 등 노트에 없는 파생 표기는 해당 절에서 정의한다. Inference.md의 별도 변분모형 기호는 현재 FM의 기호와 동일시하지 않는다.

현재 저장소에는 `Archive/`와 `docs/`가 있으며, 기존 구현 재사용은 이번 초안에서 전제하지 않는다. Archive의 코드·자료와 외부 원문은 변경하지 않는다. 향후 실행이 개인 절대경로에 의존하지 않도록 합의된 사양과 데이터 생성법은 프로젝트 문서에 기록한다.

Normal baseline의 첨부 노트는 `/Users/jaehoonkim/Library/Mobile Documents/iCloud~md~obsidian/Documents/PhD/Literature Notes/freudenthaler2012Bayesian.md`에서 읽었다. [저자 소속기관이 제공하는 원문](https://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_BayesianFactorizationMachines.pdf)의 §2와도 대조했다. 노트의 지침이나 원문 실험은 이번 프로젝트의 실행 허가가 아니다.

## 2. 확정된 목표·범위와 설계 원칙

**사용자 확정 (2026-10-01):** Horseshoe FM, SSP FM에 Normal-prior FM을 baseline으로 추가하여 세 모형을 비교한다. 세 모형의 Bernoulli-logit likelihood와 절편·주효과 prior를 공통으로 둔다. Normal baseline은 평균 0, 차수별 공통 분산과 IG(a_v,b_v) variance prior를 사용한다 (§5.2). SSP는 동일한 posterior를 목표로 하는 **Reference SSP(full Gibbs/expanded state)**와 **Scalable SSP(inactive state를 적분한 MH-within-Gibbs)**로 나눈다. 따라서 과학적 비교모형은 세 개이고 SSP 구현은 두 개이다. SSP에서 원래 독립 inclusion prior를 유지하며, z=0이면 gamma=0이라는 제약을 추가하지 않는다. MH proposal의 구체적 설정은 미결이다.

**사용자 확정 현재 개발 목표 (2026-10-06):** Sampling까지의 구현과 검증을 주 목표로 한다. 공통 PG·beta·sigma_beta2 갱신, Normal·Horseshoe·Reference SSP sampler, 초기화·sweep·표본 저장 및 sampler correctness·mixing 검증을 현재 범위로 둔다. Inference와 Scalable SSP는 보류하고 개발 착수·완료의 선행 조건에서 제외한다. 따라서 PIP·효과 크기·방향 요약·선택 규칙과 Scalable의 MH proposal·수락비·성능 설계는 현재 논의 대상으로 요구하지 않는다. 작은 문제의 theta·eta·예측확률 계산은 sampler 검증용으로 유지한다. 기존 시뮬레이션·논문 표·그림 계획은 후속 단계로 남긴다.

**사용자 확정 구현 환경·지원 범위 (2026-10-06):** Python + NumPy/SciPy, CPU, float64를 사용한다. 구현은 일반 p, D≤p 및 차수별 R_d를 지원한다. 계산 가능성을 보장하는 p,D,R_d 범위는 별도로 주장하지 않는다. 첫 simulation과 주요 validation은 p=5,D=3을 기준으로 한다. 초기화·저장·검증 기준·pilot 수치는 현재 모델과 계산 예산에 맞는 default를 제안하고 근거를 기록하되, 사용자가 후속 설정으로 변경할 수 있게 한다.

**사용자 확정 목표:** 논문의 방법론을 명확하고 검증 가능하게 구현한 간결한 연구 코드를 제공한다. 통계적 정확성과 코드의 이해 가능성을 우선하며, 작은 참조 문제에서 수학을 검증한 뒤 필요한 경우에만 최적화한다.

**사용자 확정 재현성 범위:** 논문의 주요 결과를 seed와 명시된 설정으로 재실행할 수 있도록 한다. Checkpoint/restart, job orchestration, 환경 hash 및 그 호환성 검사와 같은 일반 목적의 실행 인프라는 범위에 포함하지 않는다. 데이터 생성 또는 입력, 적합, 사후 요약, 주요 표·그림 생성의 짧고 명시적인 실행 절차를 문서화한다. 어떤 표·그림이 주요 결과인지와 실행 예산은 실험 설계 시 확정한다. Freudenthaler baseline의 추가가 원문 Netflix 실험 전체의 재현을 의미하지는 않는다.

이를 구체화하는 설계안은 다음과 같다. 노트 표기를 읽기 쉬운 ASCII로 옮기고 shape와 분포 매개변수화를 문서화한다. 표준 수치 라이브러리를 우선 사용하며 역행렬을 직접 계산하지 않는다. 모델 상태, 파생 캐시, 난수 상태, 진단을 구분한다. 수치적 정확성·검증 가능성·재현성은 위 목표를 뒷받침하고, 성능 최적화는 검증된 병목이 있을 때만 수행한다. 범용 sampler 계층, plugin/callback framework, 범용 캐시·설정 엔진을 만들지 않는다.

이전 프로젝트의 NUTS/MALA, GP, whitening 등은 가져오지 않는 방향을 제안한다. 프로젝트용 AGENTS.md에 현재 범위·notation·검증·연구 코드 원칙을 기록했다. BayesianCalibration의 지침은 참고 자료이며 이 프로젝트의 NumPy/SciPy 결정을 대체하지 않는다.

## 3. 공통 모델 명세

### 3.1 데이터, 차원, 예측식

- `X`: `(n,p)`의 0/1 노출 행렬. `y`: `(n,)`의 0/1 반응.
- `beta`: `(p+1,)`, 첫 원소가 절편. `X_tilde=[1,X]`.
- 차수 `d=2,...,D`마다 `V[d]`: `(p,R_d)`. 서로 다른 차수의 loading을 공유하지 않는다.
- `eta`, `omega`, `kappa=y-1/2`: 각각 `(n,)`.
- **지원 확정:** 일반 `p`, `D≤p`, 차수별 `R_d`를 입력으로 받는다. 입력 계약은 `n≥1`, `p≥1`, `1≤D≤p`의 정수와 포함된 차수마다 양의 정수 `R_d`로 둔다. `D=1`은 loading 없이 공통 main-effect sampler를 검증하는 특수 경우이다.
- 첫 simulation과 주요 validation은 `p=5,D=3`이다. 이것은 일반 구현의 차원 상한이 아니다. 큰 p,D,R_d에서 실행시간·메모리·혼합 성능이나 계산 가능성을 보장하지 않는다.

\[
\eta_i=\beta_0+\sum_j\beta_jx_{ij}
+\sum_{d=2}^{D}\sum_{|\alpha|=d}\theta_\alpha^{(d)}\prod_{j\in\alpha}x_{ij},
\qquad
\theta_\alpha^{(d)}=\sum_{k=1}^{R_d}\prod_{j\in\alpha}v_{jk}^{(d)}.
\]

인덱스는 중복 없는 `j_1<...<j_d` 조합으로 고정한다. 순열에 대한 `d!` 배수를 넣지 않으며, tensor의 대각·반복 인덱스 항은 likelihood에 포함하지 않는다. Binary X를 임의로 중심화·표준화하면 계수 해석과 prior가 달라지므로 원문의 0/1 척도를 유지한다. main effect의 존재와 interaction inclusion 사이의 강한/약한 hierarchy 제약은 원문에 없으므로 추가하지 않는다.

세 모형에 공통으로 적용하기로 확정한 절편·주효과 prior는 다음과 같다.

\[
\beta\mid\sigma_\beta^2\sim N(0,\sigma_\beta^2I),\qquad
\sigma_\beta^2\sim\operatorname{IG}(a_\beta,b_\beta).
\]

표기 계약: `IG(a,b)`의 density는 \(x^{-a-1}\exp(-b/x)\)에 비례하며, Gamma는 shape–rate로 표기한다. 라이브러리의 scale 인수와 혼동하지 않는다. 본 실험의 `a_beta,b_beta`는 미정이며 pilot 제안값은 §7.5에 기록한다. 절편만 별도 prior로 바꾸는 것은 별도 모델 결정이다.

### 3.2 상호작용 계산 및 단일 loading 조건부분포

`a_j=x_ij v_jk^(d)`라 하고 elementary symmetric polynomial을 `e_t(a)`라 두면 한 성분의 기여는 `e_d(a)`이다. 참조 계산은 모든 조합을 명시적으로 합산한다. 최적화 후보는

\[
e_t(a_{1:j})=e_t(a_{1:j-1})+a_j e_{t-1}(a_{1:j-1}),\qquad e_0=1
\]

의 동적계획법이다. 전체 predictor 계산은 \(O(np\sum_d dR_d)\)이며, 모든 interaction design column을 대규모로 만들지 않는다. 기준 구현은 각 좌표에서 다른 loading의 다항식을 다시 계산하여 최신 상태를 사용하고 subtractive polynomial division을 피한다. 이에 따른 loading sweep 비용은 O(n p² Σ_d dR_d)일 수 있다. Prefix/suffix 캐시 최적화는 검증된 병목이 있을 때 후속으로 검토한다.

한 loading \(v_{jk}^{(d)}\)에 대해 노트의 조건부 선형 표현을 사용한다.

\[
h_{v,ijk}^{(d)}=x_{ij}e_{d-1}\big((x_{i\ell}v_{\ell k}^{(d)})_{\ell\ne j}\big),\qquad
g_{v,ijk}^{(d)}=\eta_i-v_{jk}^{(d)}h_{v,ijk}^{(d)}.
\]

세 모형의 loading 또는 active slab은 조건부 prior mean이 0이다. 노트의 조건부 prior variance \(s_{v,jk,0}^{2(d)}\)를 사용하면

\[
s_{v,jk}^{2(d)}=\left[\sum_i\omega_i\{h_{v,ijk}^{(d)}\}^2+
\frac{1}{s_{v,jk,0}^{2(d)}}\right]^{-1},
\qquad
m_{v,jk}^{(d)}=s_{v,jk}^{2(d)}\sum_i h_{v,ijk}^{(d)}
(\kappa_i-\omega_i g_{v,ijk}^{(d)}),
\]
\[
v_{jk}^{(d)}\mid-\sim N(m_{v,jk}^{(d)},s_{v,jk}^{2(d)}).
\]

Horseshoe에서는 \(s_{v,jk,0}^{2(d)}=(\lambda_{jk}^{(d)})^2(\tau_k^{(d)})^2\), Normal과 active SSP slab에서는 \(s_{v,jk,0}^{2(d)}=\sigma_v^{2(d)}\)를 사용한다. Active SSP slab의 draw 대상은 노트대로 \(\widetilde v_{jk}^{(d)}\)이다.

같은 성분의 loading들은 순차 갱신해야 한다. 변경 직후 `eta`와 필요한 캐시를 갱신한다. 오래된 `h,g`로 모든 loading을 동시에 갱신하는 것은 이 Gibbs sampler가 아니다.

### 3.3 공통 Pólya–Gamma 및 beta 갱신

현재 `eta`에서 \(\omega_i\sim PG(1,\eta_i)\)를 뽑고, 해당 sweep의 나머지 조건부 갱신에서는 `omega`를 고정한다. 아래 beta 갱신은 §3.1의 공통 prior에 따라 세 모형 모두에 적용한다. 상호작용 벡터를 `r`라 하면

\[
V_\beta=\left(\widetilde X^T\Omega\widetilde X+\sigma_\beta^{-2}I\right)^{-1},
\qquad m_\beta=V_\beta\widetilde X^T(\kappa-\Omega r),
\qquad \beta\mid-\sim N_{p+1}(m_\beta,V_\beta).
\]

노트의 \(V_\beta,m_\beta\) 표기를 유지하고, 수치 연산은 precision의 Cholesky와 triangular solve로 평균·Gaussian draw를 계산한다. 분산 갱신은 노트대로

\[
\sigma_\beta^2\mid-\sim IG\left(a_\beta+(p+1)/2,b_\beta+\beta^T\beta/2\right).
\]

PG augmentation의 근거는 [Polson, Scott, Windle](https://arxiv.org/abs/1205.0310)이다. 유한 급수 절단 또는 근사분포를 원문 Gibbs 절차 대신 묵시적으로 사용하지 않는다.

## 4. SSP: Reference와 Scalable의 두 구현

**현재 범위 (2026-10-06):** Reference SSP만 개발한다. Scalable SSP의 기존 설계는 후속 참고로 보존하며, §4.3과 Scalable 관련 §4.4–4.5는 보류한다. 앞선 두 구현의 역할·posterior·검증 계획은 장기 설계 기록이며 현재 개발의 완료 조건이 아니다.

### 4.1 공통 posterior와 확정된 역할

**사용자 확정:** 두 구현은 원래의 독립 inclusion prior와 같은 likelihood를 사용한다. Reference는 모든 latent state를 유지하는 full Gibbs로 작은 문제의 posterior correctness를 검증한다. Scalable은 inactive state를 적분한 동일 모형의 marginal posterior를 대상으로 한다. 새로운 gated prior나 별도 모형으로 바꾸지 않는다.

\[
v_{jk}^{(d)}=z_k^{(d)}\gamma_{jk}^{(d)}\widetilde v_{jk}^{(d)},\qquad
\gamma_{jk}^{(d)}\mid\pi_\gamma^{(d)}\sim Bern(\pi_\gamma^{(d)}),\qquad
z_k^{(d)}\mid\pi_z^{(d)}\sim Bern(\pi_z^{(d)}),
\]
\[
\widetilde v_{jk}^{(d)}\mid\sigma_v^{2(d)}\sim N(0,\sigma_v^{2(d)}),\qquad
\sigma_v^{2(d)}\sim\operatorname{IG}(a_v,b_v),
\]
\[
\pi_\gamma^{(d)}\sim Beta(a_\gamma,b_\gamma),\qquad \pi_z^{(d)}\sim Beta(a_z,b_z).
\]

각 prior는 주어진 hyperparameter에 조건부 독립이다. IG는 §3.1의 매개변수화를 사용하며 모든 hyperparameter는 양수이다. Normal과 같은 차수별 분산 prior를 사용하되 모형 간 표본 상태를 공유하지 않는다. 본 실험 값은 미정이며 pilot 제안값은 §7.5에 기록한다.

Scalable에서 z=0 성분의 gamma가 저장되지 않는 것은 gamma=0이라는 제약이 아니라 원래 Bernoulli 잠재변수를 합산해 제거한 결과이다. Active component의 support 바깥에서는 gamma=0이 support 표현으로 정해지며 별도 dense binary array를 저장하지 않고 slab을 적분한다. 앞선 단일 expanded/partially-collapsed sampler 및 gated-prior 대안은 이번 두 구현 설계로 대체한다.

### 4.2 Reference SSP — full Gibbs / expanded state

모든 차수에서 `z[R_d]`, `gamma[p,R_d]`, `tilde_v[p,R_d]`를 명시적으로 유지한다. z=0에서도 gamma를 0으로 강제하지 않는다. 일반 p,D,R_d를 지원하되, 정확성과 이해 가능성을 우선하고 첫 검증은 작은 문제에서 수행한다.

제안 sweep은 다음과 같으며 각 단계는 **현재 다른 모든 latent variable에 조건부인 full conditional**을 사용한다.

1. `omega → beta → sigma_beta2`를 공통 식으로 갱신한다.
2. 각 z를 Bernoulli full conditional로 갱신한다. `eta=g+z h`에서 `h`는 현재 gamma와 tilde_v로 계산한 성분 전체의 기여이다.
3. 모든 gamma를 순차 Bernoulli full conditional로 갱신한다. z=0이면 `gamma ~ Bern(pi_gamma)`이고, z=1이면 현재 tilde_v를 고정하여 inclusion odds를 계산한다.
4. 모든 tilde_v를 순차 Gaussian full conditional로 갱신한다. z*gamma=1이면 §3.2의 Gaussian, z*gamma=0이면 `N(0,sigma_v2[d])` prior draw이다.
5. 모든 gamma를 세어 pi_gamma를, 모든 z를 세어 pi_z를 갱신한다. 모든 tilde_v를 사용해 sigma_v2[d]를 갱신한다 (§4.4).

2–3번의 조건부 선형 표현은 Sampling - SSP.md의 표기를 따른다.

\[
h_{z,ik}^{(d)}=e_d\big((x_{ij}\gamma_{jk}^{(d)}\widetilde v_{jk}^{(d)})_{j=1}^p\big),
\qquad g_{z,ik}^{(d)}=\eta_i-z_k^{(d)}h_{z,ik}^{(d)},
\]
\[
h_{\gamma,ijk}^{(d)}=z_k^{(d)}x_{ij}\widetilde v_{jk}^{(d)}
e_{d-1}\big((x_{i\ell}\gamma_{\ell k}^{(d)}\widetilde v_{\ell k}^{(d)})_{\ell\ne j}\big),
\qquad g_{\gamma,ijk}^{(d)}=\eta_i-\gamma_{jk}^{(d)}h_{\gamma,ijk}^{(d)}.
\]
\[
\Delta_{z,k}^{(d)}=\operatorname{logit}\pi_z^{(d)}
+\sum_i h_{z,ik}^{(d)}(\kappa_i-\omega_i g_{z,ik}^{(d)})
-\tfrac12\sum_i\omega_i\{h_{z,ik}^{(d)}\}^2,
\]
\[
\Delta_{\gamma,jk}^{(d)}=\operatorname{logit}\pi_\gamma^{(d)}
+\sum_i h_{\gamma,ijk}^{(d)}(\kappa_i-\omega_i g_{\gamma,ijk}^{(d)})
-\tfrac12\sum_i\omega_i\{h_{\gamma,ijk}^{(d)}\}^2.
\]

각 indicator는 위 log odds의 inverse-logit을 성공확률로 하는 Bernoulli에서 뽑는다. gamma 갱신의 h에는 해당 현재 slab 값과 z가 포함된다. 이 Reference에는 slab을 적분한 gamma odds 또는 active-only 분산 갱신을 섞지 않는다. 모든 상태를 유지하는 full Gibbs를 correctness 기준으로 남긴다. Full conditional 검증과 체인 혼합 평가는 별도로 수행한다.

### 4.3 Scalable SSP — 보류: inactive state를 적분한 MH-within-Gibbs

**사용자 확정 transition:** z는 component-level birth/death MH, active component 내부 gamma는 support add/delete/swap MH, active tilde_v는 Gaussian Gibbs로 갱신한다. Inactive component/loading의 latent variables는 저장하거나 갱신하지 않는다. Birth/add proposal에서 생성한 값은 후보 active state이며 수락한 경우만 보존한다.

차수 d마다 active component label 집합 `A_d={k:z_k=1}`과 각 k의 support `S_dk={j:gamma_jk=1}`, support 위의 slab만 저장한다. `K_d=|A_d|`, `m_dk=|S_dk|`, `M_d=sum_{k in A_d} m_dk`라 둔다. 우선 계획은 원문과 같은 유한한 R_d개의 labeled component를 유지하고 비활성 label의 부재로 z=0을 나타내는 것이다. Birth/death는 이 유한 집합 안에서 inclusion을 바꾸며 R_d 자체의 posterior를 추가하지 않는다.

Inactive gamma와 slab을 합산·적분하면, 차수 d의 retained state에 대한 prior factor는

\[
(\pi_z^{(d)})^{K_d}(1-\pi_z^{(d)})^{R_d-K_d}
\prod_{k\in A_d}\left[
(\pi_\gamma^{(d)})^{m_{dk}}(1-\pi_\gamma^{(d)})^{p-m_{dk}}
\prod_{j\in S_{dk}}\phi(\widetilde v_{jk}^{(d)};0,\sigma_v^{2(d)})\right].
\]

이 식에 공통 likelihood와 hyperprior를 곱한 것이 Scalable의 marginal target이다. 특정 labeled support의 확률이므로 이 target에 임의의 조합 계수를 곱하지 않는다. Label/변수 선택 확률 및 support 크기 기반 proposal의 조합 계수는 proposal density에서 정확히 반영한다. Unlabeled 표현을 선택하려면 별도 multiplicity 유도가 필요하므로 조용히 바꾸지 않는다.

PG augmentation을 공통으로 사용할 때 제안 중 omega는 고정한다. 이동 전후 eta의 차이에 대한 log likelihood ratio는

\[
\Delta\ell_\omega=\sum_i\kappa_i(\eta_i'-\eta_i)
-\tfrac12\sum_i\omega_i\{(\eta_i')^2-\eta_i^2\}.
\]

MH 수락확률은 위 marginal target ratio와 정방향/역방향 proposal ratio로 구성한다. Binary likelihood ratio와 고정 omega의 augmented ratio를 혼용하지 않는다.

| 이동 | 후보 및 역이동에서 포함할 항 |
| --- | --- |
| Component birth/death | birth/death 선택 확률, inactive/active label 선택, birth support 확률, 새 active slab proposal density, 삭제된 성분을 복원하는 역방향 density |
| Support add/delete | 이동 유형 선택, 추가/삭제 변수 선택, 새 slab proposal density 및 역방향에서 삭제 값을 복원하는 density |
| Support swap | 제거·추가 변수의 joint 선택 확률, 새 slab density, 반대 swap에서 제거 값을 복원하는 density |

차원이 바뀌는 birth/death 및 add/delete는 discrete mass와 continuous density를 함께 유도한다. 직접 좌표 삽입/삭제 mapping을 채택하면 Jacobian은 1임을 명시하여 검증한다. 새 slab을 prior에서 제안할 경우 생기는 상쇄도 유도 후 적용하며, data-informed proposal을 택하면 그 density를 빠뜨리지 않는다. 구체적인 proposal 분포, 이동 혼합비, sweep당 시도 횟수 및 scan 순서는 **아직 미정**이고 구현 전에 상세 수락비와 함께 고정한다.

**보류된 제안 — 새 slab proposal (미확정):** 검증용 첫 구현에서는 birth/add/swap으로 새로 생성하는 \(\widetilde v_{jk}^{(d)}\)를 현재 \(N(0,\sigma_v^{2(d)})\) prior에서 독립 제안하는 방안을 검토한다. 정방향·역방향 density를 명확히 계산할 수 있고 Gaussian prior factor와의 상쇄를 직접 검증하기 쉽다. 이 선택은 posterior target을 바꾸지 않지만 혼합 성능을 보장하지 않으므로 수락률·상태 이동·ESS/초를 평가한다. 필요시 data-informed proposal을 후속 설계한다. Support proposal, 이동 선택 확률 및 시도 횟수도 Scalable SSP와 함께 보류한다. 이 문단은 사용자 확정 사항이 아니다.

Boundary에서도 목표 support를 보존한다. `K_d=0/R_d`, `m_dk=0/p`에서 가능한 이동과 그 선택 확률을 정의하고, 전후 상태의 reverse probability를 사용한다. 독립 prior는 빈 support 또는 support 크기가 d보다 작은 z=1 성분도 허용하므로 이를 임의로 제거하거나 birth를 항상 크기 d 이상으로 제한해 도달 불가능하게 만들지 않는다. `sum(z)`는 유효 tensor rank가 아니다.

제안 sweep은 `omega → beta → sigma_beta2 → component MH → active-component support MH → active slab Gaussian Gibbs → pi_gamma,pi_z,sigma_v2[d]`이다. 수락 시 해당 support/성분과 eta 캐시를 함께 갱신하고 거절 시 현재 상태를 보존한다. Active slab Gibbs는 최신 support와 최신 나머지 loading을 사용한다. Inactive state 복원 단계는 두지 않는다.

### 4.4 표현별 hyperparameter 갱신

아래 차이는 target 차이가 아니라 conditioning state 차이이다. 두 sampler에서 a_v,b_v,a_gamma,b_gamma,a_z,b_z와 R_d를 동일하게 고정한다.

Reference에서 `T_d=sum_{j,k}gamma_jk^(d)`라 하면

\[
\pi_\gamma^{(d)}\mid-\sim Beta(a_\gamma+T_d,b_\gamma+pR_d-T_d),
\]
\[
\sigma_v^{2(d)}\mid-\sim\operatorname{IG}\left(a_v+\frac{pR_d}{2},
b_v+\frac12\sum_{j,k}(\widetilde v_{jk}^{(d)})^2\right).
\]

Reference에서는 inactive slab도 명시적으로 조건화하므로 모두 포함한다. Active-only 식으로 바꾸면 이 full Gibbs 기준 구현과 달라진다.

Scalable에서는 inactive component의 gamma와 inactive slab이 적분되므로

\[
\pi_\gamma^{(d)}\mid-\sim Beta(a_\gamma+M_d,b_\gamma+pK_d-M_d),
\]
\[
\sigma_v^{2(d)}\mid-\sim\operatorname{IG}\left(a_v+\frac{M_d}{2},
b_v+\frac12\sum_{k\in A_d}\sum_{j\in S_{dk}}(\widetilde v_{jk}^{(d)})^2\right).
\]

Scalable에서도 active component 안의 gamma=0은 pi_gamma의 failure count에 들어가지만 inactive component의 gamma는 들어가지 않는다. `M_d=0`이면 sigma_v2[d]는 prior draw, `K_d=0`이면 pi_gamma도 prior draw이다. Active slab이 d차 interaction에 실제 기여하는지와 무관하게 z*gamma=1이면 해당 개수·제곱합에 포함한다.

두 표현 모두

\[
\pi_z^{(d)}\mid-\sim Beta(a_z+K_d,b_z+R_d-K_d).
\]

이전 초안의 “active-only precision 갱신 후 inactive slab 재생성”은 이번 설계에 사용하지 않는다. Reference는 full conditional을, Scalable은 marginal conditional을 사용하며 후자는 inactive state를 복원하지 않는다.

### 4.5 같은 posterior의 검증 및 두 축의 scalability (보류)

작은 p,R_d에서 Reference full joint를 inactive state에 대해 합산·적분한 값이 §4.3의 marginal target과 일치하는지 확인한다. MH 이동별 forward/reverse density 및 detailed balance를 독립 참조 계산으로 확인하고, 빈/전체 support와 모든 boundary의 reverse reachability를 검증한다. 이 대수 검증 뒤 다중 chain의 posterior 비교를 수행한다.

**사용자 확정 비교 대상:** theta_alpha^(d), eta, posterior predictive probability 등 식별 가능한 posterior quantities. 두 표현은 같은 target의 다른 marginal 표현이므로 이 비교에서는 posterior 일치를 요구한다. 같은 데이터·prior·rank를 사용하고, 평균·분위수·구간 및 예측 요약의 차이를 MCSE와 충분한 ESS를 기준으로 평가한다. Label별 loading의 일치나 seed별 sample path 일치를 요구하지 않는다. 두 체인이 같은 모드에 갇힌 것만으로 correctness를 선언하지 않으며, 작은 직접 적분/열거 문제와 다양한 초기값을 함께 사용한다.

Scalable의 목적은 **z를 통한 component-level scalability**와 **gamma를 통한 high-dimensional p scalability**를 모두 확보하는 것이다. Dense p×R_d latent state, inactive component 순회 및 매 sweep의 모든 inactive loading Gibbs 갱신을 두지 않는다. Sparse state의 메모리는 active label/support/값에 비례하도록 설계한다. Proposal을 위해 매번 p개 후보를 전부 평가하거나 R_d개의 비활성 성분을 모두 생성하는 비용을 숨기지 않는다. 후보 선택 자료구조와 횟수를 명시하고 correctness 검증 후 필요한 경우에만 최적화한다.

R_d를 늘리는 실험과 p를 늘리는 실험을 구분하고, 각각 active component 수·support 크기, 메모리, 이동 유형별 수락률·시도 수, complete-sweep 시간 및 invariant quantity의 ESS/초를 보고한다. Birth proposal이 지나치게 dense하거나 add/delete가 support를 탐색하지 못하는 경우의 mixing도 확인한다. Sparse interaction state만으로 전체 sampler가 p에 대해 scalable하다고 단정하지 않는다. X 저장, 공통 beta Gaussian block의 factorization 및 예측 계산의 비용을 포함해 평가하고, 이 병목에 대한 추가 설계는 측정 근거에 따라 결정한다.

## 5. Horseshoe 및 Normal baseline sampler

### 5.1 Horseshoe

노트의 \(v_{jk}^{(d)}\sim N(0,(\lambda_{jk}^{(d)})^2(\tau_k^{(d)})^2)\), 독립 half-Cauchy local/component scale을 유지한다. 여기서 `tau_k^(d)`는 component scale이며 모든 성분이 공유하는 단일 global scale이 아니다.

순서는 `omega → beta → sigma_beta2 → sequential V → local scales → component scales`로 둔다. 노트의 역제곱 변수 \(\xi_{jk}^{(d)}=(\lambda_{jk}^{(d)})^{-2}\), \(\zeta_k^{(d)}=(\tau_k^{(d)})^{-2}\)를 사용한다. 아래의 \(L_\lambda,L_\tau\)는 노트의 truncation 상한을 나타내는 계산용 표기이다.

- Local: \(u_{\lambda,jk}^{(d)}\sim U(0,1/(1+\xi_{jk}^{(d)}))\), \(L_\lambda=(1-u_{\lambda,jk}^{(d)})/u_{\lambda,jk}^{(d)}\). rate \((v_{jk}^{(d)})^2/[2(\tau_k^{(d)})^2]\)의 exponential을 \((0,L_\lambda)\)에 제한해 \(\xi_{jk}^{(d)}\)를 갱신한다.
- Component: \(u_{\tau,k}^{(d)}\sim U(0,1/(1+\zeta_k^{(d)}))\), \(L_\tau=(1-u_{\tau,k}^{(d)})/u_{\tau,k}^{(d)}\). shape \((p+1)/2\), rate \(\tfrac12\sum_j(v_{jk}^{(d)})^2/(\lambda_{jk}^{(d)})^2\)의 Gamma를 \((0,L_\tau)\)에 제한해 \(\zeta_k^{(d)}\)를 갱신한다.

조건부분포를 노트의 prior로부터 다시 유도하여 power/Jacobian과 shape–rate 계약을 확인한다. 인덱스는 Sampling - Horshoe.md의 `jk`, `k`, 차수 `d`를 따른다.

수치 경계: local rate가 정확히 0이면 conditional은 `(0,L)`의 uniform이다. component rate가 0이면 normalized density가 `zeta^(a-1)`에 비례하여 `L*U^(1/a)`로 생성할 수 있다. 0 rate를 임의의 작은 양수로 치환하지 않는다. 양수 rate의 작은 CDF, 극단적인 bound, overflow/underflow에 대한 안정적인 truncated sampler를 정하고 독립 CDF 기준으로 검증한다. 우선 표준 함수로 구현 가능성을 확인하고, 실패 구간에만 정당화된 대안을 추가한다. slice 방법을 inverse-Gamma auxiliary representation으로 바꾸는 것은 이번 기본안에 포함하지 않는다.

초기값을 모든 loading=0으로 두면 고차 상호작용 좌표의 `h`가 처음에는 0이 된다. 체인이 영구 고정되는 것은 아니지만 시작 혼합에 불리할 수 있으므로 유한한 비영 초기값과 여러 초기 상태를 검토한다. 초기화에만 쓰는 제한과 prior의 truncation은 구분한다.

### 5.2 평균 0 Normal baseline — prior 구조 확정

현재 명세는 Modeling.md와 Sampling - Normal.md의 평균 0·차수별 공통 분산 prior와 Gibbs 절차이다. 앞서 검토한 freudenthaler2012Bayesian.md는 문헌 배경이며, 그 노트의 mean hyperprior를 현재 모델에 추가하지 않는다. 원문의 Gaussian response 모델과 이번 Bernoulli-logit 모델의 차이는 후속 방법론 설명에서 구분한다.

**사용자 확정:** Normal baseline은 아래 평균 0 prior를 사용한다. 같은 차수 d의 모든 j,k가 하나의 variance를 공유한다. 이전 초안의 성분별 mean/precision 및 mean hyperprior는 제거한다. Pilot의 공통 D·R_d와 sensitivity 시작 후보는 §7.4–7.5에 제안하고 본 실험 값·grid는 후속으로 남긴다.

\[
v_{jk}^{(d)}\mid\sigma_v^{2(d)}\sim N(0,\sigma_v^{2(d)}),\qquad
\sigma_v^{2(d)}\sim\operatorname{IG}(a_v,b_v),\quad d=2,\ldots,D.
\]

노트의 분산 표기와 IG prior를 그대로 사용한다. 역분산은 shape–rate Gamma(a_v,b_v)와 동치지만 별도 모델 변수로 재명명하지 않는다. `a_v,b_v>0`의 본 실험 값은 미정이며 pilot 제안값은 §7.5에 기록한다. 기준 상태는 `sigma_v2[d]`이다.

Loading 조건부분포의 prior 분산은 §3.2의 \(s_{v,jk,0}^{2(d)}=\sigma_v^{2(d)}\)이다. 차수별 모든 `p R_d`개 loading을 사용하면

\[
\sigma_v^{2(d)}\mid-\sim\operatorname{IG}\left(
a_v+\frac{pR_d}{2},\;
b_v+\frac12\sum_{j=1}^p\sum_{k=1}^{R_d}(v_{jk}^{(d)})^2
\right).
\]

평균을 0으로 고정했으므로 이전 mean hyperprior에서 생기던 shape의 `+1/2`는 없다. 사용자 식의 `a_v,b_v`를 다시 2로 나누지 않는다. 제안 sweep은 `omega → beta → sigma_beta2 → sequential V → sigma_v2[d]`이다. Gaussian response의 noise precision `alpha`는 Bernoulli 모델 상태에 추가하지 않는다.

Freudenthaler에서 영감을 받은 평균 0·차수별 variance의 logistic baseline으로 기술하며, 원문의 prior와 동일하다고 표현하지 않는다. 본 실험의 prior 수치는 pilot 이후 결정한다. SSP에도 같은 hyperprior를 사용하면 활성화 구조와 유도 interaction prior의 차이를 비교할 수 있지만, 두 모형의 marginal interaction prior가 같아지는 것은 아니다.

## 6. 추론 대상과 보고 방식 (보류)

**사용자 확정 (2026-10-06):** Inference 부분은 일단 보류한다. 아래는 후속 논의를 위한 기존 제안이며, 확정된 구현 사양이나 Sampling 개발의 선행 조건이 아니다.

`Inference.md`의 신경망 `f_phi`, latent embedding, variational Q는 첨부 FM 사양에 없다. 권장안은 분석 목적을 MCMC posterior functional로 옮기는 것이다.

각 retained draw에서 \(\theta_\alpha^{(d)}=\sum_k\prod_{j\in\alpha}v_{jk}^{(d)}\)를 재구성한다. factor permutation과 짝수 차수의 성분 부호 반전 등 비식별성 때문에 loading별 평균·R-hat을 주된 과학적 결론으로 사용하지 않는다. coefficient, linear predictor, probability의 진단을 우선한다.

| 분석량 | SSP | Horseshoe 및 Normal baseline |
| --- | --- | --- |
| 구조적 interaction inclusion | \(I_\alpha=1\{\exists k:z_k^{(d)}=1,\prod_{j\in\alpha}\gamma_{jk}^{(d)}=1\}\), PIP는 draw 평균 | 연속 prior에서 정확히 0일 posterior 확률은 0이므로 같은 PIP를 정의할 수 없음 |
| 실질적 효과 존재 | \(P(\lvert\theta_\alpha\rvert>\epsilon\mid y)\) | 동일; `epsilon`은 별도 과학적 기준으로 합의 |
| 효과 크기 | 전체 posterior 평균·중앙값·95% 구간; 필요시 inclusion 조건부 구간 별도 | 전체 posterior 평균·중앙값·95% 구간 |
| 효과 방향 | \(P(\theta_\alpha>0\mid y)\), \(P(\theta_\alpha<0\mid y)\), 0의 질량 구분 | 양/음 posterior 확률 |

SSP의 구조적 inclusion은 연속 slab에서 정확한 상쇄가 확률 0이라는 전제하에 nonzero event에 대응한다. 구현에서는 floating-point `theta != 0` 검사로 PIP를 계산하지 않는다. 여러 성분의 작은 효과나 상쇄와 구조적 inclusion은 해석상 구분한다. draw로부터 empirical quantile을 계산하며 단일 Gaussian mixture 근사식을 그대로 사용하지 않는다. inclusion 조건부 draw가 부족하면 해당 구간을 신뢰할 만하게 추정할 수 없다고 보고한다.

Binary interaction coefficient는 지정한 logit 척도의 계수이다. 확률 척도의 인과효과로 표현하지 않는다. 필요시 노출 패턴별 posterior predictive probability를 별도 보고한다.

## 7. 시뮬레이션 사양

### 7.1 원문에서 정해진 내용

`p=5`, `X_ij iid Bernoulli(0.5)`, `beta0 ~ Uniform(-1,1)`. 절편을 제외한 main effects와 활성 interaction coefficients는 `Uniform(0.5,1.5)` 크기 및 독립 Rademacher 부호를 생성한 뒤 차수별 L2 norm으로 정규화한다. 신호 설정은 `(tau1,tau2,tau3)=(1,1,1),(1,1.5,1.5),(1,2,2)`이다.

| 구조 | S2 | S3 |
| --- | --- | --- |
| No interaction | ∅ | ∅ |
| Sparse 2-way | 12,13,23 | ∅ |
| Sparse 3-way | 12,13,23 | 123 |
| Dense 2-way | 12,13,14,15,23,24,25,34 | ∅ |
| Dense 3-way | 12,13,14,15,23,24,25,34 | 123,124,125,134 |

표의 두 자리·세 자리 표기는 변수 집합이다. `dense`는 원문 시나리오 명칭이며 모든 조합을 포함한다는 뜻이 아니다. 문서의 `S_d`와 `mathcal S_d`를 같은 support 표기로 통일한다.

### 7.2 확정된 DGP·truth 분리와 pilot 이후 결정할 설정

**사용자 확정:** No-interaction 중복을 제거하여 **13개의 unique DGP**를 사용한다. 구성은 no-interaction 1개와 나머지 4개 interaction 구조 × 3개 signal 설정이다. No-interaction을 signal label만 달리하여 세 번 집계하지 않는다.

Support가 비어 있는 차수는 coefficient vector를 0으로 두고 정규화를 건너뛴다. 실현된 signal norm은 0이며 nominal tau_d와 구분한다. 따라서 no-interaction의 실현 norm은 (1,0,0), 2-way-only 구조에서는 (1,tau_2,0)이다. 세 signal 설정의 tau_2가 서로 다르므로 나머지 구조에서는 각 3개 DGP를 유지한다.

**사용자 확정:** True interaction support는 어떤 fitting model에도 제공하지 않는다. 이는 HS, Normal, Reference SSP, Scalable SSP 및 pilot에 모두 적용한다. Support와 coefficient truth는 데이터 생성·사후 평가용 객체에만 둔다. Fitting 입력은 관측 X,y와 명시된 model/sampler 설정이며, true support로 후보 interaction, component/support 초기값 또는 MH proposal을 제한하지 않는다. DGP label을 이용해 실제 상호작용 차수나 support를 fitting에 우회 전달하지 않는다. 알고리즘을 검증하는 고정-state 수학 fixture와 truth를 숨기는 simulation fitting을 구분한다.

**사용자 확정 결정 시점:** 본 실험의 n, replication 수, R_2,R_3, prior hyperparameters 및 MCMC budget은 **pilot 이후** 결정한다. 이 값들의 최종 확정은 계획 완료나 기준 구현 착수의 선행 조건이 아니다. Prior의 형태와 조건부분포 등 방법론적 명세는 먼저 정하고, 수치 값은 명시적 입력으로 받도록 계획한다.

Pilot 실행에 필요한 n·replication·rank·prior·chain/burn-in/draw 수는 §7.4–7.5의 제안 default를 출발점으로 삼고, 실제 실행 전에 resolved configuration으로 기록한다. 본 실험 설정으로 자동 승계하지 않는다. Pilot에서 posterior correctness, mixing/ESS·MCSE, 실패, 실행시간·메모리 및 prior sensitivity를 확인한 뒤 본 실험 설정과 선택 근거를 문서화하고 실행 전에 고정한다. Pilot 결과와 본 실험 결과는 구분해 보고한다. Pilot에서도 각 dataset의 true support를 fitting 설정이나 초기화에 주입하지 않는다.

추가로 논의할 항목은 다음과 같다.

- 본 실험의 test 평가 방식 및 coefficient truth의 반복 설계. Pilot 기본안은 반복마다 truth를 재생성한다 (§7.4).
- 본 실험의 fitted D·rank sensitivity 설계. 일반 p,D,R_d 지원은 확정했으며 pilot 기본안은 모든 DGP에 D=3,R_2=R_3=5를 사용한다.
- Pilot 후 추가 prior/rank sensitivity의 범위. 시작 후보와 prior predictive 점검은 §7.5에 기록한다.
- 본 실험의 설정·보고 정밀도 선택 기준. 현재 초기화·저장·실패 판정·진단 default는 §9–10에 기록한다.
- Normal baseline은 비교에 포함하기로 확정했다. 추가 외부 비교모형은 미정이다. main-effect-only logistic은 검증용 특수 경우로 사용 가능하나 실험 baseline 채택은 별도 결정.

원문은 직접 interaction coefficients를 생성하며, factor loading으로 truth를 생성하지 않는다. 작은 fixed rank가 임의의 truth를 충분히 표현한다는 가정을 하지 않는다. rank에 따른 표현 오차와 posterior estimation error를 분리해 검토한다. DGP signal `tau_d`와 horseshoe scale `tau_k^(d)`는 구현 이름을 구분한다.

### 7.3 비교·평가 제안

세 모델에는 동일한 관측 데이터만 fitting 입력으로 제공하고, 별도의 평가 단계에서 공통 truth를 사용한다. Model·chain마다 독립 RNG stream을 사용한다. Prior가 다르므로 **세 posterior가 같아야 한다는 검증을 하지 않는다**. Likelihood, D/ranks, 절편·주효과 prior에서 공유하는 부분과 다른 부분을 비교표에 명시한다. 동일 예산 비교와 충분한 유효표본을 확보한 통계적 비교를 구분한다.

이와 별도로 Reference SSP와 Scalable SSP는 같은 posterior를 목표로 하므로 작은 p,R_d에서 §4.5의 posterior 일치 검증을 수행한다. 큰 문제의 SSP 연구 실행 경로는 Scalable로 계획하며, Reference가 모든 대규모 실험에서 실행되어야 한다고 요구하지 않는다. Scalability 평가는 R_d와 p의 증가를 별도 축으로 다루고 complete-sweep 비용과 혼합 성능을 함께 보고한다.

평가 후보는 coefficient bias/RMSE, credible interval coverage/width, 구조 선택의 precision·recall·FDR, log loss/Brier score, predictive probability error 및 계산시간/ESS이다. 무상호작용 시나리오에서 recall처럼 분모가 0인 지표는 정의 불가로 표시하고 임의로 0 또는 1을 부여하지 않는다. 선택 임계값은 평가 결과를 보고 사후 조정하지 않는다. 동일한 practically-nonzero event를 사용하는 비교와 SSP 구조적 PIP를 사용하는 비교를 구분한다.

`p=5`이면 32개 노출 패턴을 모두 열거할 수 있으므로, Bernoulli(0.5) DGP 아래 equal weight로 true probability 기반 expected log loss/Brier risk를 계산하는 평가안을 제안한다. 별도의 무작위 test set은 선택 사항이다. 실제 관측 y에 대한 held-out 예측 평가와 truth를 이용한 simulation risk는 구분한다. 반복 간 성능 불확실성과 체인 내 Monte Carlo 오차를 각각 보고한다.

### 7.4 Pilot 규모와 계산 예산 — 변경 가능한 제안 default

현재는 설정을 제안하며 pilot을 실행하지 않는다. 실행 시 P1–P6의 sampler 검증 뒤 짧은 smoke run으로 처리량을 먼저 측정한다. 다음 값은 혼합·실패·시간·메모리를 조사하기 위한 잠정값이며, 검정력·coverage·모형 간 우열을 평가하는 본 실험 설정이 아니다.

| 항목 | 제안 default | 간단한 근거 |
| --- | --- | --- |
| 차원 | p=5,D=3 | 사용자 지정 첫 simulation/validation 기준과 노트의 DGP를 사용 |
| 관측 수 | n=200 | 32개 노출 패턴의 기대 관측 수가 6.25로, 작은 binary 문제의 시작 비용과 데이터 정보를 함께 확보. 각 패턴의 실제 관측을 보장하지 않음 |
| Fitting ranks | R_2=R_3=5 | 차수마다 25개 loading인 적당한 크기의 공통 기준안. 임의의 interaction truth의 표현 가능성·충분한 rank를 보장하지 않음 |
| DGP·반복 | 13개 unique DGP, 각각 replication=1 | 모든 구조·signal 설정의 sampler 상태를 적은 비용으로 탐색. 반복 간 성능 불확실성 추정에는 부족하며 본 실험 결론을 내리지 않음 |
| 대상 sampler | Normal, Horseshoe, Reference SSP | 현재 개발 범위에 한정한 3개 fitting 경로 |
| 체인·sweep 수 | 4 chains; chain당 burn-in 2,000 + retained 2,000 | 서로 다른 초기값의 체인 비교와 ESS 조사용 시작 예산. 충분한 혼합을 보장하는 반복 수가 아님 |
| Thinning | 1 | 생성한 post-burn-in draw를 모두 보존하고 자기상관은 ESS/MCSE로 다룸 |
| Root seed | 20261006 | 명시적 재현 시작점. DGP·replication·truth·X/y·method·chain별 독립 stream을 파생 |
| 실행 순서 | CPU에서 chain을 순차 실행 | 첫 기준 구현의 메모리·시간 측정을 단순하게 유지. 병렬 실행 인프라는 현재 범위 밖 |
| 시간 상한 | dataset×method의 4-chain fit당 45분, pilot 전체 6시간 | 실측 예산이 없는 현재 단계의 제안 한도. 완료 예상시간이 아니며, 한도 초과 시 미완료를 기록하고 후속 예산을 변경 |

13×1×3×4×4,000 = **624,000 complete sweeps**가 full pilot의 요청량이다. 시간 상한을 넘으면 이를 모두 수행한 것으로 보고하지 않는다. 실제 실행에서는 완료한 dataset·method·chain, sweep 수, elapsed time과 미실행 목록을 기록한다. 시간 확인은 각 complete sweep 이후에 하며, 가장 오래 걸리는 한 sweep 때문에 상한을 소폭 넘을 수 있다.

Smoke default는 n=64,p=5,D=3,R_2=R_3=5, 4 chains, burn-in 100 + retained 100이다. No-interaction, sparse 3-way·dense 3-way의 강한 signal (1,2,2) 사례를 각각 1개 사용해 3 samplers를 실행한다. 총 7,200 sweeps이며, smoke는 실행·캐시·출력 경로와 대략의 비용 점검용이다. Smoke에 본 pilot의 R-hat/ESS 기준 충족을 요구하거나 posterior 검증 완료를 선언하지 않는다.

Truth는 DGP·replication마다 새로 생성하고 그 데이터셋의 X,y는 세 sampler가 공유한다. Fitting D와 ranks는 모든 DGP에 동일하게 적용하여 구조 label로 실제 interaction 차수를 전달하지 않는다. Truth seed와 X/y seed를 분리하고, 두 객체의 seed·설정은 평가용 결과에 보존하되 fitting 입력에는 관측 X,y와 model/sampler 설정만 제공한다.

진단이 부족한 fit은 `mixing_flagged`로 기록한다. 후속 실행의 제안은 같은 데이터·prior·rank에서 4 chains, burn-in 4,000 + retained 8,000으로 독립 재실행하는 것이다. 자동 연장·무제한 재시도·checkpoint 재시작은 두지 않는다. 재실행이 필요하면 예산과 설정을 명시적으로 변경하고 첫 실행 결과도 보존한다.

### 7.5 Pilot prior 수치와 sensitivity — 변경 가능한 제안 default

노트의 prior 형태와 공유 단위는 그대로 유지하며, 아래는 pilot용 수치 제안이다. Fitting 설정에는 모든 양의 hyperparameter를 직접 지정할 수 있게 하고 resolved configuration에 실제 값을 기록한다.

| Hyperparameter | 제안 default | 간단한 근거 |
| --- | --- | --- |
| a_beta,b_beta | 3,2 | IG(3,2)의 평균 분산은 1, mode는 0.5. 절편·주효과의 공통 logit 척도에서 유한한 평균·분산을 가진 시작 prior |
| a_v,b_v — Normal·SSP | 4,1, 모든 차수에 동일 | 평균 분산은 1/3, mode는 0.2. p=5,D=3의 큰 product를 완화하면서 분산의 3차 moment까지 유한하게 함 |
| a_gamma,b_gamma — SSP | 1,1 | π_gamma의 Uniform(0,1) hyperprior를 시작점으로 사용. DGP support를 이용한 tuning을 하지 않음 |
| a_z,b_z — SSP | 1,1 | π_z의 Uniform(0,1) hyperprior, 기대 활성 component 수 R_d/2의 중립적인 시작점 |
| Horseshoe scale prior | 노트의 λ_jk^(d),τ_k^(d) 각각 C⁺(0,1) | 노트에서 고정한 half-Cauchy scale을 유지. Pilot 수치 선택으로 모델 prior를 변경하지 않음 |

Normal에서 E[(σ_v²)^2]=E[(σ_v²)^3]=1/6이므로 R_d=5일 때 각 2·3차 coefficient의 prior variance는 5/6이다. 이는 차수별 **공통 분산**을 적분한 계산으로, loading들이 주변적으로 독립이라고 가정하여 (E[σ_v²])^d로 대체하지 않는다. 이 유한 moment 설명은 D=3 pilot에 한정하며 일반 D의 moment 존재를 보장하지 않는다. SSP는 inclusion에 의해 다른 interaction prior를 가지며 Horseshoe는 heavy tail이므로, 세 prior가 같은 coefficient 규모를 가진다고 주장하지 않는다.

실행 전 prior predictive default는 모형별 독립 prior draw 500개, p=5의 32개 노출 패턴이다. β·분산·loading·indicator·scale을 **joint prior**에서 생성하고 η와 예측확률의 median·5/95% 분위수 및 예측확률 <0.01 또는 >0.99의 비율을 기록한다. Heavy tail의 평균·분산 추정만으로 규모를 맞추지 않는다. 이 점검을 통해 나타난 prior 차이를 기록하고, prior 값을 바꾸면 이후 pilot에 쓰인 값을 별도 설정으로 보존한다.

추가 sensitivity의 시작 후보는 별도 선택된 작은 DGP 집합에서 (i) 공통 fitting rank R_2=R_3∈{3,10}, (ii) Normal·SSP의 a_v=4를 고정하고 b_v∈{0.5,2}, (iii) SSP의 γ 또는 z hyperprior만 하나씩 Beta(1,3)으로 바꾸는 것이다. 전체 Cartesian product를 default로 실행하지 않고 처리량·진단과 남은 예산을 보고 후보를 선택한다. Horseshoe half-Cauchy scale을 변경하는 비교는 별도 모델 결정으로 남긴다.

## 8. 구현 환경 및 모듈 구조 제안

### 8.1 Backend 결정

**Q04 사용자 확정 (2026-10-06):** Python + NumPy/SciPy, CPU, float64를 기준 구현으로 사용한다. RNG는 NumPy Generator와 명시적 SeedSequence stream으로 통일한다. 구현 환경은 BayesianCalibration을 참고하여 Python 3.13.15·uv 0.12.16·uv_build를 사용하고, 이 프로젝트의 pyproject.toml에서 자체 uv.lock을 생성했다. 참고 프로젝트의 lockfile·JAX/BlackJAX 의존성은 복사하지 않았다.

PG adapter의 제안 default는 [polyagamma](https://github.com/zoj613/polyagamma)의 Devroye 방법을 명시적으로 선택하고 chain의 NumPy Generator를 전달하여 `PG(1,eta)`를 생성하는 것이다. h=1은 이 방법의 정수 shape 조건을 만족하며, 유한 급수 절단·saddlepoint·Normal 근사를 default로 사용하지 않는다. 설치·독립 분포 검증을 통과한 polyagamma 2.0.2와 ArviZ 0.22.0을 포함한 환경을 uv.lock에 고정했다. 로컬 실행 버전과 결과는 docs/validation.md에 기록한다. 진단의 정의와 참조 기준은 §10.2에 둔다.

### 8.2 작은 연구 패키지 구조

현재 구현 파일은 아래와 같다. 조건부분포는 conditionals.py, prior를 포함한 진단용 posterior density는 target.py로 분리했다. 루트에 pyproject.toml·uv.lock·.python-version·.gitignore·.gitattributes·.editorconfig·README·AGENTS를 두고 .github/workflows/tests.yml에 locked 설치·lint·test workflow를 구성했다. 개발 방법과 작업 규칙은 README·AGENTS에 기록한다. 로컬 Git main 저장소는 초기화했으며 remote 설정·commit/push는 수행하지 않았다.

| 위치 | 책임 |
| --- | --- |
| src/factorregression/model.py | FM predictor, theta reconstruction, 참조 likelihood 및 prior |
| state.py | validated fixed specification과 모델별 state |
| conditionals.py | 공통 Gaussian/IG 조건부분포 매개변수 |
| samplers/horseshoe.py | local/component slice와 HS sweep |
| samplers/ssp_reference.py | 모든 z,gamma,tilde_v를 유지하는 full conditional과 Reference sweep |
| samplers/ssp_scalable.py — 보류 | marginal target, component birth/death 및 support add/delete/swap MH, active slab Gibbs와 Scalable sweep |
| samplers/normal.py | 승인된 Normal loading prior 및 hyperparameter 갱신과 sweep |
| distributions.py | PG adapter 및 필요한 truncated distribution sampling |
| mcmc.py | 초기화, 명시적 sweep loop, burn-in 및 retained sampling |
| simulation.py — 후속 | 13개 unique DGP, 생성·평가용 truth와 관측 fitting input 분리. 현재 sampler 검증에 필요한 작은 fixture는 별도 검증 코드에서 제공 |
| inference.py — 보류 | coefficient/PIP/interval/prediction summaries. 현재 Sampling 구현 대상에서 유보 |
| diagnostics.py | 체인 수렴, MCSE, immobility 및 실패 보고 |
| cli.py | 최소 실행·분석 진입점 |
| configs/, tests/ | 합의된 설정과 단계별 참조 검증 |

run 설정은 입력·출력, method, seed, chain 수, iteration 수를 포함한다. scientific 설정은 D/ranks/prior/DGP를 담는다. sampler 설정은 scan·초기화·수치 알고리즘을 담는다. JSON 및 명시적 CLI override를 구현했다. 효력이 없는 옵션이나 알 수 없는 옵션은 거부한다.

현재 SSP 구현은 `ssp_reference`로 둔다. 향후 Scalable을 재개하면 같은 scientific specification에 대해 `ssp_scalable`을 구분하고, move mixture·proposal 분포·시도 횟수를 sampler 설정으로 기록한다. 이때 correctness 참조가 Scalable의 MH 코드를 재사용하여 동일 오류를 반복하지 않도록 독립적인 dense target/계산을 남긴다. 두 구현을 위해 범용 trans-dimensional sampler framework를 만들지는 않는다.

## 9. 상태, 캐시, 수치 및 저장 계약

| 구분 | 내용 |
| --- | --- |
| 공통 고정 데이터 | X,y,X_tilde,kappa, 차수·rank, prior hyperparameters |
| 공통 표본 상태 | beta,sigma_beta2,omega |
| HS 표본 상태 | V, local/component scale의 한 가지 기준 표현 |
| Reference SSP 표본 상태 | 모든 z[R_d],gamma[p,R_d],tilde_v[p,R_d],pi_z,pi_gamma 및 차수별 sigma_v2[d] |
| Scalable SSP 표본 상태 — 보류 | active component label 집합, 각 성분의 support와 active slab 값,pi_z,pi_gamma,sigma_v2[d]; inactive gamma/slab 배열 없음 |
| Normal 표본 상태 | V, 차수별 sigma_v2[d]; 별도 mean state 없음 |
| 파생값 | effective V, interaction contributions, eta, polynomial caches |
| 실행·진단 | method/chain RNG, sweep 번호, burn-in 구분, timing, failure context |

SSP effective V를 잠재변수와 독립적으로 수정하지 않는다. `eta`를 캐시한다면 모델 상태에서 재구성한 값과 주기적으로 비교한다. beta 변경은 main predictor, loading/gamma/z 변경은 해당 성분과 total predictor를 즉시 무효화·갱신한다. scale 변경은 다음 조건부분포의 prior precision에 반영한다. omega 변경은 모든 omega-weighted statistic을 무효화한다. omega 갱신 이후의 eta 변경만으로 같은 sweep 안에서 omega를 다시 뽑지는 않는다.

현재 Reference SSP는 모든 slab에 조건부로 분산을 갱신한다. 아래의 Scalable 계약은 보류된 후속 설계이다: active slab만으로 분산을 갱신하며 inactive state를 복원하지 않는다. Birth/add 후보 slab은 현재 sigma_v2[d] 및 명시된 proposal로 생성한다. State·캐시 변경은 MH 수락 시에만 반영하고 거절된 후보를 posterior state에 남기지 않는다. Sparse predictor가 dense p×R_d 배열을 매번 생성하지 않도록 하고, 작은 검증에서만 dense reference를 사용한다.

float64, 안정적인 sigmoid/logaddexp/log1p/expm1과 Cholesky solve를 사용한다. 계수 clipping, scale floor, 추가 ridge, 임의 jitter로 실패를 숨기지 않는다. 분산·분포 파라미터가 유효하지 않으면 chain/sweep/update/관련 요약값으로 원인을 보고한다.

독립 seed stream을 data truth, exposure/response, model, chain으로 분리하고 NumPy Generator를 명시적으로 전달한다. Stream key는 고정된 DGP 번호·replication 번호·용도·method 번호·chain 번호로 구성하여 실행 순서나 method 추가가 기존 stream을 바꾸지 않게 한다. Python의 process별 hash나 전역 RNG를 숨겨 사용하지 않는다. 동일 환경·설정 내 재현과 다른 환경에서 통계적으로 비교 가능한 재현을 구분한다.

저장 결과는 resolved configuration, seeds, 필요한 라이브러리 버전·backend 정보, timings, retained draws, diagnostics이다. 버전 정보는 실행 설명용이며 환경 hash, 호환성 gate, 과거 환경 복원 시스템을 만들지 않는다. Checkpoint 직렬화·재시작·job scheduling도 구현하지 않는다. 저장 범위의 제안 default는 §9.2에 명시하며 사용자가 후속 설정으로 변경할 수 있다.

### 9.1 MCMC 초기화와 scan — 변경 가능한 제안 default

초기화는 관측 X,y와 지정된 설정·chain RNG만 사용한다. 검증용 고정-state fixture를 제외한 fitting에서는 DGP label·true support·true coefficient를 참조하지 않는다. 초기값은 sampler state일 뿐 prior나 본 실험 설정을 바꾸지 않는다.

| 대상 | 제안 default | 간단한 근거 |
| --- | --- | --- |
| Chain별 loading·β jitter 크기 | c=(0.5,1,2,4), s_init=0.1c | 작은 비영 값부터 서로 다른 크기로 시작해 초기값 의존성을 살펴보고 큰 product의 시작 overflow를 줄임. s_init은 계산용 초기 SD이며 prior 변수와 구별 |
| β_0 | log((Σy+0.5)/(n−Σy+0.5)) + N(0,s_init²) | 관측 반응의 smoothed log odds로 극단적인 초기 logit을 줄임. 모든 y가 0/1이어도 유한 |
| β_j, j≥1 | 독립 N(0,s_init²) | 주효과에 truth를 주입하지 않고 chain별 작은 차이를 제공 |
| σ_beta² | b_beta/(a_beta+1) | IG prior의 유한한 mode로 시작. Pilot 제안값에서는 0.5 |
| Normal V 및 SSP의 모든 tilde_v | 독립 N(0,s_init²) | 연속 loading을 모두 0으로 시작하는 고차 좌표의 초기 퇴화를 피하고 inactive slab도 상태에 유지 |
| Normal·SSP σ_v^{2(d)} | b_v/(a_v+1) | 차수별 prior mode로 시작. Pilot 제안값에서는 0.2 |
| Horseshoe V | 독립 N(0,s_init²) | 비영 loading과 서로 다른 초기 효과 규모를 제공 |
| Horseshoe λ_jk^(d),τ_k^(d) | 모두 1 | 노트의 C⁺(0,1) median에서 시작하여 유한한 초기 precision을 확보 |
| SSP π_gamma^(d),π_z^(d) | 각각 a/(a+b) | 지정한 Beta hyperprior의 평균. Pilot 제안값에서는 모두 0.5 |
| SSP z — 4 chains | 차례로 all 0; Bern(0.5); all 1; all 1 | 비활성·중간·활성 상태를 함께 시작해 inclusion 탐색을 점검 |
| SSP γ — 4 chains | 첫 세 chain은 Bern(0.5), 네 번째는 all 1 | 다양한 support에서 시작. z=0인 component도 γ를 보유하며 0으로 강제하지 않음 |
| η·ω | 초기 state에서 η를 재구성하고 첫 sweep에서 ω~PG(1,η) | 오래된 predictor나 임의의 고정 ω로 β 조건부분포를 시작하지 않음 |

4개를 넘는 chain은 위 4개 시작 유형을 순환하되 독립 RNG draw를 사용한다. 사용자가 initial state를 제공하면 shape·값·양수 scale을 검사하고 제공된 값을 기록한다. 유효하지 않은 초기값을 clipping/floor로 수정하지 않는다.

Scan은 노트의 sweep 순서와 각 단계 내부의 오름차순 d→k→j로 고정한다. Reference SSP는 전체 z 단계→전체 γ 단계→전체 slab 단계→차수별 π_gamma,π_z,σ_v² 갱신으로 둔다. PG는 sweep의 첫 단계에서 한 번 갱신하고 이후 고정한다. Burn-in 동안 별도의 prior tuning이나 adaptive kernel 변경은 하지 않는다.

### 9.2 표본 저장과 출력 — 변경 가능한 제안 default

| 항목 | 제안 default | 간단한 근거 |
| --- | --- | --- |
| 보존 시점 | burn-in을 제외한 모든 complete sweep, thinning=1 | ESS·MCSE와 사후 재계산에 쓸 정보를 보존. 저장된 draw 번호에 원래 sweep 번호를 함께 기록 |
| 공통 draw | β,σ_beta² | 모든 모형의 공통 상태·검증에 필요 |
| Normal draw | 모든 V^(d),σ_v^{2(d)} | interaction 재구성과 분산 조건부분포 점검에 필요 |
| Horseshoe draw | 모든 V^(d),λ_jk^(d),τ_k^(d) | scale·loading 상태를 재검토하고 다른 요약을 계산할 수 있게 보존 |
| Reference SSP draw | 모든 z,γ,tilde_v,π_z,π_gamma,σ_v² | inactive indicator/slab까지 포함한 expanded state를 보존 |
| ω·polynomial cache·관측별 η | 기본 저장 제외 | 보통 큰 파생/보조 배열이며 retained parameter state에서 필요한 predictor를 재계산할 수 있음. 요청 시 ω 저장 옵션 제공 |
| p=5,D=3 검증용 파생 draw | 20개 θ_α^(d), 32개 노출 패턴의 예측확률, marginal log likelihood·log posterior | Label·부호 비식별성에 덜 민감한 sampler 진단용. Inference의 PIP·선택·효과 보고 구현과 구별 |
| 일반 p,D 파생 draw | 요청한 interaction tuple·노출 패턴만 | 조합 전체를 자동 열거하지 않아 기본 출력의 폭증을 피함. 요청이 없으면 관측 X의 첫 min(16,n)행에 대한 예측확률을 진단용으로 보존 |
| 파일 | chain별 compressed NPZ, configuration·metadata·diagnostics JSON | NumPy로 직접 읽을 수 있는 간단한 연구 결과 형식. Python 객체 pickle이나 재시작용 state 직렬화는 사용하지 않음 |
| Draw 저장 예산 | fit당 추정 비압축 draw 크기 512 MiB | Pilot 크기에서 full latent draw 보존이 작으며, 일반 차원에서 큰 저장 요청은 실행 전에 크기를 알림. 초과 시 명시적으로 저장 범위·draw 수·한도를 변경 |

Draw 크기 추정은 float64·indicator dtype·chain/draw 수·요청한 파생값을 반영한다. 압축률을 가정하여 예산을 줄이지 않는다. 이 예산은 retained draw 저장의 한도이며 전체 sampler의 메모리·계산 가능성을 보장하는 값이 아니다. 일반 차원에서도 raw state의 기본 저장은 같지만, 실험별로 사용자가 저장 범위를 변경할 수 있다.

Marginal log posterior는 ω를 적분한 Bernoulli likelihood와 **현재 모형의 전체 prior/hyperprior**를 사용한다. Reference SSP에서는 inactive slab·γ도 포함한다. 시간·상태 이동 기록은 diagnostics로 저장하며, 임의의 효과 threshold나 PIP 계산은 보류한다.

### 9.3 수치·시간 실패와 캐시 점검 — 변경 가능한 제안 default

매 update에서 분포 매개변수·draw·predictor가 유효한지 검사한다. Nonfinite 값, 비양수 분산/scale/ω, Cholesky 실패, 캐시 불일치는 해당 chain의 `numerical_failure`로 기록하고 중단한다. 예산 상한 도달은 `budget_exhausted`로 구별한다. 성공한 complete sweep까지만 보존하며 완료되지 않은 체인을 완료 draw 수로 채우거나 불완전 결과로 진단 통과를 선언하지 않는다.

η 캐시는 매 100 sweeps마다 현재 state에서 직접 재계산한 predictor와 비교한다. 제안 기준은 atol=1e−10, rtol=1e−8이다. 통과한 뒤 재계산값으로 캐시를 갱신해 누적 rounding을 줄이고, 실패하면 오차·chain·sweep·update를 기록한다. 작은 단위 검증에서는 각 좌표 갱신 직후에 더 엄격한 §10.1 기준으로 확인한다. 입력이나 prior를 clipping/floor/ridge/jitter로 바꾸어 실패를 숨기지 않는다.

완료 상태와 진단 상태는 구별한다. 설정한 sweep 수를 채웠어도 §10.2 기준이 부족하면 `mixing_flagged`로 보고한다. `diagnostics_ok`는 지정한 점검에서 문제를 발견하지 않았다는 뜻이며 posterior 탐색의 증명이 아니다.

## 10. 검증 전략과 단계별 완료 조건

아래는 단계별 완료 기준과 현재 상태이다. P1–P6의 기준 구현 및 아래에 한정한 참조 검증은 통과했다. 상세 테스트·환경·한계는 docs/validation.md에 기록하며 full pilot·본 실험 결과를 뜻하지 않는다. 수치 참조를 별도로 만들고 implementation을 그대로 복제하는 테스트를 피한다. 확률적 검증 허용오차는 Monte Carlo uncertainty를 기준으로 사전 설정한다.

| 단계 | 산출물 | 완료 조건 |
| --- | --- | --- |
| P0 — 개발 준비 완료 | Sampling 계획과 모델 명세 | 공통·Normal·Horseshoe·Reference SSP의 명세, Python/NumPy/SciPy·CPU·float64, 일반 p,D,R_d 지원 및 초기화·저장·검증 default 명시. 현재 범위의 착수를 막는 추가 사용자 결정 없음. 패키지 설치·버전 고정과 수학·수치 검증은 구현 단계에서 수행 |
| P1 — 기준 구현·검증 완료 | 데이터·notation·FM 참조 계산 | p=5,D=3의 explicit combinations와 predictor/theta 일치; 단일 loading 선형 분해, 차수별 서로 다른 ranks, D=1/D=p의 작은 경계 fixture, zero exposure, inactive component 검증 |
| P2 — 기준 구현·검증 완료 | 공통 PG 및 Gaussian/IG blocks | PG mean `tanh(c/2)/(2c)` 및 c=0의 1/4, variance/CDF 참조, 양수 support·부호대칭·tail 검증; Gaussian/IG joint-ratio 및 sampling moments |
| P2a — 기준 구현·검증 완료 | Normal baseline Gibbs | 평균 0 loading/분산 조건부분포의 joint-density ratio 일치; 차수별 p R_d count, IG 매개변수화, mean hyperprior 항 부재 및 sampling moments 검증 |
| P3 — 기준 구현·검증 완료 | HS Gibbs | local/component density와 truncated CDF, 0 rate 경계, 최신 캐시 의존성 및 complete-sweep 검증 |
| P4a — 기준 구현·검증 완료 | Reference SSP full Gibbs | full-joint ratio로 z/gamma 조건부분포 검증; z=0에서 gamma Bernoulli prior, inactive slab Gaussian prior; 전체 gamma/slab을 반영한 Beta/IG 조건부분포와 Gaussian moments 확인 |
| P4b — 보류 | Scalable SSP marginal target 및 MH-within-Gibbs | inactive 합산/적분 identity; birth/death·add/delete/swap의 target/proposal ratio, detailed balance·경계·reverse reachability; active slab Gibbs, pK_d와 M_d의 hyperparameter count, K_d/M_d=0의 prior draw, inactive latent state 미저장·미갱신 검증 |
| P4c — 보류 | 두 SSP sampler posterior 일치 | 작은 p,R_d·공통 데이터/prior/rank에서 theta,eta,predictive probability의 posterior 요약을 MCSE·ESS와 함께 비교; 다양한 초기값 및 독립 작은 참조 문제로 확인 |
| P5 — 작은 참조 범위 검증 완료 | posterior end-to-end 검증 | main-only 작은 logistic posterior를 quadrature와 비교; 작은 interaction 모델의 독립 계산/참조 MCMC와 식별 가능한 functional 비교; posterior invariance와 mixing을 구분 |
| P6 — 기준 구현·검증 완료 | Sampling loop·저장·diagnostics | 명시적 seed와 설정, 초기화·burn-in·retained draw 구분, 표본 저장, rank-normalized R-hat·bulk/tail ESS·MCSE, binary-state 고정 및 chain 불일치 표시. theta·eta 등 검증용 계산 유지 |
| P6a — 보류 | 추론 및 효과 요약 | 구조 PIP, sign mass, quantile/conditional summary 및 선택 규칙은 후속 논의·구현 |
| P7 — 후속 | 13개 DGP 및 소규모 pilot | 중복 제거·support·norm·빈 support·독립 seed 확인; 모든 fitting 경로에 true support가 전달되지 않음을 확인; 명시된 잠정 설정으로 13개 DGP의 pilot을 수행하고 correctness·혼합·실패·시간/메모리·prior sensitivity 평가 |
| P8 — 후속 | pilot 이후 설정 확정 및 본 실험 | pilot 근거로 n·replication·R_2,R_3·prior hyperparameters·MCMC budget을 결정하고 실행 전 기록; 선택 threshold 등 평가 규칙 확정; 13개 DGP에서 유효 결과와 실패·반복 오차 보고; seed·설정으로 주요 표·그림 재생성 |
| P9 — 필요시 | 프로파일 기반 최적화 | P1 참조 및 posterior 검증 보존; 전체 sweep 시간을 비교하고 병목 개선의 근거 제시 |

Scalable을 재개한 뒤 P4c 이후 두 축(R_d,p) 성능 실험을 설계한다. Active 수·support 크기를 기록하며 메모리, proposal 비용, complete-sweep 시간, 수락률, invariant ESS/초를 평가한다. 이 성능 검증을 P4c의 posterior correctness 대신 사용하지 않는다. 현재 Sampling 개발에서는 이 비교·성능 실험을 요구하지 않는다.

P5에서 quadrature가 가능한 차원으로 문제를 제한하며 고차원 posterior에 무리한 quadrature를 요구하지 않는다. 필요시 모델의 joint prior에서 자료를 생성하는 simulation-based calibration을 별도로 설계한다. 첨부 fixed-signal DGP의 coverage 실험은 prior 기반 SBC가 아니다. HS/SSP/Normal이라는 다른 prior 사이의 posterior 일치로 correctness를 대신 검증하지 않는다. 같은 SSP posterior의 Reference/Scalable 일치는 Scalable 재개 시 P4c에서 수행한다.

현재 제안 tolerance와 진단 기준은 §10.1–10.2에 명시한다. 여러 chain 및 R-hat만으로 posterior 탐색이 충분하다고 단정하지 않는다. SSP에서는 inclusion 변화 횟수·활성 성분 수와 invariant theta/predictive trace를 함께 점검한다. 극단적 heavy tail에서는 scale의 평균 추정 안정성을 별도로 확인한다.

### 10.1 수학·수치 검증 tolerance — 변경 가능한 제안 default

아래는 float64의 작은 기준 문제를 위한 초기 기준이다. 첫 주요 fixture는 p=5,D=3, 차수별 서로 다른 ranks를 포함한다. 일반 loop의 경계는 D=1, D=p 및 zero exposure·불활성 state 같은 작은 추가 fixture로 점검한다. 결과를 보고 tolerance를 넓혀 통과시키지 않으며, 조건수가 큰 문제나 극단적인 scale에는 독립 reference의 오차·문제의 크기에 맞는 기준을 따로 기록한다.

| 검증 종류 | 제안 default | 근거·판정 방식 |
| --- | --- | --- |
| Deterministic predictor·θ·h/g·좌표 직후 η | atol=1e−12, rtol=1e−10 | 명시적 interaction 조합 reference와 DP/캐시 계산의 작은 float64 rounding 차이를 허용. abs error ≤ atol + rtol×abs(reference)로 판정 |
| Gaussian mean·variance 및 joint-density difference | atol=1e−12, rtol=1e−10 | Gaussian/IG·indicator 식을 독립 joint density의 차이와 비교. Density 자체보다 log-density difference를 사용 |
| Cholesky·linear solve | 상대 backward residual ≤1e−10 | \(\lVert A m-b\rVert/(\lVert A\rVert\lVert m\rVert+\lVert b\rVert)\)를 검사. 0 분모인 exact-zero fixture는 절대오차로 검사 |
| 주기적 η 캐시 검사 | 매 100 sweeps, atol=1e−10, rtol=1e−8 | 반복적인 delta update의 누적 rounding을 고려. 좌표별 단위 검증보다 완화하되 실패를 숨기지 않음 |
| 독립 적분 reference | quadrature epsabs=1e−10, epsrel=1e−8 | 작은 main-only·interaction fixture의 기준량 계산. 더 엄격한 설정/다른 적분 범위에서 reference 안정성을 확인하고 잔여 수치 오차를 비교 tolerance에 포함 |
| IID conditional sampling moment | case당 N=50,000, 평균 차이 ≤5 MCSE | 고정한 조건부 분포에서 반복 생성한 **독립 draw**만 대상. 알려진 variance로 MCSE를 계산하고, variance moment 검증은 충분한 moment가 존재하는 fixture를 사용 |
| IID CDF·PIT 검증 | case당 N=50,000, 사전 정의한 검정 family의 총 α=0.001 | Continuous CDF의 PIT를 Uniform과 비교하고 Bernoulli 등 discrete 분포는 적절한 exact probability 검정을 사용. m개 검정이면 α/m의 Bonferroni 기준 적용 |
| MCMC와 작은 독립 posterior reference | 유한 평균·CDF probability 차이 ≤5 MCSE + reference error | MCMC 자기상관을 반영한 MCSE를 사용. 비교량별 값을 사전 명시하고 서로 다른 prior의 posterior를 같다고 요구하지 않음 |

5 MCSE는 무조건적인 정확성 보장이 아닌 보수적인 stochastic discrepancy flag이다. MCSE가 너무 커서 차이를 판별할 수 없는 상태를 correctness 통과로 처리하지 않고 §10.2의 정밀도 조건을 함께 확인한다. 확률적 검증에 실패하면 reference·seed·실제 discrepancy를 보존하고 원인을 조사하며 통과할 때까지 seed를 바꾸는 방식을 사용하지 않는다.

PG fixture는 η∈{0,±1,±5,±20,±100}의 `PG(1,η)`에 대해 양수·finite support, 부호대칭, 알려진 mean/variance 및 별도의 density/CDF 참조를 검사한다. η=0의 mean=1/4과 variance=1/24를 포함한다. Sampling library 자체의 CDF와 일치하는 것만으로 PG 검증을 완료하지 않는다. [Polson, Scott, Windle의 PG 논문](https://arxiv.org/abs/1205.0310)의 분석식을 기준으로 독립 계산을 남긴다.

Horseshoe는 u를 고정한 truncated conditional sampler를 IID 검증 대상으로 삼는다. Rate=0의 Uniform 또는 bounded power 분포, 양수 rate의 작은/큰 rate×bound, Gamma의 작은 truncation mass를 포함한다. Slice sweep에서 연속으로 갱신된 scale draw는 IID라고 가정하지 않는다. SSP는 z=0의 γ prior draw, inactive slab의 prior draw와 **전체** indicator/slab count를 독립적으로 확인한다.

End-to-end 기준 문제의 기본안은 관측 X=0의 작은 main-only logistic fixture에서 σ_beta²와 정보 없는 β_j를 적분한 intercept posterior의 1차원 quadrature 비교, 그리고 β·σ_v²를 고정한 작은 Normal 2-way loading block의 2차원 quadrature 비교이다. 후자는 nuisance 고정 상태의 conditional kernel 검증임을 명시한다. 이를 p=5,D=3의 full-sweep joint-ratio·캐시·다중-chain 검증과 함께 사용하고, 고차원 posterior 전체를 quadrature로 계산한다고 주장하지 않는다.

### 10.2 MCMC diagnostics와 판정 — 변경 가능한 제안 default

| 항목 | 제안 default | 간단한 근거 |
| --- | --- | --- |
| R-hat | rank-normalized split 및 folded-split R-hat의 최대값 <1.01 | 서로 다른 위치·scale·heavy tail에 더 민감한 다중-chain 점검 |
| ESS | bulk ESS≥400, tail ESS≥400, 4 chains 전체 기준 | Pilot의 최소 정보량 점검. Tail ESS는 5/95% quantile의 작은 쪽을 사용하며 chain당 400으로 해석하지 않음 |
| β·θ 평균 MCSE | posterior SD 대비 ≤0.05 | 유한하고 안정적인 mean/SD가 확인되는 비교량의 상대 정밀도 기준. SD=0 또는 heavy tail로 불안정하면 NA와 이유 표시 |
| 검증용 예측확률 평균 MCSE | 절대값 ≤0.01 | 유계 probability의 1 percentage point 수준 pilot 정밀도. 본 실험의 과학적 효과 threshold가 아님 |
| Scale·tail 정밀도 | log-scale trace·rank plot, median·5/95% quantile MCSE 점검 | Horseshoe의 raw scale 평균·분산이 안정적이라고 가정하지 않음. Quantile MCSE의 제안 기준은 5–95% 폭의 5% 이하이며 폭=0/추정 불가이면 NA |
| SSP 이동 기록 | z·γ 변화 횟수, K_d=Σz, T_d=Σγ, M_d=Σzγ의 trace | inclusion 탐색과 label-invariant 크기를 함께 관찰. 각 chain/차수의 K_d,M_d 변화가 0이면 경고, 1–19회면 낮은 이동성 주석 |
| Constant·불완전 chain | 진단 NA 및 원인 표시 | Constant state의 R-hat/ESS를 임의의 좋은 값으로 채우지 않음. 서로 다른 상수의 chain 또는 불완전 fit은 통과로 표시하지 않음 |

R-hat·ESS의 초기 기준과 rank/folding 정의는 [Vehtari et al.의 MCMC 진단 논문](https://arxiv.org/html/1903.08008v5)에 따른다. MCSE 정밀도·SSP 이동 주석·시간/저장 상한은 이 프로젝트의 변경 가능한 제안이다. Thinning=1은 동일하게 생성한 draw를 모두 보존하는 정책이며, 메모리가 허용되면 임의 thinning으로 정보를 버리지 않는다는 [Stan의 설명](https://mc-stan.org/docs/reference-manual/analysis.html#thinning)을 참고한다.

주요 진단 대상은 모든 β, σ_beta², Normal/SSP의 차수별 σ_v², SSP π_z·π_gamma·K_d·T_d·M_d, marginal log likelihood·log posterior, 검증용 θ와 예측확률이다. Loading·slab·local/component scale 및 개별 indicator도 보조 진단을 기록한다. Factor label permutation·짝수 차수의 성분 부호 비식별성 때문에 loading별 요약의 불일치만으로 coefficient posterior가 다르다고 결론내리지 않으며, 해당 경고를 지우거나 전체 joint posterior가 탐색되었다고 선언하지도 않는다.

`diagnostics_ok`의 기본 조건은 유효하게 완료된 4개 chain, 정의 가능한 주요 진단의 R-hat·bulk/tail ESS·MCSE 기준 충족, 수치 오류 없음이다. 주요 진단의 실패나 설명되지 않은 constant·이동 정체는 `mixing_flagged`로 보고하고 원인을 점검한다. SSP의 높은 posterior certainty와 실제 trapping은 추가 draw·서로 다른 초기값·작은 reference 문제로 구분한다. Smoke run에는 이 기준을 적용하지 않는다. 진단을 통과해도 다른 미탐색 모드의 부재를 보장하지 않으며 P1–P5의 수학 검증을 대신하지 않는다.

## 11. 결정 기록 및 다음 논의

| ID | 쟁점 | 현재 상태 / 제안 |
| --- | --- | --- |
| U01 | 개발 전 계획 준비 | 개발 전 명세 준비 완료 뒤 사용자 요청에 따라 현재 범위 구현·검증 수행 |
| U02 | 연구 코드 목표·우선순위 | 사용자 확정: 방법론을 명확하고 검증 가능하게 구현; 통계적 정확성과 이해 가능성 우선; 작은 참조 문제 검증 후 필요한 최적화만 수행 |
| U03 | 재현성·실행 인프라 범위 | 사용자 확정: seed·명시된 설정으로 주요 결과 재실행; checkpoint, job orchestration, 환경 hash 등 일반 목적 실행 인프라 제외 |
| Q01 | 비교모형 범위 | 사용자 확정: HS + SSP + Normal baseline의 세 모형. 현재 SSP 개발은 Reference만; Scalable은 보류 |
| Q02 | SSP prior/transition 불일치 | 사용자 확정: 원래 독립 prior 유지, z=0⇒gamma=0 제약 없음; 현재 Reference full Gibbs 개발. 같은 posterior의 Scalable 설계는 보류 |
| Q03 | Inference.md의 적용 | 사용자 요청으로 보류 (2026-10-06). Sampling 개발의 선행 조건에서 제외 |
| Q04 | 언어/backend | 사용자 확정 (2026-10-06): Python + NumPy/SciPy, CPU, float64 |
| Q05 | 일반 p,D 지원 및 fitting ranks | 사용자 확정: 일반 p,D≤p, 차수별 R_d 지원; 계산 가능성의 범위 보장 없음. 첫 simulation/validation p=5,D=3. Pilot 제안 R_2=R_3=5; 본 실험 rank는 pilot 이후 결정 |
| Q06 | prior 수치 | 본 실험 값은 pilot 이후 결정. 사용자 요청에 따른 pilot 제안값: IG_beta(3,2), IG_v(4,1), SSP Beta_gamma/Beta_z(1,1); §7.5의 근거와 변경 절차 참조 |
| Q07 | 실험 규모 | 본 실험 n·replication·MCMC budget은 pilot 이후 결정. Pilot 제안은 n=200,13 DGP×1 replication,4 chains×(2,000 burn-in+2,000 retained); truth 반복 재생성, §7.4의 시간 상한 적용 |
| Q08 | 추론 선택 규칙 | Inference와 함께 보류: 실질적 효과 epsilon, PIP threshold, conditional interval 필요 여부 |
| Q09 | 빈 support 및 중복 DGP | 사용자 확정: no-interaction 1개 + 4개 구조 × 3개 signal = 13개 unique DGP. 빈 support의 계수·실현 norm은 0 |
| Q10 | 입력·출력 및 추가 비교모형 | 현재 draw 저장 제안은 §9.2에 명시. 실제 데이터·추가 baseline·주요 논문 표/그림은 후속 범위. Archive 재사용을 전제로 하지 않음 |
| Q11 | Normal baseline의 prior 적용 범위 | 사용자 확정: 공통 likelihood·절편/주효과 prior 아래 loading에 계층적 Normal 적용 |
| Q12 | Normal mean 및 variance 공유 단위 | 사용자 확정: mean=0, 차수별 모든 j,k가 sigma_v2[d] 공유; variance ~ IG(a_v,b_v). 노트 표기로 통일 |
| Q13 | SSP hyperparameter 갱신 | 두 표현에 맞춰 §4.4 유도: Reference는 전체 gamma/slab 조건부; Scalable은 active component의 gamma count와 active slab 사용. Scalable inactive 복원 없음 |
| Q14 | SSP 두 구현 및 검증 | 기존 두 구현 설계 보존. 현재 Reference expanded-state full Gibbs만 개발; Scalable 및 두 sampler의 posterior 비교는 보류 (2026-10-06) |
| Q15 | Scalable MH 세부 설계 | 사용자 요청으로 보류 (2026-10-06): support/slab proposal, 이동 혼합비, 선택 확률, 시도 횟수, 초기화·수락비. Sampling 개발의 선행 조건에서 제외 |
| Q16 | Scalable 성능 목표 | 기존 component-level·high-dimensional p scalability 목표는 장기 기록으로 유지; 현재 설계·검증은 보류 |
| U04 | True support 비공개 | 사용자 확정: true interaction support는 어떤 fitting model에도 제공하지 않음. 생성·평가에만 사용하며 pilot도 동일 |
| U05 | 본 실험 수치의 결정 시점 | 사용자 확정: n·replication·R_2,R_3·prior hyperparameters·MCMC budget은 pilot 이후 결정. 계획 완료의 선행 조건으로 요구하지 않음 |
| U06 | 참고 노트 및 notation | 사용자 확정 (2026-10-06): 지정한 여덟 노트 참고, notation은 노트 기준. SSP gamma·pi_gamma, 차수별 sigma_v2 및 IG 표기로 통일 |
| U07 | 현재 개발 범위 | 사용자 확정 (2026-10-06): 공통·Normal·Horseshoe·Reference SSP Sampling 구현·검증을 주 목표로 함. Inference·Scalable SSP 보류. 현재 단계는 P1·P2·P2a·P3·P4a·P5·P6 |
| U08 | Default 제안 방식 | 사용자 요청 (2026-10-06): 초기화·저장·검증 tolerance/diagnostic·pilot 잠정 수치는 합리적인 default와 근거를 계획에 기록하고 사용자가 변경 가능하게 함. 본 실험 scientific setting으로 간주하지 않음 |
| Q17 | MCMC 초기화·scan | §9.1에 변경 가능한 제안 default 명시: 관측 데이터 기반 β, 작은 비영 loading, IG mode·half-Cauchy median, SSP의 서로 다른 시작 support, 노트의 고정 scan |
| Q18 | Draw 저장·실패 정책 | §9.2–9.3에 제안 default 명시: thinning=1, full latent state·NPZ/JSON, ω 기본 제외, 512 MiB draw 예산, 캐시 검사·실패 상태 구분 |
| Q19 | 검증 tolerance·diagnostics | §10.1–10.2에 제안 default 명시: deterministic atol/rtol, IID N=50,000·MCSE/CDF 기준, R-hat<1.01·bulk/tail ESS≥400 및 정밀도·이동 기록 |

현재 Sampling 범위의 P1–P6 기준 구현을 제공했다. 전체 68개 테스트와 lint/format, locked offline 설치, sdist/wheel 빌드 및 9개 smoke fit(7,200 sweeps)이 통과했다. Smoke의 혼합 진단은 모두 mixing_flagged이며 수렴 또는 scientific accuracy를 주장하지 않는다. Q03/Q08의 Inference와 Q15/Q16의 Scalable SSP는 계속 보류한다. 다음 연구 단계는 P7의 full 13-DGP pilot과 충분한 표본에서의 혼합·비용·prior/rank sensitivity 조사이다. 본 실험 scientific setting은 pilot 이후 결정한다. GitHub 원격 연결·공개·push는 아직 수행하지 않았다.

## 12. 검토 기록

아래는 시간순 기록이며 현재 사양은 본문의 최신 결정과 §11을 따른다. 이전 기록의 미결 상태·제안은 후속 결정으로 대체될 수 있다.

- 2026-10-01: 외부 원문 네 파일을 읽고 이전 프로젝트의 지침·계획 구조 및 최신 public-scope 방침을 참고했다.
- 통계 검토에서 SSP 비활성 상태와 prior의 불일치, collapsed gamma/slab 갱신 순서, 변분추론 문서와 FM 사양의 차이를 확인했다. 수정안은 아직 사용자 승인 전이다.
- 시뮬레이션에서 빈 support의 정규화와 중복 DGP, 미정 sample size/rank/prior 설정을 기록했다.
- 문서만 작성했다. 패키지 설치, 코드·테스트 작성, sampler 또는 시뮬레이션 실행은 하지 않았다.
- 2026-10-01 추가 논의: 사용자가 Freudenthaler Normal baseline 추가 및 간결한 연구 코드·제한된 재현성 범위를 확정했다. 첨부 문헌 노트와 원문을 확인하고 Normal prior/PG 확장 제안, 모형별 상태·검증·평가 및 명시적 인프라 제외를 반영했다. SSP 수정안과 Inference.md 해석은 미확정으로 유지했다. 이번 수정도 계획 문서에 한정했다.
- 같은 날 사용자 답변: Normal baseline은 공통 likelihood·주효과 prior와 loading의 계층적 normal prior 조합으로 확정했다. 이에 따라 세 모형의 beta 갱신을 공통 명세로 정리했다. Pooling 단위와 prior 수치는 미확정이다.
- 같은 날 후속 결정: Normal은 mean=0, 차수별 공통 variance 및 Gamma(a_v,b_v) precision prior로 확정했다. 앞선 성분별 mean/precision 제안을 대체하고 관련 상태·조건부분포·검증 계획을 수정했다. SSP의 같은 prior 및 active-only 갱신 제안에는 inactive slab을 적분한 Gamma conditional과 새 variance에서의 복원 block을 유도했다. SSP inclusion 사양과 hyperparameter 값은 여전히 미결이며 코드는 작성하지 않았다.
- 같은 날 SSP 구현 분리 확정: 사용자가 Reference full Gibbs/expanded state와 Scalable inactive-collapsed MH-within-Gibbs를 지정했다. §4를 두 표현의 공통 prior, 조건부/주변 target, 상태, 이동, hyperparameter 갱신 및 검증으로 재작성했다. Reference의 collapsed gamma/active-only precision 제안과 Scalable의 inactive 복원을 제거했다. 모듈·상태·실험 비교·P4a–P4c 완료 조건 및 결정 기록도 일치시켰다. Component와 p의 두 scalability 목표를 기록했고, 구체적 MH proposal 설정은 미결로 남겼다. 문서만 수정했으며 코드·테스트는 작성하거나 실행하지 않았다.
- 같은 날 시뮬레이션 결정: no-interaction 중복을 제거한 13개 unique DGP와 모든 fitting model에 true interaction support를 제공하지 않는 원칙을 확정했다. 본 실험 n·replication·R_2,R_3·prior hyperparameters·MCMC budget은 pilot 이후 결정하도록 §7, P0/P7/P8 및 결정 기록을 수정했다. Pilot 잠정 설정과 본 실험 확정값을 구분하고 코드 작성 금지를 유지했다.
- 2026-10-06: 사용자가 지정한 현재 여덟 노트를 읽고 자료 목록을 갱신했다. 사용자 요청에 따라 SSP indicator·inclusion 확률은 gamma·pi_gamma, loading 분산은 sigma_v2 및 IG로 통일했다. 현재 모델 기호에 대응하는 수식·상태·조건부분포·결정 기록을 정리했다.
- 같은 날 범위 결정: 사용자가 Inference를 보류하고 Sampling까지의 개발을 주 목표로 지정했다. 추론 정의를 P0 선행 조건에서 제거하고 P1–P6를 현재 개발 범위로 명시했다. 추론·본 실험은 후속으로 남기고 sampler correctness·mixing 검증은 유지했다. 문서만 수정했다.
- 같은 날 추가 범위 결정: 사용자가 Scalable SSP도 보류했다. 공통·Normal·Horseshoe·Reference SSP로 현재 개발 범위를 좁히고 Scalable의 proposal·수락비·성능 설계와 두 SSP 구현 비교를 현재 선행 조건·완료 조건에서 제거했다. 기존 Scalable 설계는 후속 참고로 보존하며 다음 논의 항목을 backend로 변경했다. 문서만 수정했다.
- 같은 날 환경·지원 범위 확정: 사용자가 Python/NumPy/SciPy·CPU·float64와 일반 p,D≤p,차수별 R_d 지원을 지정했다. 계산 가능성을 보장하는 범위를 주장하지 않고 첫 simulation/validation은 p=5,D=3으로 유지했다.
- 같은 날 default 제안 요청: 사용자가 초기화·표본 저장·검증 tolerance/diagnostic·pilot 잠정 수치의 default와 근거를 계획에 기록하도록 요청했다. §7.4–7.5, §9.1–9.3, §10.1–10.2에 변경 가능한 제안을 추가하고 본 실험 수치와 구분했다. 진단의 primary source 및 PG adapter 자료를 확인했다. 모델 코드·테스트·설정 파일 작성이나 설치·sampler/pilot 실행은 하지 않았다.
- 같은 날 준비 상태 점검: 현재 Sampling 범위의 확정 사항·default·후속 미결 항목을 검토하여 개발 착수를 막는 추가 사용자 결정이 없음을 확인했다. P0를 개발 준비 완료로 표시하고 패키지 설치·라이브러리/수치 검증은 구현 단계의 작업으로 구분했다. 이 기록은 코드 구현 또는 sampler correctness 검증 완료를 뜻하지 않는다.

- 같은 날 개발 진행 요청: 사용자가 현재 Sampling 범위의 개발과 GitHub 연동을 고려한 구조, uv.lock 사용 및 BayesianCalibration 참고를 요청했다. Python 3.13.15·uv_build 기준 환경, 자체 uv.lock, src 패키지·세 sampler·CLI·저장·diagnostics·configs·테스트와 GitHub 개발 파일을 구현했다. 전체 68개 테스트 및 lint/format·locked 설치·배포 빌드가 통과했고 9개 smoke fit은 수치 실패 없이 완료했다. 짧은 smoke의 혼합 진단은 모두 mixing_flagged로 보존했다. 로컬 Git을 초기화했으며 Archive·외부 노트는 변경하지 않았다. Full pilot·Inference·Scalable SSP·GitHub remote/push는 후속으로 남겼다.

# unity-20261008 결과 분석

분석일: 2026-10-08. 저장된 draws·설정·진단·실행 로그와 Slurm accounting을 대조했다.
**현재 결과에서 Normal이 가장 안정적이다. SSP는 long-run에서 혼합이 개선되었으며,
Normal과의 예측 성능 차이는 작고 설정에 따라 바뀐다. Horseshoe는 수치 실패와 일부
긴 꼬리의 혼합 문제가 있어 전체 방법 비교를 완료했다고 볼 수 없다.**

## 실제 실행 범위와 완료 상태

원래 준비 디렉터리는 pilot 39건, long-run 9건이지만, 실제 추가 실행은
`data/pilot-long-run-all-settings-20261008/manifest.json`에 기록된 long-run 39건을
대상으로 했다. 이 확장 manifest로 평가 CLI를 다시 실행했다. 두 준비 디렉터리의
13개 observations/truth 파일과 두 stage 설정은 byte 단위로 같고, 기존 작업의
method/seed/task 매핑도 같다. 완료된 fit의 observations와 seed는 평가 CLI에서 확인했다.

- 각 DGP: n=200, p=5, D=3, R_2=R_3=5, 관측 데이터 하나.
- Pilot: 4 chains × (burn-in 2,000 + retained 2,000).
- Long-run: 4 chains × (burn-in 4,000 + retained 20,000).
- 78건 중 **62건 완료, 7건 numerical_failure, 9건 결과 없음**. 완료 중 진단 통과는 34건.

| Stage | 방법 | 완료 | 진단 통과 | 완료했으나 mixing_flagged | 수치 실패 | 결과 없음 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | Normal | 13 | 11 | 2 | 0 | 0 |
| Pilot | Horseshoe | 12 | 0 | 12 | 1 | 0 |
| Pilot | Reference SSP | 13 | 0 | 13 | 0 | 0 |
| Long-run | Normal | 10 | 10 | 0 | 0 | 3 |
| Long-run | Horseshoe | 4 | 3 | 1 | 6 | 3 |
| Long-run | Reference SSP | 10 | 10 | 0 | 0 | 3 |

누락된 9건은 **no-interaction, sparse-3way-s3, dense-3way-s3 × 세 방법**이다.
Slurm `11254970_[0-8]`은 `CANCELLED`, elapsed=0, Start=None으로 기록되어 있다.
제출 기록에는 pilot 전체에 대한 `afterok:11254969` 의존성이 있고, pilot task 16은
실패했다. 따라서 해당 의존성은 충족되지 않았다. 다만 accounting의 Reason=None이므로
누가/어떤 절차가 취소했는지는 이 기록만으로 확정할 수 없다. 현재 이 세 배열의
실행·대기 작업은 없다. 이번 분석에서 작업을 제출하거나 취소하지 않았다.

## 혼합 진단

기존 기준은 R-hat < 1.01, bulk/tail ESS ≥ 400, beta/theta 상대 평균 MCSE ≤ 0.05,
확률 MCSE ≤ 0.01, sigma 계열 quantile MCSE ratio ≤ 0.05이다.
Fit 통과 여부에는 공통 58개 functional 외에 primary hyperparameter·상태 요약·
log likelihood/posterior 진단도 포함된다. 따라서 예측이 안정적이어도 fit 전체는 실패할 수 있다.

- **Normal:** pilot `sparse-3way-s3`는 theta의 R-hat/ESS/MCSE와 sigma_v2_3 ESS가
  문제다. `dense-3way-s1`은 sigma_v2_3 R-hat만 실패했다. 공통 functional만 보면
  pilot 12/13건, long-run 10/10건 통과한다.
- **Horseshoe:** 완료 pilot의 공통 functional은 6/12건 통과하지만, 전체 primary
  진단은 모두 실패했다. theta·prediction과 log_posterior의 혼합 경고가 반복된다.
  완료 long-run 4건 중 `sparse-3way-s2` 한 건이 계속 실패한다.
- **SSP:** pilot에서는 theta와 pi_gamma, T, M 등의 혼합 경고가 많다. 공통 functional만
  통과한 fit은 4/13건이다. 완료 long-run 10건은 모두 전체 기준을 통과한다.
  이 예산에서는 장기 실행이 효과적이지만, 빠진 강한 3차 상호작용 두 설정까지의 보장은 아니다.

## Horseshoe의 두 가지 문제

모든 수치 실패는 **eta cache check**이며, truncated Gamma 오류는 기록되지 않았다.
아래 chain은 1부터, sweep은 실패가 발생한 sweep이다.

| Stage | 설정 | Chain | Sweep | 기록된 최대 eta 차이 |
| --- | --- | ---: | ---: | ---: |
| Pilot | sparse-3way-s2 | 2 | 1,500 | 6.10e-8 |
| Long-run | sparse-2way-s2 | 4 | 21,900 | 1.45e-6 |
| Long-run | sparse-2way-s3 | 1 | 1,800 | 2.51e-6 |
| Long-run | sparse-3way-s1 | 3 | 9,700 | 8.81e-6 |
| Long-run | dense-2way-s1 | 4 | 12,100 | 9.17e-8 |
| Long-run | dense-2way-s3 | 2 | 9,500 | 4.12e-6 |
| Long-run | dense-3way-s2 | 2 | 13,400 | 4.33e-6 |

검사는 원소별 `atol=1e-8, rtol=1e-6`을 사용하므로 eta가 0 근처이면 작은 절대 차이도
실패할 수 있다. 기록된 최대 logit 차이를 단순히 확률로 환산한 상한은 최대 약 2.20e-6
(expit의 최대 미분 1/4)으로 현재 확률 MCSE보다 작다. 이는 **검사 시점의 두 predictor
차이**에 관한 설명이며, 이전 Gibbs 조건부 계산의 오차나 전체 궤적의 정확성을 보장하지 않는다.
오류 크기만 보고 fit을 성공으로 재분류하지 않았다. 큰 loading에서 recurrence cancellation이
발생했는지, 0 근처의 비교 기준이 주원인인지 실패 상태를 재현하여 구분하는 것이 다음 점검이다.
Tolerance와 sampler는 변경하지 않았다.

별도로, 완료된 `sparse-3way-s2` long-run은 **theta[10] = (0,1,2) 상호작용**이 불안정하다.

| 항목 | 값 |
| --- | ---: |
| 생성 truth | 2.000 |
| 표본 평균 / 중앙값 | 9.735 / 6.075 |
| 표본 5% / 95% quantile | 2.058 / 30.104 |
| R-hat | 1.0113 |
| Bulk / tail ESS | 378 / 202 |
| 평균 MCSE | 1.079 |
| Chain별 평균 | 8.214, 11.571, 10.652, 8.501 |

관측 데이터에서 X_0 X_1 X_2=1인 21개 행은 **전부 y=1**이다. 다른 계수를 고정하면
이 계수의 양의 방향을 likelihood가 억제하지 않는 구조다. 실제로 표본은 긴 오른쪽 꼬리와
체인별 평균 차이를 보인다. 이 데이터 구조는 계수의 불안정성을 설명하는 근거이며,
수치 실패의 직접 원인을 증명하는 것은 아니다. 확률 RMSE는 0.1145지만 theta3 RMSE는
2.4514이다. 예측 확률이 그럴듯하다는 사실만으로 interaction 크기를 신뢰할 수 없다.
이 fit은 수렴한 reference로 사용하지 않는다.

## 예측 성능과 계수 오차

확률 RMSE는 32개 binary pattern을 동일 가중하고, posterior mean probability를
생성 truth와 비교한다. 반복 표본 평균이나 학습 데이터의 분류 정확도가 아니다.
아래는 완료 long-run의 확률 RMSE이며, `실패`는 수치 실패다.

| 설정 | Normal | Horseshoe | SSP |
| --- | ---: | ---: | ---: |
| sparse-2way-s1 | 0.0798 | 0.0882 | 0.0693 |
| sparse-2way-s2 | 0.1004 | 실패 | 0.1125 |
| sparse-2way-s3 | 0.0918 | 실패 | 0.0897 |
| sparse-3way-s1 | 0.1243 | 실패 | 0.1111 |
| sparse-3way-s2 | 0.1159 | 0.1145 (혼합 경고) | 0.1254 |
| dense-2way-s1 | 0.0930 | 실패 | 0.0980 |
| dense-2way-s2 | 0.0743 | 0.0761 | 0.0784 |
| dense-2way-s3 | 0.1109 | 실패 | 0.1095 |
| dense-3way-s1 | 0.1547 | 0.1677 | 0.1420 |
| dense-3way-s2 | 0.1413 | 실패 | 0.1439 |

Normal·SSP가 공통으로 통과한 10개 설정에서 평균 RMSE는 **0.10864 대 0.10798**,
설정별 승수는 **5 대 5**다. 평균 expected log loss도 0.61577 대 0.61566으로 비슷하다.
이는 이 10개 고정 데이터의 기술적 요약이며 우열의 통계적 증거가 아니다.

SSP는 실제 3차 효과가 없는 2way DGP 6개 모두에서 Normal보다 theta3 RMSE가 작다.
반면 `sparse-3way-s2`에서는 theta3 RMSE가 Normal 0.2368, SSP 0.0579로 더 작으면서도
확률 RMSE는 SSP가 더 크다. 계수별 오차와 결합된 예측 오차는 같은 순위를 보장하지 않는다.
세 방법 모두 통과한 3개 설정의 평균 확률 RMSE는 Normal 0.1030, Horseshoe 0.1107,
SSP 0.0966이지만, 실패가 많은 Horseshoe의 전체 성능을 이 subset으로 대표할 수 없다.

## Pilot 대비 long-run 안정성

양쪽이 완료된 비교는 Normal 10쌍, Horseshoe 3쌍, SSP 10쌍이다.
Horseshoe sparse-3way-s2는 pilot 실패로 비교에서 빠진다.

| 방법 | 비교 functional | Reference 자격 충족 | long/pilot MCSE 비율 중앙값 | 최대 확률 평균 변화 |
| --- | ---: | ---: | ---: | ---: |
| Normal | 580 | 500 | 0.318 | 0.00752 |
| Horseshoe | 174 | 107 | 0.327 | 0.00553 |
| SSP | 580 | 347 | 0.326 | 0.00628 |

Reference 자격은 long-run 전체 진단 통과와 해당 functional의 MCSE ≤ pilot MCSE/3을
동시에 요구한다. Long-run 진단 통과만으로 모든 functional이 reference가 되지는 않는다.
총 1,334개 중 954개가 이 조건을 충족한다. MCSE 감소는 retained draws 10배 증가에서
기대할 수 있는 약 1/sqrt(10)=0.316과 대체로 비슷하다.

Combined-MCSE로 표준화한 평균 차이의 최대 절댓값은 Normal 3.65, Horseshoe 2.86,
SSP 3.00이다. Normal에서만 580개 중 1개가 3을 초과한다. 다중·상관된 functional 비교이며
일부 pilot은 진단 경고가 있으므로 이를 교정된 검정이나 sampler 정확성 증명으로 해석하지 않는다.

완료 fit의 sampling 시간 중앙값은 pilot Normal/Horseshoe/SSP 23.0/28.8/27.3초,
long-run 184.8/216.0/222.0초다. Sampling 시간은 burn-in을 포함하고 최종 diagnostics·
파일 저장을 제외한다. 서로 다른 노드와 서로 다른 완료 subset이 섞여 있어 정밀한
ESS/sec 우열 비교는 하지 않는다. 실패·누락을 제거한 시간/성능만으로 방법을 평가하면 편향된다.

## 산출물과 검증

- [전체 78건 fits.csv](../outputs/unity-20261008/report-analysis-20261008/fits.csv)
- [3,596개 functional 요약](../outputs/unity-20261008/report-analysis-20261008/functionals.csv)
- [stage 비교 CSV](../outputs/unity-20261008/report-analysis-20261008/pilot-vs-long-run.csv)
- [완료·진단 그림](../outputs/unity-20261008/report-analysis-20261008/completion-and-diagnostics.png)
- [확률 RMSE 그림](../outputs/unity-20261008/report-analysis-20261008/probability-rmse.png)
- [Horseshoe theta[10] trace·running mean](../outputs/unity-20261008/report-analysis-20261008/horseshoe-theta10.png)
- [그림·독립 계산 재현 스크립트](../outputs/unity-20261008/report-analysis-20261008/analyze.py)

완료 62건, 248개 chain의 첫/중간/마지막 retained draw 총 744개에서 직접 combinations로
theta와 확률을 재구성했다. 저장 theta와의 최대 절대 차이는 0, 확률은 5.00e-16이었다.
62건의 전체 확률 draws로 RMSE를 다시 계산한 최대 차이는 1.96e-15였다. 기존
atol=1e-8, rtol=1e-6 기준을 그대로 사용했다. 이는 저장/요약 일관성 점검이며 Gibbs
궤적 전체나 실패 chain을 검증한 것은 아니다.

`uv run pytest`: **103 passed (21.57s)**. `uv run ruff check .`: 통과.
실제 `experiment summarize` CLI가 78건/62건 완료를 보고했다. 새 sampling은 실행하지 않았다.
Outputs는 Git ignore 대상이며 기존 진행 중 보고서·raw 결과는 보존했다.

우선 후속 과제는 (1) Horseshoe cache 오류 재현과 오차 영향 확인,
(2) 누락된 9건과 실패 7건의 실행 계획 정리, (3) sparse-3way-s2의 긴 꼬리 혼합 점검이다.
현재 관측만으로 prior·tolerance를 바꾸거나 Inference/Scalable SSP로 범위를 넓히지는 않는다.

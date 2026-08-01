# SECOM 분석 — 도구 풀이 & "혼자 직접 해보기" 가이드

> 이 문서는 두 가지를 담는다.
> (A) **도구 용어 풀이** — Python·pandas·scikit-learn 같은 게 각각 뭔지.
> (B) **직접 재현 가이드** — 내가 한 분석을 처음부터 혼자 따라 할 수 있게 단계별 코드+설명.
> 실제 코드는 `secom_analysis.py` / `secom_eda.py` / `secom_advanced.py` 에 있고, 아래는 그 핵심을
> 초보자가 읽을 수 있게 풀어 쓴 것이다.

---

## A. 도구 용어 풀이 (이게 각각 뭐예요?)

| 도구 | 한 줄 정의 | 비유 |
|---|---|---|
| **Python** | 데이터 분석에서 제일 많이 쓰는 프로그래밍 언어 | 분석의 '공용어'. 문법이 쉬워 입문용 |
| **라이브러리(library)** | 남이 만들어 둔 기능 꾸러미. `import`로 불러 씀 | 요리의 '반조리 재료'. 매번 밑바닥부터 안 만듦 |
| **pandas** | 엑셀 같은 표(表)를 코드로 다루는 라이브러리 | '코드로 조작하는 엑셀'. 행·열 필터, 결측 처리 등 |
| **numpy** | 숫자 계산(배열·평균·수학)을 빠르게 하는 라이브러리 | '계산기 엔진'. pandas 밑에서 돌아감 |
| **scikit-learn (sklearn)** | 머신러닝 모델·전처리 도구 모음 | '머신러닝 공구세트'. 모델을 import 한 줄로 |
| **matplotlib** | 그래프(그림) 그리는 라이브러리 | '그림 그리는 붓'. 히스토그램·박스플롯 등 |
| **scipy** | 통계 검정·과학계산 라이브러리 | '통계 계산기'. Mann-Whitney U 검정 등 |
| **imbalanced-learn (imblearn)** | 불균형 데이터 처리(SMOTE 등) 라이브러리 | '소수 데이터 보정 도구' |
| **DataFrame** | pandas의 표 객체 (행+열) | 엑셀 시트 한 장 |
| **Pipeline** | 전처리~모델을 한 줄로 묶는 sklearn 장치 | '컨베이어 벨트'. 순서대로 자동 처리 |
| **모델(model) 학습(fit)** | 데이터로 규칙을 배우게 하는 것 | 문제집 풀며 공부시키기 |
| **예측(predict)** | 배운 모델로 새 데이터의 답을 맞히기 | 시험 보기 |

**설치는 어떻게?** 터미널에 한 줄:
```bash
pip install pandas numpy scikit-learn matplotlib scipy imbalanced-learn
```
`pip` = 파이썬 라이브러리 설치 도구(앱스토어 같은 것).

---

## B. 혼자 직접 해보기 — 단계별 (코드 + 왜 그렇게 하는지)

> 아래 코드는 실제 `secom_*.py`에서 뽑은 핵심이다. 한 줄씩 '무슨 뜻인지' 옆에 달았다.

### 0단계 — 준비물
- 파이썬 설치(아나콘다 추천), 위 `pip install` 한 줄 실행.
- 데이터 파일 `secom.data`, `secom_labels.data`를 `public_data/secom/` 폴더에 둠.

### 1단계 — 데이터 불러오기 (pandas)
```python
import pandas as pd          # pandas를 'pd'라는 별명으로 불러옴
import numpy as np

# 공백으로 구분된 표 읽기. 헤더(제목줄)가 없으니 header=None, 'NaN'은 결측으로 인식
X = pd.read_csv("public_data/secom/secom.data", sep=r"\s+", header=None, na_values="NaN")
X.columns = [f"S{i}" for i in range(X.shape[1])]   # 열 이름을 S0, S1, ... 로

# 라벨(정답) 읽기: "-1 날짜" 형태 → 앞 숫자만 씀. 1=불량, -1=합격
labels = []
with open("public_data/secom/secom_labels.data") as f:
    for line in f:
        labels.append(int(line.split()[0]))
y = (np.array(labels) == 1).astype(int)   # 불량이면 1, 아니면 0
```
**왜?** 원본이 그냥 숫자 덩어리라, 표(`X`)와 정답(`y`)으로 나눠 담는 게 첫 단추.

### 2단계 — 데이터 살펴보기 (EDA, 그림 그리기)
```python
import matplotlib.pyplot as plt
from scipy import stats

# 예: S59 센서를 합격/불량으로 나눠 히스토그램 겹쳐 그리기
fail = (y == 1)
plt.hist(X.loc[~fail, "S59"].dropna(), bins=30, alpha=0.5, label="pass")
plt.hist(X.loc[ fail, "S59"].dropna(), bins=30, alpha=0.5, label="fail")
plt.legend(); plt.title("S59 분포: 합격 vs 불량"); plt.show()

# 두 그룹 분포가 '진짜' 다른지 통계 검정 (정규분포 가정 없는 Mann-Whitney U)
u, p = stats.mannwhitneyu(X.loc[~fail,"S59"].dropna(), X.loc[fail,"S59"].dropna())
print("p값:", p)   # 작을수록(예:0.00000003) 불량과 관련 큼
```
**왜?** 모델 돌리기 전에 '어떤 센서가 불량과 관련 있나'를 눈(그림)과 숫자(검정) 둘 다로 확인.

### 3단계 — 전처리 + 모델을 컨베이어벨트(Pipeline)로 묶기 (scikit-learn)
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer            # 결측 채우기
from sklearn.feature_selection import VarianceThreshold  # 상수열 버리기
from sklearn.preprocessing import StandardScaler    # 눈금 맞추기
from sklearn.ensemble import RandomForestClassifier # 예측 모델

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # ① 빈칸을 중앙값으로 채움
    ("var",    VarianceThreshold(0.0)),             # ② 항상 같은 값인 센서 제거
    ("scale",  StandardScaler()),                   # ③ 모든 센서를 같은 잣대로
    ("clf",    RandomForestClassifier(              # ④ 랜덤포레스트로 예측
        n_estimators=300,                           #    나무 300그루
        class_weight="balanced_subsample",          #    드문 불량에 가중치(불균형 처리)
        random_state=42, n_jobs=-1)),
])
```
**왜 Pipeline?** ①~④를 따로 하면 실수로 '시험 답'을 전처리에 섞는 실수(데이터 누수)를 하기 쉬움.
벨트로 묶으면 학습 데이터로만 통계를 계산해 그 실수를 막음.

### 4단계 — 올바르게 성적 매기기 (정확도 함정 피하기)
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score  # = PR-AUC

# 데이터를 학습용 75% / 시험용 25%로 나눔 (불량 비율 유지: stratify)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pipe.fit(Xtr, ytr)                          # 학습(공부시키기)
proba = pipe.predict_proba(Xte)[:, 1]       # 시험지: 각 제품의 '불량 확률'
print("PR-AUC:", average_precision_score(yte, proba))  # 정확도 대신 이 점수를 봄
```
**왜 PR-AUC?** 불량이 6.6%뿐이라 정확도는 '다 합격'만 찍어도 93%. 불량을 실제로 잡는지 재려면
PR-AUC를 봐야 함.

### 5단계 — 실제 공장 조건으로 검증 (시간순)
```python
# 시간(날짜) 순서로 정렬해서 '앞 70%로 배우고 뒤 30%를 맞히기'
# → 무작위로 섞어 시험하면 성적이 부풀려짐(미래를 미리 본 셈)
# secom_drift.py 에 구현: temporal split. 결과 PR-AUC가 0.21→0.06으로 떨어지면
#   '모델이 시간 지나면 낡는다 = 재학습 필요'라는 뜻.
```

### 6단계 — 비용으로 판단 기준 정하기
```python
# '불량 놓침(손해 큼) vs 헛검사(손해 작음)' 비용을 10:1로 두고
# 확률 임계값을 0.01~0.99까지 훑어 '총비용이 가장 작은' 지점을 찾음.
# secom_advanced.py 에 구현: 기준선 0.5 대신 0.08에서 비용 38% 절감.
```

---

## C. 어떤 파일이 어느 단계인가 (재현 지도)

| 하고 싶은 것 | 실행 명령 | 나오는 결과 |
|---|---|---|
| 데이터 살펴보기(분포·검정) | `python src/secom_eda.py` | 히스토그램·박스플롯·통계랭킹 그림 4장 |
| 기본 예측 모델·성적 | `python src/secom_analysis.py` | PR-AUC, 상위센서, 그림 3장 |
| 시간순 검증·SPC 관리도 | `python src/secom_drift.py` | 드리프트 증거, 그림 3장 |
| 비용운영점·SMOTE 검증 | `python src/secom_advanced.py` | 비용 38%절감, 그림 3장 |
| 신뢰구간(부트스트랩) | `python src/secom_ci.py` | PR-AUC·불량률 변화 95% CI |
| XGBoost 튜닝 | `python src/secom_xgb.py` | GridSearchCV 최적 파라미터·PR-AUC |

> 모든 그림은 `results/` 폴더에 저장된다. 레포 루트에서 실행하며, 순서대로 돌리면 분석 전체가 재현된다.

---

## D. "혼자 할 수 있다"를 면접에서 이렇게 말하기

> "분석은 Python으로 했습니다. pandas로 표 데이터를 정리하고, scikit-learn으로 전처리(결측 대치·
> 표준화)와 랜덤 포레스트 모델을 파이프라인으로 묶어 데이터 누수 없이 학습시켰습니다. scipy로
> 통계 검정을, matplotlib으로 분포 시각화를 했고요. 무엇보다 각 단계를 **왜 그렇게 했는지**(왜
> 중앙값인지, 왜 정확도 대신 PR-AUC인지, 왜 시간순 검증인지) 근거를 갖고 선택했습니다. 모든 과정을
> 스크립트로 남겨 `python 파일명`으로 누구나 같은 결과를 재현할 수 있게 했습니다."

"""
SECOM 반도체 공정 수율 예측·이상센서 분석 (공개 데이터 트랙)
실행: python secom_analysis.py
출력: analysis_result_plot/secom_*.png  +  터미널 요약
데이터: public_data/secom/secom.data (1567런 × 590센서), secom_labels.data (pass/fail)
        UCI SECOM. 다운로드: curl .../secom/secom.data, secom_labels.data
설치: pip install scikit-learn pandas numpy matplotlib   (sklearn 신규 의존)

왜 이 트랙인가: 내 증착 데이터(254런)는 깨끗해서 '이상탐지·수율예측' 수치가 안 나온다.
    SECOM은 실제 반도체 라인의 590개 센서 + pass/fail 라벨이라, 양산 스케일의 이상탐지·
    수율영향 센서 식별을 '진짜 수치'로 증명하는 데이터축이다. 공개데이터이므로 외부LLM 사용도
    무방(랩 실데이터와 달리 보안 제약 없음 — public_data/ 폴더로 분리).

핵심 관점: pass가 ~93%라 '정확도'는 무의미(전부 pass로 찍어도 93%). 불량은 소수클래스이므로
    PR-AUC(평균정밀도)·recall로 평가하고, 수율에 영향 큰 상위 센서를 도출한다(공정 최적화 단서).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, classification_report,
                             confusion_matrix)

DATA_DIR = os.path.join("public_data", "secom")
OUTPUT_DIR = "results"
STEEL, CORAL = "steelblue", "coral"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load():
    X = pd.read_csv(os.path.join(DATA_DIR, "secom.data"), sep=r"\s+",
                    header=None, na_values="NaN")
    X.columns = [f"S{i}" for i in range(X.shape[1])]   # 센서 S0..S589
    # 라벨: "<label> <timestamp>" — label -1=pass, 1=fail
    labels, times = [], []
    with open(os.path.join(DATA_DIR, "secom_labels.data"), encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            labels.append(int(parts[0]))
            times.append(parts[1].strip('"') if len(parts) > 1 else "")
    y = (np.array(labels) == 1).astype(int)   # fail=1(양성/소수), pass=0
    return X, y, pd.to_datetime(times, format="%d/%m/%Y %H:%M:%S", errors="coerce")


def build_pipeline(clf):
    # NaN 대치(중앙값) → 상수열 제거 → 표준화 → 분류기. 불균형은 class_weight로 보정.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def main():
    X, y, times = load()
    n, p = X.shape
    n_fail = int(y.sum())
    base_acc = 1 - y.mean()
    print("=" * 60)
    print(f"  SECOM — {n}런 × {p}센서 | 불량(fail) {n_fail}건 = {y.mean()*100:.1f}%")
    print(f"  기준선: 전부 pass로 찍어도 정확도 {base_acc*100:.1f}% → 정확도는 무의미, PR-AUC로 평가")
    print(f"  결측 셀: {X.isna().mean().mean()*100:.1f}% | 기간: {times.min()} ~ {times.max()}")
    print("=" * 60)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    models = {
        "LogReg(bal)": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest(bal)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name, clf in models.items():
        pipe = build_pipeline(clf)
        cv_ap = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="average_precision", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        ap = average_precision_score(yte, proba)
        roc = roc_auc_score(yte, proba)
        results[name] = {"pipe": pipe, "proba": proba, "ap": ap, "roc": roc, "cv_ap": cv_ap}
        print(f"  [{name}]  CV PR-AUC {cv_ap.mean():.3f}±{cv_ap.std():.3f} | "
              f"test PR-AUC {ap:.3f} | ROC-AUC {roc:.3f}")
    print(f"  (무작위 기준 PR-AUC = 불량률 {y.mean():.3f})")

    best = max(results, key=lambda k: results[k]["ap"])
    proba = results[best]["proba"]

    # ── recall 우선 운영점: 불량 재현율 ~60% 지점의 정밀도 보고 (양산 스크리닝 관점)
    prec, rec, thr = precision_recall_curve(yte, proba)
    idx = np.argmin(np.abs(rec[:-1] - 0.60))
    op_thr = thr[idx]
    yhat = (proba >= op_thr).astype(int)
    print("-" * 60)
    print(f"  최우수 모델: {best} | 운영점(불량 recall≈0.60): precision {prec[idx]:.3f}")
    print(f"  혼동행렬(행=실제 pass/fail, 열=예측):\n{confusion_matrix(yte, yhat)}")
    print(classification_report(yte, yhat, target_names=["pass", "fail"], digits=3))

    # ── 그림 1: 클래스 불균형
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(["pass", "fail"], [int((y == 0).sum()), n_fail], color=[STEEL, CORAL])
    ax.set_ylabel("Wafers"); ax.set_title("SECOM class balance", fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate([int((y == 0).sum()), n_fail]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_1_class_balance.png", dpi=150)
    plt.close(fig); print("Saved: secom_1_class_balance.png")

    # ── 그림 2: PR 곡선 (두 모델 + 무작위 기준)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, c in zip(results, [STEEL, CORAL]):
        pr, rc, _ = precision_recall_curve(yte, results[name]["proba"])
        ax.plot(rc, pr, color=c, label=f"{name} (AP={results[name]['ap']:.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.6, label=f"random (AP={y.mean():.3f})")
    ax.set_xlabel("Recall (fail)"); ax.set_ylabel("Precision (fail)")
    ax.set_title("SECOM yield-fail Precision-Recall", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_2_pr_curve.png", dpi=150)
    plt.close(fig); print("Saved: secom_2_pr_curve.png")

    # ── 그림 3: 수율영향 상위 센서 (RandomForest 중요도, 원 센서 인덱스로 역매핑)
    rf = results["RandomForest(bal)"]["pipe"]
    kept = rf.named_steps["var"].get_support()            # VarianceThreshold 통과 열
    kept_cols = np.array(X.columns)[kept]
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=kept_cols)
    top = imp.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top.index, top.values, color=STEEL)
    ax.set_xlabel("Feature importance"); ax.set_ylabel("Sensor")
    ax.set_title("Top-15 yield-impacting sensors", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_3_top_sensors.png", dpi=150)
    plt.close(fig); print("Saved: secom_3_top_sensors.png")

    print("-" * 60)
    print("  수율영향 상위 5 센서:", ", ".join(top.index[::-1][:5]))
    print("  → 이 센서들의 공정 파라미터를 우선 점검·관리하면 불량 저감 여지 큼")


if __name__ == "__main__":
    main()

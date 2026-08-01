"""
SECOM 결측치 처리 비교 — 평균 vs 중앙값 vs KNN (왜 중앙값을 골랐나)
실행: python src/secom_impute_compare.py   (레포 루트에서)
출력: results/secom_impute_compare.png + 터미널 요약
전제: secom_analysis.py 와 같은 데이터. load() 재사용.

왜: "평균/중앙값/KNN 중 무엇으로 채웠나?"에 근거로 답하기 위해, 같은 모델(RandomForest)·같은
    분할에서 세 대치법의 PR-AUC를 실제로 비교한다. 센서 상당수가 왜도 큰 분포라 평균은 이상치에
    끌려간다는 가설을 수치로 확인한다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score

from secom_analysis import load, STEEL, CORAL, OUTPUT_DIR


def make_pipe(imputer):
    return Pipeline([
        ("impute", imputer),
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ])


def main():
    X, y, _ = load()
    y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    strategies = {
        "mean (평균)": SimpleImputer(strategy="mean"),
        "median (중앙값)": SimpleImputer(strategy="median"),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)

    print("=" * 64)
    print("  SECOM 결측치 처리 비교 — 평균 / 중앙값 / KNN (지표: PR-AUC)")
    print(f"  참고: 전체 결측 {X.isna().mean().mean()*100:.1f}%, 최대 센서 결측 {X.isna().mean().max()*100:.0f}%")
    print("=" * 64)

    results = {}
    for name, imp in strategies.items():
        pipe = make_pipe(imp)
        cv_ap = cross_val_score(pipe, Xtr, ytr, cv=cv,
                                scoring="average_precision", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        test_ap = average_precision_score(yte, pipe.predict_proba(Xte)[:, 1])
        results[name] = (cv_ap.mean(), cv_ap.std(), test_ap)
        print(f"  {name:<16} CV PR-AUC {cv_ap.mean():.3f}±{cv_ap.std():.3f} | "
              f"test PR-AUC {test_ap:.3f}")

    best = max(results, key=lambda k: results[k][0])
    print("-" * 64)
    print(f"  → CV 기준 최고: {best}")
    print("  해석: 차이가 작으면 '대치법 선택보다 강건성·속도'가 결정 기준. 중앙값은 왜도에 강건")
    print("        하고 KNN(590차원 거리계산)보다 빠르고 안정적이라 실무 기본값으로 적합.")

    # ── 그림: CV PR-AUC 막대(오차막대) ── (플롯 라벨은 영문: 폰트 안전)
    keys = list(results.keys())
    names = ["mean", "median", "KNN (k=5)"]
    means = [results[k][0] for k in keys]
    stds = [results[k][1] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=[STEEL, "#2f8f6b", CORAL])
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.6, label=f"random {y.mean():.3f}")
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("CV PR-AUC (fail)")
    ax.set_title("Imputation comparison: mean vs median vs KNN", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_impute_compare.png", dpi=150)
    plt.close(fig); print("Saved: secom_impute_compare.png")


if __name__ == "__main__":
    main()

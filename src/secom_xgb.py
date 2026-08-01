"""
SECOM XGBoost + GridSearchCV — 하이퍼파라미터 최적화 파이프라인
실행: python secom_xgb.py
출력: analysis_result_plot/secom_xgb_pr.png + 터미널(최적 파라미터·PR-AUC 비교)
전제: secom_analysis.py 와 같은 데이터. load() 재사용.
설치: pip install xgboost   (신규 의존)

왜 이 실습인가 (이 포트폴리오 원칙 유지):
  - 튜닝 기준을 '정확도'가 아니라 PR-AUC(average_precision)로 둔다. 불균형(6.6%)에서 정확도로
    최적화하면 '전부 합격' 방향으로 튜닝돼 무의미하기 때문.
  - 불균형은 XGBoost의 scale_pos_weight(= 음성수/양성수)로 보정.
  - 데이터 누수 차단: 전처리~모델을 Pipeline으로 묶어 GridSearchCV가 각 fold 학습분할로만
    대치 통계를 계산하게 한다.
  - 튜닝된 XGBoost를 기존 RandomForest(PR-AUC 0.235)와 '같은 test'에서 공정 비교한다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

from secom_analysis import load, build_pipeline, STEEL, CORAL, OUTPUT_DIR


def main():
    X, y, _ = load()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    # 불균형 보정 가중치: 음성(합격) 수 / 양성(불량) 수
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    print("=" * 64)
    print(f"  SECOM XGBoost 튜닝 | scale_pos_weight = {spw:.1f} (불균형 보정)")
    print("  튜닝 기준(scoring) = average_precision(PR-AUC), 정확도 아님")
    print("=" * 64)

    # ── 파이프라인: 대치 → 상수열 제거 → XGBoost (트리라 표준화 불필요) ──
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            scale_pos_weight=spw, tree_method="hist",
            random_state=42, n_jobs=-1)),
    ])

    # ── GridSearchCV 탐색 격자 (clf__ 접두어로 파이프라인 단계 지정) ──
    grid = {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, grid, scoring="average_precision",
                      cv=cv, n_jobs=-1, verbose=0)
    n_comb = np.prod([len(v) for v in grid.values()])
    print(f"  탐색: {n_comb}개 조합 × 5-fold = {n_comb*5}회 학습 ...")
    gs.fit(Xtr, ytr)

    print("-" * 64)
    print(f"  최적 파라미터: {gs.best_params_}")
    print(f"  최적 CV PR-AUC: {gs.best_score_:.3f}")

    # ── 튜닝 XGB vs 기본 XGB vs RandomForest, 같은 test 에서 비교 ──
    best = gs.best_estimator_
    ap_xgb = average_precision_score(yte, best.predict_proba(Xte)[:, 1])

    default_xgb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("clf", XGBClassifier(objective="binary:logistic", eval_metric="aucpr",
                              scale_pos_weight=spw, tree_method="hist",
                              random_state=42, n_jobs=-1)),
    ]).fit(Xtr, ytr)
    ap_xgb0 = average_precision_score(yte, default_xgb.predict_proba(Xte)[:, 1])

    rf = build_pipeline(RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1)).fit(Xtr, ytr)
    ap_rf = average_precision_score(yte, rf.predict_proba(Xte)[:, 1])

    print("-" * 64)
    print(f"  [test PR-AUC 비교]  (무작위 기준 {y.mean():.3f})")
    print(f"    RandomForest(기존)   : {ap_rf:.3f}")
    print(f"    XGBoost(기본)        : {ap_xgb0:.3f}")
    print(f"    XGBoost(GridSearch)  : {ap_xgb:.3f}   "
          f"(튜닝 이득 {ap_xgb - ap_xgb0:+.3f})")

    # ── PR 곡선 비교 ──
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for proba, c, lab, ap in [
        (rf.predict_proba(Xte)[:, 1], STEEL, "RandomForest", ap_rf),
        (best.predict_proba(Xte)[:, 1], CORAL, "XGBoost (tuned)", ap_xgb)]:
        pr, rc, _ = precision_recall_curve(yte, proba)
        ax.plot(rc, pr, color=c, label=f"{lab} (AP={ap:.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.6,
               label=f"random (AP={y.mean():.3f})")
    ax.set_xlabel("Recall (fail)"); ax.set_ylabel("Precision (fail)")
    ax.set_title("SECOM: tuned XGBoost vs RandomForest", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_xgb_pr.png", dpi=150)
    plt.close(fig); print("Saved: secom_xgb_pr.png")

    print("-" * 64)
    print("  교훈: 튜닝은 PR-AUC 기준으로. 정확도로 튜닝하면 소수클래스를 버리는 방향으로 최적화됨.")


if __name__ == "__main__":
    main()

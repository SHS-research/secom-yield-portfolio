"""
SECOM 결측(빈 칸) 자체가 신호인가? — '채우기 vs 비워두고 의미 해석' 비교
실행: python src/secom_missing.py   (레포 루트에서)
출력: results/secom_missing_signal.png + 터미널 요약
전제: secom_analysis.py 와 같은 데이터. load() 재사용.

질문(정당한 데이터 분석 논쟁):
  "빈 칸을 그럴듯한 값으로 채우면 '왜 비어 있었나'라는 정보가 사라지지 않나?"
  → 맞다. 그래서 세 가지 전략을 실측 비교한다.
    A) 중앙값 대치만 (기존 baseline)           — 빈 칸의 의미를 버림
    B) 중앙값 대치 + '결측 플래그' 특성 추가     — 값도 채우고, 빈 칸의 '있었다'도 특성으로 보존
    C) XGBoost 네이티브 결측 처리 (채우지 않음)  — 모델이 빈 칸을 스스로 학습
  또한 '결측 자체'가 불량과 관련 있는지(런별 결측 개수, 결측 플래그만으로의 예측력)를 본다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from secom_analysis import load, STEEL, CORAL, OUTPUT_DIR

FLAG = "#C00000"


def ap(pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    return average_precision_score(yte, pipe.predict_proba(Xte)[:, 1])


def main():
    X, y, _ = load()
    y = np.asarray(y)
    miss = X.isna()                      # 결측이면 True
    per_run_missing = miss.sum(axis=1)   # 런(웨이퍼)별 빈 칸 개수

    print("=" * 66)
    print("  SECOM — 빈 칸(결측)은 그냥 잡음인가, 신호인가?")
    print("=" * 66)

    # ── ① 결측 자체가 불량과 관련 있나? (런별 결측 개수 pass vs fail) ──
    m_pass = per_run_missing[y == 0]
    m_fail = per_run_missing[y == 1]
    u, p = stats.mannwhitneyu(m_pass, m_fail, alternative="two-sided")
    print(f"  런별 결측 개수: pass 평균 {m_pass.mean():.1f} vs fail 평균 {m_fail.mean():.1f} "
          f"(Mann-Whitney p={p:.3g})")
    sig = "관련 있음(유의)" if p < 0.05 else "뚜렷한 관련 없음"
    print(f"  → 결측량과 불량의 관계: {sig}")

    # ── ② '결측 플래그만'으로 불량 예측이 되나? (값은 아예 안 쓰고) ──
    #    각 센서의 NaN 여부(0/1)만으로 예측 → 빈 칸의 '위치 패턴'에 신호가 있는지.
    flags = miss.astype(int)
    flags = flags.loc[:, flags.nunique() > 1]     # 항상 같은(전부측정/전부결측) 열 제거
    Xtr, Xte, ytr, yte = train_test_split(
        flags, y, test_size=0.25, stratify=y, random_state=42)
    ap_flags = ap(Pipeline([("clf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1))]), Xtr, ytr, Xte, yte)
    print(f"  '결측 플래그만'으로 예측한 PR-AUC: {ap_flags:.3f} "
          f"(무작위 기준 {y.mean():.3f}) — 기준보다 크면 빈 칸 위치에 신호 있음")

    # ── ③ 세 전략 PR-AUC 비교 (같은 분할) ──
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    # A) 중앙값 대치만
    pipe_A = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    ap_A = ap(pipe_A, Xtr, ytr, Xte, yte)

    # B) 중앙값 대치 + 결측 플래그(add_indicator=True)
    pipe_B = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("var", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1))])
    ap_B = ap(pipe_B, Xtr, ytr, Xte, yte)

    # C) XGBoost 네이티브 결측 (채우지 않고 NaN 그대로)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    pipe_C = Pipeline([("clf", XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", scale_pos_weight=spw,
        tree_method="hist", missing=np.nan, random_state=42, n_jobs=-1))])
    ap_C = ap(pipe_C, Xtr, ytr, Xte, yte)

    print("-" * 66)
    print("  [세 전략 test PR-AUC 비교]")
    print(f"    A) 중앙값 대치만              : {ap_A:.3f}  (빈 칸 의미 버림)")
    print(f"    B) 대치 + 결측 플래그 추가     : {ap_B:.3f}  (빈 칸의 '있었다'도 특성화)")
    print(f"    C) XGBoost 네이티브 결측       : {ap_C:.3f}  (채우지 않음)")
    best = max([("A", ap_A), ("B", ap_B), ("C", ap_C)], key=lambda t: t[1])
    print(f"  → 최고: {best[0]} ({best[1]:.3f}). "
          f"B가 A보다 높으면 '빈 칸을 특성으로 남기는 게' 실제로 이득.")

    # ── 그림: 결측개수 분포 + 전략 비교 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(m_pass, bins=30, alpha=0.6, color=STEEL, label="pass", density=True)
    ax1.hist(m_fail, bins=30, alpha=0.6, color=CORAL, label="fail", density=True)
    ax1.set_xlabel("Missing sensors per run"); ax1.set_ylabel("density")
    ax1.set_title(f"Missingness per run (MW p={p:.2g})", fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True, linestyle="--", alpha=0.4)

    names = ["A: impute\nonly", "B: impute +\nmissing-flag", "C: XGBoost\nnative NaN"]
    vals = [ap_A, ap_B, ap_C]
    colors = [STEEL, "#2f8f6b", CORAL]
    ax2.bar(names, vals, color=colors)
    ax2.axhline(y.mean(), color="gray", ls="--", alpha=0.6, label=f"random {y.mean():.3f}")
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Test PR-AUC (fail)")
    ax2.set_title("Missing-value strategy comparison", fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_missing_signal.png", dpi=150)
    plt.close(fig); print("Saved: secom_missing_signal.png")


if __name__ == "__main__":
    main()

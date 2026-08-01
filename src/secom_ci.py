"""
SECOM 신뢰구간(부트스트랩) — 점 추정값에 '폭'을 붙인다
실행: python secom_ci.py
출력: analysis_result_plot/secom_ci_prauc.png / _ci_driftdiff.png + 터미널 요약
전제: secom_analysis.py 와 같은 데이터. load/build_pipeline 재사용.

왜: PR-AUC 0.235, 불량률 차이 같은 숫자는 '표본 하나'에서 나온 점 추정값이다. 진짜 값이
    어디쯤인지 '범위'로 말하려면 신뢰구간이 필요하다. 정규분포 가정을 세우기 어려우므로
    (PR-AUC는 이론 표준오차 공식이 없다) 데이터를 재추출하는 '부트스트랩'으로 구간을 뽑는다.

부트스트랩이란: 가진 데이터에서 '복원추출'(같은 표본을 여러 번 뽑을 수 있음)로 새 표본을 B개
    만들고, 매번 통계량을 다시 계산 → 그 B개 분포의 2.5%/97.5% 분위수가 95% 신뢰구간.
    "만약 실험을 여러 번 반복했다면 값이 어떻게 흩어졌을까"를 데이터로 시뮬레이션하는 것.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from secom_analysis import load, build_pipeline, STEEL, CORAL, OUTPUT_DIR

FLAG = "#C00000"
B = 2000            # 부트스트랩 반복 횟수 (많을수록 구간이 안정적)
rng = np.random.default_rng(42)


def pctl_ci(samples, alpha=0.05):
    """percentile 방식 신뢰구간: 부트스트랩 분포의 2.5%/97.5% 분위수."""
    lo = np.percentile(samples, 100 * alpha / 2)
    hi = np.percentile(samples, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    X, y, times = load()
    print("=" * 66)
    print(f"  SECOM 부트스트랩 신뢰구간 (B={B}, 95% 구간)")
    print("=" * 66)

    # ── ① 모델 성능 PR-AUC 의 신뢰구간 ────────────────────────────────
    #   방법: train으로 1번 학습 → test 예측확률 고정 → test '표본'을 복원추출로
    #   B번 다시 뽑아 매번 PR-AUC 재계산 → 그 분포의 2.5/97.5% 분위수.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = build_pipeline(RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1))
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    yte = np.asarray(yte)

    point_ap = average_precision_score(yte, proba)
    n = len(yte)
    boot_ap = np.empty(B)
    b = 0
    while b < B:
        idx = rng.integers(0, n, n)              # 복원추출로 test 표본 재구성
        if yte[idx].sum() == 0:                  # 불량이 0개면 PR-AUC 정의불가 → 재추출
            continue
        boot_ap[b] = average_precision_score(yte[idx], proba[idx])
        b += 1
    lo, hi = pctl_ci(boot_ap)
    print(f"  [PR-AUC]  점추정 {point_ap:.3f} | 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"           무작위기준(불량률) {y.mean():.3f} — CI 하한이 이보다 크면 '진짜 예측력' 있음")
    verdict = "있음(하한>기준선)" if lo > y.mean() else "불확실(기준선 걸침)"
    print(f"           판정: 예측력 {verdict}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(boot_ap, bins=40, color=STEEL, alpha=0.8)
    ax.axvline(point_ap, color="black", lw=2, label=f"point est. {point_ap:.3f}")
    ax.axvline(lo, color=FLAG, ls="--", label=f"95% CI [{lo:.3f}, {hi:.3f}]")
    ax.axvline(hi, color=FLAG, ls="--")
    ax.axvline(y.mean(), color="gray", ls=":", label=f"random baseline {y.mean():.3f}")
    ax.set_xlabel("Bootstrapped PR-AUC"); ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap 95% CI of model PR-AUC", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_ci_prauc.png", dpi=150)
    plt.close(fig); print("Saved: secom_ci_prauc.png")

    # ── ② 전반기 vs 후반기 불량률 '차이'의 신뢰구간 (드리프트 검증) ─────
    #   방법: 기간 중앙값으로 전/후반 분할. 각 그룹에서 불량여부를 복원추출로
    #   B번 뽑아 (후반 불량률 - 전반 불량률)을 매번 계산 → 차이의 CI.
    df = pd.DataFrame({"y": y, "t": times}).dropna(subset=["t"]).sort_values("t")
    mid = df["t"].iloc[len(df) // 2]
    early = df[df["t"] < mid]["y"].to_numpy()
    late = df[df["t"] >= mid]["y"].to_numpy()
    p_early, p_late = early.mean(), late.mean()
    point_diff = p_late - p_early

    boot_diff = np.empty(B)
    for i in range(B):
        e = rng.choice(early, len(early), replace=True)
        l = rng.choice(late, len(late), replace=True)
        boot_diff[i] = l.mean() - e.mean()
    dlo, dhi = pctl_ci(boot_diff)
    print("-" * 66)
    print(f"  [불량률 차이]  전반기 {p_early*100:.1f}% → 후반기 {p_late*100:.1f}% "
          f"(차이 {point_diff*100:+.1f}%p)")
    print(f"                95% CI [{dlo*100:+.1f}%p, {dhi*100:+.1f}%p]")
    crosses = dlo < 0 < dhi
    print(f"                판정: {'0을 걸침 → 변화 단정 못함(우연 가능)' if crosses else '0 안 걸침 → 유의미한 변화'}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(boot_diff * 100, bins=40, color=CORAL, alpha=0.8)
    ax.axvline(point_diff * 100, color="black", lw=2, label=f"point est. {point_diff*100:+.1f}%p")
    ax.axvline(dlo * 100, color=FLAG, ls="--", label=f"95% CI [{dlo*100:+.1f}, {dhi*100:+.1f}]%p")
    ax.axvline(dhi * 100, color=FLAG, ls="--")
    ax.axvline(0, color="gray", lw=1.5, label="no change (0)")
    ax.set_xlabel("Late - Early defect rate  [%p]"); ax.set_ylabel("Frequency")
    ax.set_title("Bootstrap 95% CI of defect-rate change", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_ci_driftdiff.png", dpi=150)
    plt.close(fig); print("Saved: secom_ci_driftdiff.png")

    print("-" * 66)
    print("  요약: 점 추정값에 '폭'을 붙여, 개선/변화가 우연인지 진짜인지를 0 기준으로 판정.")


if __name__ == "__main__":
    main()

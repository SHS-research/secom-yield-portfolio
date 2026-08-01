"""
SECOM 탐색적 데이터 분석(EDA) — 분포 파악 · 공정 파라미터 가정 · 통계적 판별
실행: python secom_eda.py
출력: analysis_result_plot/secom_eda_hist.png / _box.png / _stat_rank.png / _corr.png
      + 터미널 요약(공정 파라미터 가정표, 통계검정 상위센서)
전제: secom_analysis.py 와 같은 데이터. load() 재사용.
설치: pip install scipy   (통계검정용)

왜: 모델을 돌리기 전에 '원 데이터가 무엇인가'를 먼저 본다. SECOM 센서는 익명화(S0..S589)라
    물리적 정체를 모르므로, 각 센서의 '통계적 지문'(변동계수·왜도·이봉성·시간드리프트)으로
    fab의 어떤 공정 파라미터 부류인지 가정하고, pass/fail 분포 차이를 통계검정으로 정량화한다.
    → RandomForest 중요도(모델 기반)와 별개로, '가설검정 기반 특성선택'을 교차검증한다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from secom_analysis import load, STEEL, CORAL, OUTPUT_DIR

FLAG = "#C00000"


def classify_sensor(x):
    """센서의 통계적 지문 → fab 공정 파라미터 '부류' 가정.
       (익명 데이터라 물리적 확정은 불가 — 분포 형태 기반의 공학적 가정)"""
    x = x.dropna()
    if len(x) < 30 or x.std() == 0:
        return "상수/불활성 (미사용 채널·상수 셋포인트)"
    cv = x.std() / (abs(x.mean()) + 1e-9)
    skew = stats.skew(x)
    # 이봉성 지표: 히스토그램 dip 근사 (분위 간격 대비 중앙 밀도)
    kurt = stats.kurtosis(x)
    if cv < 0.01:
        return "초정밀 제어값 (MFC 가스유량·압력 셋포인트)"
    if abs(skew) > 3:
        return "카운트/이벤트성 (파티클·누설·아크 카운트)"
    if kurt < -1.0:
        return "이봉/다봉 (챔버·레시피·설비 간 차이)"
    if 0.01 <= cv < 0.1:
        return "안정 아날로그 계측 (온도·압력·RF파워)"
    return "광변동 계측 (막두께·증착률·유효면적)"


def main():
    X, y, times = load()
    fail = y == 1

    # ── ① 가설검정 기반 특성선택: pass vs fail 분포차 (Mann-Whitney U) ──
    #     정규성 가정 없이 두 그룹 분포차를 검정. p작을수록 불량과 관련 큰 센서.
    recs = []
    for c in X.columns:
        a = X.loc[~fail, c].dropna()
        b = X.loc[fail, c].dropna()
        if len(a) < 20 or len(b) < 10 or (a.nunique() + b.nunique()) < 3:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        # 효과크기(rank-biserial) = 방향·크기
        rbc = 1 - 2 * u / (len(a) * len(b))
        recs.append((c, p, abs(rbc), X[c].isna().mean()))
    rank = pd.DataFrame(recs, columns=["sensor", "p", "effect", "missing"])
    # 다중검정 보정(Benjamini-Hochberg FDR)
    rank = rank.sort_values("p").reset_index(drop=True)
    m = len(rank)
    rank["p_fdr"] = (rank["p"] * m / (rank.index + 1)).clip(upper=1.0)
    sig = rank[rank["p_fdr"] < 0.05]
    top = rank.head(8)

    print("=" * 70)
    print(f"  SECOM EDA — {X.shape[0]}런 × {X.shape[1]}센서 | 검정가능 {m}센서")
    print(f"  FDR<0.05 로 pass/fail 분포차 유의한 센서: {len(sig)}개")
    print("=" * 70)
    print("  [가설검정 기반 상위 8센서]  (모델 중요도와 별개의 특성선택 경로)")
    print(f"  {'센서':<6}{'p_fdr':>10}{'효과크기':>9}{'결측%':>7}  공정파라미터 가정")
    for _, r in top.iterrows():
        klass = classify_sensor(X[r["sensor"]])
        print(f"  {r['sensor']:<6}{r['p_fdr']:>10.2e}{r['effect']:>9.2f}"
              f"{r['missing']*100:>6.0f}%  {klass}")

    top_sensors = top["sensor"].tolist()

    # ── ② 히스토그램: 상위 6센서, pass vs fail 겹쳐 그리기 ──────────────
    show = top_sensors[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), show):
        a = X.loc[~fail, c].dropna(); b = X.loc[fail, c].dropna()
        lo, hi = np.nanpercentile(X[c], [1, 99])
        bins = np.linspace(lo, hi, 30)
        ax.hist(a, bins=bins, density=True, alpha=0.55, color=STEEL, label="pass")
        ax.hist(b, bins=bins, density=True, alpha=0.55, color=CORAL, label="fail")
        ax.set_title(f"{c}  (p_fdr={top.set_index('sensor').loc[c,'p_fdr']:.1e})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("value"); ax.set_ylabel("density")
        ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("SECOM top sensors — distribution: pass vs fail (histogram)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_eda_hist.png", dpi=150)
    plt.close(fig); print("Saved: secom_eda_hist.png")

    # ── ③ 박스플롯: 상위 6센서 pass/fail, 이상치(IQR) 가시화 ────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), show):
        a = X.loc[~fail, c].dropna(); b = X.loc[fail, c].dropna()
        bp = ax.boxplot([a, b], tick_labels=["pass", "fail"], patch_artist=True,
                        widths=0.6, showfliers=True, flierprops=dict(marker=".", markersize=3))
        for patch, col in zip(bp["boxes"], [STEEL, CORAL]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        # IQR 이상치 개수 주석
        for i, s in enumerate([a, b]):
            q1, q3 = np.percentile(s, [25, 75]); iqr = q3 - q1
            out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
            ax.text(i + 1, ax.get_ylim()[1], f"out={out}", ha="center",
                    va="top", fontsize=7, color=FLAG)
        ax.set_title(c, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("SECOM top sensors — boxplot & IQR outliers: pass vs fail",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_eda_box.png", dpi=150)
    plt.close(fig); print("Saved: secom_eda_box.png")

    # ── ④ 통계검정 랭킹 막대: -log10(p_fdr) 상위 15 ────────────────────
    r15 = rank.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(r15["sensor"], -np.log10(r15["p_fdr"].clip(lower=1e-300)), color=STEEL)
    ax.axvline(-np.log10(0.05), color=FLAG, ls="--", alpha=0.8, label="FDR=0.05")
    ax.set_xlabel("-log10(p_fdr)  (higher = stronger pass/fail separation)")
    ax.set_ylabel("Sensor")
    ax.set_title("Hypothesis-test feature ranking (Mann-Whitney U + BH-FDR)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_eda_stat_rank.png", dpi=150)
    plt.close(fig); print("Saved: secom_eda_stat_rank.png")

    # ── ⑤ 상위센서 상관 히트맵: 다중공선성(중복정보) 점검 ──────────────
    sub = X[top_sensors].corr().abs()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_sensors))); ax.set_xticklabels(top_sensors, rotation=45, fontsize=8)
    ax.set_yticks(range(len(top_sensors))); ax.set_yticklabels(top_sensors, fontsize=8)
    for i in range(len(top_sensors)):
        for j in range(len(top_sensors)):
            ax.text(j, i, f"{sub.iloc[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if sub.iloc[i, j] > 0.5 else "black")
    ax.set_title("Top-sensor |correlation| (multicollinearity check)", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_eda_corr.png", dpi=150)
    plt.close(fig); print("Saved: secom_eda_corr.png")

    hi_corr = [(top_sensors[i], top_sensors[j], sub.iloc[i, j])
               for i in range(len(top_sensors)) for j in range(i + 1, len(top_sensors))
               if sub.iloc[i, j] > 0.7]
    print("-" * 70)
    if hi_corr:
        print("  다중공선성(|r|>0.7) 쌍:", ", ".join(f"{a}-{b}({r:.2f})" for a, b, r in hi_corr))
        print("  → 서로 중복정보. 특성선택 시 한쪽만 남기면 모델 간결·안정")
    else:
        print("  상위센서 간 |r|>0.7 없음 → 서로 독립적 정보(중복 적음)")
    print(f"  결측 요약: 전체 셀 {X.isna().mean().mean()*100:.1f}% 결측, "
          f"결측률 최대 센서 {X.isna().mean().max()*100:.0f}%")


if __name__ == "__main__":
    main()

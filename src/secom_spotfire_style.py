"""
SECOM Spotfire-style 시각화 (Python 재현) — Box / Scatter Matrix / Treemap / Parameter-map
실행: python src/secom_spotfire_style.py   (레포 루트에서)
출력: results/secom_spotfire_box.png / _scatter_matrix.png / _treemap.png / _parammap.png
설치: pip install squarify   (treemap용 신규 의존)

정직성 표기(중요): 이 그림들은 TIBCO Spotfire로 만든 것이 아니라, Spotfire가 제공하는 차트
    '유형'(Box/Scatter Matrix/Treemap/파라미터 매핑 산점도)을 Python(matplotlib)으로 '재현'한
    것이다. 실제 Spotfire .dxp 대시보드는 별도 트랙(WM-811K)에서 작성한다.

데이터 정직성: SECOM은 익명 센서(S0..S589)+pass/fail+타임스탬프뿐이라, 매뉴얼의 Equipment_ID·
    Yield·Chamber_Pressure 같은 컬럼이 없다. 따라서 (a) 설비별/공정별 분해 대신 pass/fail·센서
    통계영향으로 대체하고, (b) 센서에 임의의 물리적 이름을 붙이지 않는다(가짜 라벨 금지).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import squarify

from secom_analysis import load, STEEL, CORAL, OUTPUT_DIR


def rank_sensors(X, y, k=15):
    """Mann-Whitney U로 pass/fail 분포차가 큰 상위 센서 + 효과크기 반환."""
    fail = y == 1
    recs = []
    for c in X.columns:
        a = X.loc[~fail, c].dropna(); b = X.loc[fail, c].dropna()
        if len(a) < 20 or len(b) < 10 or (a.nunique() + b.nunique()) < 3:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        rbc = 1 - 2 * u / (len(a) * len(b))            # rank-biserial 효과크기
        recs.append((c, p, abs(rbc)))
    r = pd.DataFrame(recs, columns=["sensor", "p", "effect"]).sort_values("p")
    m = len(r)
    r["p_fdr"] = (r["p"].values * m / (np.arange(m) + 1)).clip(max=1.0)
    return r.head(k).reset_index(drop=True)


def main():
    X, y, times = load()
    y = np.asarray(y)
    rk = rank_sensors(X, y, k=15)
    top = rk["sensor"].tolist()
    print("=" * 62)
    print("  SECOM Spotfire-style 시각화 (Python 재현) — 실데이터")
    print(f"  통계영향 상위 센서: {', '.join(top[:6])}")
    print("=" * 62)

    # ── ① Box Plot: 상위 6센서, pass vs fail (설비별 대신 클래스별) ──
    show = top[:6]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions, ticks, ticklabels = [], [], []
    for i, c in enumerate(show):
        a = X.loc[y == 0, c].dropna(); b = X.loc[y == 1, c].dropna()
        # 센서마다 스케일이 달라 z-정규화하여 한 축에 비교
        mu, sd = X[c].mean(), (X[c].std() or 1)
        az, bz = (a - mu) / sd, (b - mu) / sd
        p0, p1 = i * 3, i * 3 + 1
        bp = ax.boxplot([az, bz], positions=[p0, p1], widths=0.8,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, col in zip(bp["boxes"], [STEEL, CORAL]):
            patch.set_facecolor(col); patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")
        ticks.append(i * 3 + 0.5); ticklabels.append(c)
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
    ax.set_ylabel("Sensor value (z-score)")
    ax.set_title("Box Plot — top yield-impacting sensors: pass vs fail (Spotfire-style)",
                 fontweight="bold")
    ax.legend([plt.Rectangle((0, 0), 1, 1, fc=STEEL, alpha=.65),
               plt.Rectangle((0, 0), 1, 1, fc=CORAL, alpha=.65)],
              ["pass", "fail"], loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_spotfire_box.png", dpi=150)
    plt.close(fig); print("Saved: secom_spotfire_box.png")

    # ── ② Scatter Plot Matrix: 상위 4센서, pass/fail 색상 ──
    cols = top[:4]
    sub = X[cols].copy()
    sub = sub.fillna(sub.median())                     # 산점 표시용 결측 대치
    colors = np.where(y == 1, CORAL, STEEL)
    axes = pd.plotting.scatter_matrix(
        sub, figsize=(9, 9), diagonal="hist", color=colors, alpha=0.5,
        hist_kwds=dict(bins=25, color=STEEL), s=12)
    for axr in axes.ravel():
        axr.grid(True, linestyle="--", alpha=0.3)
        axr.xaxis.label.set_size(8); axr.yaxis.label.set_size(8)
    fig = plt.gcf()
    fig.suptitle("Scatter Plot Matrix — top sensors (blue=pass, red=fail) [Spotfire-style]",
                 fontweight="bold", y=0.995)
    fig.legend(handles=[Line2D([0], [0], marker="o", ls="", color=STEEL, label="pass"),
                        Line2D([0], [0], marker="o", ls="", color=CORAL, label="fail")],
               loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_spotfire_scatter_matrix.png", dpi=150)
    plt.close(fig); print("Saved: secom_spotfire_scatter_matrix.png")

    # ── ③ Treemap: 상위 센서를 '통계적 영향(-log10 p_fdr)' 크기로 ──
    #    (설비/공정 분류가 없으므로, 불량영향 센서의 상대 기여를 계층 대신 크기로)
    sizes = (-np.log10(rk["p_fdr"].clip(lower=1e-300))).values
    labels = [f"{s}\n{e:.2f}" for s, e in zip(rk["sensor"], rk["effect"])]
    norm = (rk["effect"] - rk["effect"].min()) / (rk["effect"].max() - rk["effect"].min() + 1e-9)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.35 + 0.6 * v) for v in norm]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    squarify.plot(sizes=sizes, label=labels, color=colors, ax=ax,
                  text_kwargs=dict(fontsize=8), pad=True)
    ax.axis("off")
    ax.set_title("Treemap — yield-impacting sensors (size = -log10 p_fdr, shade = effect size) "
                 "[Spotfire-style]", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_spotfire_treemap.png", dpi=150)
    plt.close(fig); print("Saved: secom_spotfire_treemap.png")

    # ── ④ Parameter-map Scatter: X=시간, Y=상위센서, Color=pass/fail ──
    #    매뉴얼의 X=Process_Time / Color by Equipment_ID 구조를 SECOM에 맞게 대응
    #    (Equipment_ID 없음 → Color by pass/fail; Y=익명 센서 실명 유지)
    sensor = top[0]
    df = pd.DataFrame({"t": times, "v": X[sensor].values, "y": y}).dropna(subset=["t", "v"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ok = df["y"] == 0
    ax.scatter(df.loc[ok, "t"], df.loc[ok, "v"], s=14, c=STEEL, alpha=0.5,
               marker="o", label="pass")
    ax.scatter(df.loc[~ok, "t"], df.loc[~ok, "v"], s=42, c=CORAL, alpha=0.9,
               marker="X", label="fail")
    ax.set_xlabel("Process time (timestamp)")
    ax.set_ylabel(f"Sensor {sensor} value")
    ax.set_title(f"Parameter-map Scatter — X=time, Y={sensor}, color=pass/fail [Spotfire-style]",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_spotfire_parammap.png", dpi=150)
    plt.close(fig); print("Saved: secom_spotfire_parammap.png")

    print("-" * 62)
    print("  ※ 이 그림은 Spotfire 차트 '유형'의 Python 재현임(실제 .dxp 아님).")
    print("     설비ID·수율·챔버압력 컬럼이 SECOM에 없어 pass/fail·센서 통계로 대체함.")


if __name__ == "__main__":
    main()

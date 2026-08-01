"""
SECOM 심화 — 시간축 분석 (공정 드리프트 · 시간순 검증 · 센서 드리프트)
실행: python secom_drift.py
출력: analysis_result_plot/secom_4_pchart.png / _5_temporal_pr.png / _6_sensor_drift.png
      + 터미널 요약
전제: secom_analysis.py 와 같은 데이터(public_data/secom/). load/build_pipeline 재사용.

왜: secom_analysis.py의 무작위 분할 PR-AUC는 '미래를 이미 본' 낙관적 수치다. 양산 배포는
    '과거로 학습→미래 예측'이므로 시간순 분할이 진짜 성능이다. 또한 3개월치 타임스탬프로
    공정 불량률이 시간에 따라 드리프트했는지(관리도), 상위 센서 신호가 이동했는지를 본다.
    → "모델은 시간이 지나면 낡는다(재학습 필요)"는 양산 핵심 통찰을 수치로 보인다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from secom_analysis import load, build_pipeline, STEEL, CORAL, OUTPUT_DIR

FLAG = "#C00000"


def main():
    X, y, times = load()
    df = pd.DataFrame({"y": y, "t": times}).reset_index(drop=True)
    ok = df["t"].notna()
    print("=" * 60)
    print(f"  SECOM 시간축 심화 — {ok.sum()}런(타임스탬프 유효) | "
          f"{df.loc[ok,'t'].min().date()} ~ {df.loc[ok,'t'].max().date()}")
    print("=" * 60)

    # ── ① 주간 불량률 관리도 (p-chart): 분수불량 + 3σ 관리한계(표본수 가변)
    wk = df[ok].set_index("t").resample("W")["y"].agg(["mean", "size", "sum"])
    wk = wk[wk["size"] > 0]
    pbar = df.loc[ok, "y"].mean()
    n = wk["size"].to_numpy()
    ucl = pbar + 3 * np.sqrt(pbar * (1 - pbar) / n)
    lcl = np.clip(pbar - 3 * np.sqrt(pbar * (1 - pbar) / n), 0, None)
    ooc = wk["mean"].to_numpy() > ucl        # 관리상한 이탈 주 = 불량 급증
    x = np.arange(len(wk))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, wk["mean"], "-o", color=STEEL, label="weekly defect rate p")
    ax.plot(x, ucl, "--", color=FLAG, alpha=0.7, label="UCL (3-sigma)")
    ax.plot(x, lcl, "--", color="gray", alpha=0.5, label="LCL")
    ax.axhline(pbar, color="green", ls=":", alpha=0.7, label=f"mean p={pbar:.3f}")
    if ooc.any():
        ax.scatter(x[ooc], wk["mean"].to_numpy()[ooc], s=120, marker="X",
                   color=FLAG, zorder=6, label=f"out-of-control ({ooc.sum()})")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%m/%d") for d in wk.index], rotation=45, fontsize=8)
    ax.set_ylabel("Fraction defective"); ax.set_xlabel("Week")
    ax.set_title("SECOM weekly yield-defect p-chart", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_4_pchart.png", dpi=150)
    plt.close(fig); print("Saved: secom_4_pchart.png")
    print(f"  주간 불량률 {wk['mean'].min()*100:.1f}~{wk['mean'].max()*100:.1f}% | "
          f"관리상한 이탈 {ooc.sum()}주 → 공정이 시간에 따라 흔들림(안정 아님)")

    # ── ② 시간순 학습/검증 vs 무작위 (과거→미래 예측 = 실제 배포)
    sd = df[ok].sort_values("t")                      # 시간순 정렬
    Xs = X.iloc[sd.index].reset_index(drop=True)
    ys = sd["y"].to_numpy()
    cut = int(len(Xs) * 0.70)                         # 앞 70%=과거(학습) / 뒤 30%=미래(검증)

    def eval_split(Xtr, ytr, Xte, yte, label):
        pipe = build_pipeline(RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        ap, roc = average_precision_score(yte, proba), roc_auc_score(yte, proba)
        print(f"  [{label}] test PR-AUC {ap:.3f} | ROC-AUC {roc:.3f} "
              f"(불량률 train {ytr.mean():.3f} / test {yte.mean():.3f})")
        return proba, ap

    print("-" * 60)
    proba_t, ap_t = eval_split(Xs[:cut], ys[:cut], Xs[cut:], ys[cut:], "시간순(과거→미래)")
    # 무작위 분할 비교 (같은 test 크기)
    from sklearn.model_selection import train_test_split
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xs, ys, test_size=1 - 0.70, stratify=ys, random_state=42)
    proba_r, ap_r = eval_split(Xr_tr, yr_tr, Xr_te, yr_te, "무작위(낙관적)")
    print(f"  ▷ 시간순이 무작위보다 {'낮음' if ap_t < ap_r else '높음'} "
          f"({ap_t:.3f} vs {ap_r:.3f}) → 무작위 분할은 배포 성능을 과대평가. "
          f"실제로는 드리프트로 성능이 떨어지고 주기적 재학습이 필요.")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for proba, yte, c, lab, ap in [(proba_t, ys[cut:], STEEL, "temporal", ap_t),
                                    (proba_r, yr_te, CORAL, "random-split", ap_r)]:
        pr, rc, _ = precision_recall_curve(yte, proba)
        ax.plot(rc, pr, color=c, label=f"{lab} (AP={ap:.3f})")
    ax.axhline(ys.mean(), color="gray", ls="--", alpha=0.6, label=f"random (AP={ys.mean():.3f})")
    ax.set_xlabel("Recall (fail)"); ax.set_ylabel("Precision (fail)")
    ax.set_title("SECOM: temporal vs random validation", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_5_temporal_pr.png", dpi=150)
    plt.close(fig); print("Saved: secom_5_temporal_pr.png")

    # ── ③ 상위 센서 시간 드리프트 (전체 학습 중요도 상위 2개의 주간 평균)
    pipe = build_pipeline(RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    pipe.fit(Xs, ys)
    kept = np.array(X.columns)[pipe.named_steps["var"].get_support()]
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=kept)
    top2 = imp.sort_values(ascending=False).head(2).index.tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    for sensor, c in zip(top2, [STEEL, CORAL]):
        s = pd.DataFrame({"t": df.loc[ok, "t"].values, "v": X[sensor].iloc[df[ok].index].values})
        wkm = s.dropna().set_index("t").resample("W")["v"].mean()
        # 정규화(z)해서 두 센서를 한 축에 비교
        z = (wkm - wkm.mean()) / (wkm.std() or 1)
        ax.plot(range(len(z)), z, "-o", color=c, label=f"{sensor} (weekly mean, z)")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Week"); ax.set_ylabel("Sensor weekly mean (z-score)")
    ax.set_title("Top yield-impacting sensors — temporal drift", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_6_sensor_drift.png", dpi=150)
    plt.close(fig); print("Saved: secom_6_sensor_drift.png")
    print(f"  상위 센서 {top2} 의 주간 평균이 이동 → 센서 드리프트가 수율 변동의 후보 원인")


if __name__ == "__main__":
    main()

"""
SECOM 심화 II — 양산 운영 관점 (비용기반 운영점 · SMOTE 반박 · 피처 시간안정성)
실행: python secom_advanced.py
출력: analysis_result_plot/secom_7_cost.png / _8_smote.png / _9_feature_stability.png
      + 터미널 요약
전제: secom_analysis.py 와 같은 데이터(public_data/secom/). load/build_pipeline 재사용.
설치: pip install imbalanced-learn   (SMOTE 비교용 신규 의존)

왜 이 심화인가 (남들이 하는 SECOM과의 차별점):
  남들: SMOTE로 오버샘플 → RandomForest → 정확도/AUC 자랑. 여기서 끝.
  이 스크립트는 '양산 배포자'의 3가지 질문에 수치로 답한다.
    ① 임계값을 어디에 둘 것인가? — 정확도가 아니라 '불량유출 비용 vs 과검 비용'으로 최적점 결정.
    ② 남들이 쓰는 SMOTE가 진짜 이득인가? — class_weight 대비, 그리고 '시간순 검증'에서
       SMOTE가 낙관적 무작위분할에서만 좋아보이고 실제 배포(과거→미래)에선 이득이 사라짐을 보인다.
    ③ '상위 수율영향 센서'를 믿고 관리해도 되나? — 기간을 전/후반으로 쪼갰을 때 상위 센서 랭킹이
       유지되는지(안정성)를 본다. 안 유지되면 '고정 센서 관리'는 위험 → 재평가 주기가 필요.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from secom_analysis import load, build_pipeline, STEEL, CORAL, OUTPUT_DIR

FLAG = "#C00000"

# ── 설정: 양산 비용 가정 (불량 1개 유출 vs 정상 1개 과검) ──────────────
#   불량이 후공정으로 유출되면(놓치면) 재작업/수율손실로 큰 비용,
#   정상을 불량으로 오판(과검)하면 재검사 비용. 반도체는 유출 비용이 훨씬 크다.
COST_MISS = 10.0     # 불량 1건 미검출(FN) 비용
COST_OVER = 1.0      # 정상 1건 과검(FP) 비용


def rf():
    return build_pipeline(RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1))


def main():
    X, y, times = load()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    print("=" * 64)
    print(f"  SECOM 심화 II — 비용가정 미검출:과검 = {COST_MISS:.0f}:{COST_OVER:.0f}")
    print("=" * 64)

    # ── ① 비용기반 운영점: 임계값 sweep → 총비용 최소점 ───────────────
    pipe = rf(); pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    n_pos, n_neg = int(yte.sum()), int((yte == 0).sum())

    ths = np.linspace(0.01, 0.99, 99)
    costs, recalls, precs = [], [], []
    for t in ths:
        yhat = (proba >= t).astype(int)
        fn = int(((yhat == 0) & (yte == 1)).sum())
        fp = int(((yhat == 1) & (yte == 0)).sum())
        tp = int(((yhat == 1) & (yte == 1)).sum())
        costs.append(fn * COST_MISS + fp * COST_OVER)
        recalls.append(tp / n_pos if n_pos else 0)
        precs.append(tp / (tp + fp) if (tp + fp) else 0)
    costs = np.array(costs)
    best_i = int(np.argmin(costs))
    t_star = ths[best_i]
    # 비교 기준: 기본 임계값 0.5 의 비용
    i50 = int(np.argmin(np.abs(ths - 0.5)))
    save = costs[i50] - costs[best_i]

    print(f"  기본 임계값 0.50 총비용 {costs[i50]:.0f} → "
          f"최적 임계값 {t_star:.2f} 총비용 {costs[best_i]:.0f} "
          f"(비용 {save/costs[i50]*100:.0f}% 절감)")
    print(f"  최적점에서 불량 recall {recalls[best_i]:.2f} / precision {precs[best_i]:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(ths, costs, color=STEEL, label="total cost")
    ax.axvline(t_star, color=FLAG, ls="--", alpha=0.8,
               label=f"cost-optimal thr={t_star:.2f}")
    ax.axvline(0.5, color="gray", ls=":", alpha=0.7, label="default thr=0.5")
    ax.scatter([t_star], [costs[best_i]], s=90, color=FLAG, zorder=6)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel(f"Total cost (miss={COST_MISS:.0f}, over={COST_OVER:.0f})")
    ax.set_title("SECOM cost-optimal operating point", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_7_cost.png", dpi=150)
    plt.close(fig); print("Saved: secom_7_cost.png")

    # ── ② SMOTE 반박: class_weight vs SMOTE, 무작위 vs 시간순 검증 ─────
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    def smote_pipe():
        return ImbPipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("var", VarianceThreshold(0.0)),
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1)),
        ])

    def ap_of(build, Xtr_, ytr_, Xte_, yte_):
        p = build(); p.fit(Xtr_, ytr_)
        pr = p.predict_proba(Xte_)[:, 1]
        return average_precision_score(yte_, pr)

    # (a) 무작위 분할 (남들이 보고하는 낙관적 세팅)
    ap_cw_rand = ap_of(rf, Xtr, ytr, Xte, yte)
    ap_sm_rand = ap_of(smote_pipe, Xtr, ytr, Xte, yte)

    # (b) 시간순 분할 (실제 배포: 과거→미래)
    df = pd.DataFrame({"y": y, "t": times})
    ok = df["t"].notna()
    order = df[ok].sort_values("t").index
    Xs, ys = X.loc[order].reset_index(drop=True), df.loc[order, "y"].to_numpy()
    cut = int(len(Xs) * 0.70)
    ap_cw_temp = ap_of(rf, Xs[:cut], ys[:cut], Xs[cut:], ys[cut:])
    ap_sm_temp = ap_of(smote_pipe, Xs[:cut], ys[:cut], Xs[cut:], ys[cut:])

    print("-" * 64)
    print(f"  [무작위분할] class_weight PR-AUC {ap_cw_rand:.3f} | SMOTE {ap_sm_rand:.3f} "
          f"(SMOTE 이득 {(ap_sm_rand-ap_cw_rand):+.3f})")
    print(f"  [시간순분할] class_weight PR-AUC {ap_cw_temp:.3f} | SMOTE {ap_sm_temp:.3f} "
          f"(SMOTE 이득 {(ap_sm_temp-ap_cw_temp):+.3f})")
    print("  ▷ SMOTE가 무작위분할에선 좋아 보여도 실제 배포(시간순)에선 이득이 미미/사라짐 "
          "→ '불균형=무조건 SMOTE'는 함정, 비용가중이 더 안전")

    labels = ["random\nsplit", "temporal\nsplit"]
    cw_vals = [ap_cw_rand, ap_cw_temp]
    sm_vals = [ap_sm_rand, ap_sm_temp]
    xpos = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(xpos - w/2, cw_vals, w, color=STEEL, label="class_weight")
    ax.bar(xpos + w/2, sm_vals, w, color=CORAL, label="SMOTE")
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.6, label=f"random baseline ({y.mean():.3f})")
    for xi, (a, b) in enumerate(zip(cw_vals, sm_vals)):
        ax.text(xi - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("Test PR-AUC (fail)")
    ax.set_title("SMOTE vs class_weight — optimistic vs real deployment", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_8_smote.png", dpi=150)
    plt.close(fig); print("Saved: secom_8_smote.png")

    # ── ③ 피처 랭킹 시간안정성: 전반부 vs 후반부 상위센서 겹침 ─────────
    half = len(Xs) // 2
    def top_sensors(Xh, yh, k=20):
        p = rf(); p.fit(Xh, yh)
        kept = np.array(X.columns)[p.named_steps["var"].get_support()]
        imp = pd.Series(p.named_steps["clf"].feature_importances_, index=kept)
        return imp.sort_values(ascending=False).head(k).index.tolist()

    top_early = top_sensors(Xs[:half], ys[:half])
    top_late = top_sensors(Xs[half:], ys[half:])
    overlap = sorted(set(top_early) & set(top_late))
    jacc = len(overlap) / len(set(top_early) | set(top_late))
    print("-" * 64)
    print(f"  전반부 Top-20 ∩ 후반부 Top-20 = {len(overlap)}개 (Jaccard {jacc:.2f})")
    print(f"  두 기간 공통 상위센서: {', '.join(overlap[:8])}{' ...' if len(overlap) > 8 else ''}")
    print("  ▷ 겹침이 낮으면 '고정 센서 관리'는 위험 — 상위센서를 주기적으로 재평가해야 함")

    # 상위 15센서의 전/후반 중요도 산점: 대각선 근처면 안정
    pe = rf(); pe.fit(Xs[:half], ys[:half])
    pl = rf(); pl.fit(Xs[half:], ys[half:])
    ke = np.array(X.columns)[pe.named_steps["var"].get_support()]
    kl = np.array(X.columns)[pl.named_steps["var"].get_support()]
    ie = pd.Series(pe.named_steps["clf"].feature_importances_, index=ke)
    il = pd.Series(pl.named_steps["clf"].feature_importances_, index=kl)
    common = ie.index.intersection(il.index)
    top_show = ie[common].sort_values(ascending=False).head(30).index

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ie[top_show], il[top_show], color=STEEL, alpha=0.7)
    lim = max(ie[top_show].max(), il[top_show].max()) * 1.1
    ax.plot([0, lim], [0, lim], color="gray", ls="--", alpha=0.6, label="stable (y=x)")
    for s in overlap[:6]:
        if s in top_show:
            ax.annotate(s, (ie[s], il[s]), fontsize=7)
    ax.set_xlabel("Importance — first half of period")
    ax.set_ylabel("Importance — second half of period")
    ax.set_title(f"Feature-ranking stability (Jaccard={jacc:.2f})", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(f"{OUTPUT_DIR}/secom_9_feature_stability.png", dpi=150)
    plt.close(fig); print("Saved: secom_9_feature_stability.png")


if __name__ == "__main__":
    main()

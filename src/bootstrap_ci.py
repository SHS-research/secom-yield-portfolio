"""
범용 부트스트랩 신뢰구간 도구 — 내 (보안) 공정 데이터에 직접 쓰는 로컬 전용 스크립트
실행 예:
  python bootstrap_ci.py --demo                          # 합성 데이터로 동작 시연
  python bootstrap_ci.py --csv mydata.csv --col GPC      # 한 열의 '평균' 95% 구간
  python bootstrap_ci.py --excel "ALD.xlsx" --sheet Sheet1 --col 불량률 \
                         --group 장비 --a NEX-1 --b NEX-2 # 두 그룹 '차이'의 구간

★ 보안(중요): 이 스크립트는 완전 로컬이다. 데이터를 어디로도 전송하지 않고, 화면에 '구간 숫자'만
   출력한다. 랩 공정 데이터(레시피=핵심 IP)에 그대로 써도 된다 — CLAUDE.md 데이터-경계 원칙 준수.
   단, 부트스트랩 재추출 표본은 '실제 값의 복사본'이라 익명화가 아니다. 재추출 결과를 파일로
   내보내 외부에 공유하지 말 것(값이 그대로 들어있음). 이 도구는 파일로 저장하지 않는다.

부트스트랩 원리(요약):
   가진 데이터에서 '복원추출'(replace=True)로 같은 크기 표본을 B번 만들고, 매번 통계량을
   다시 계산 → 그 B개 분포의 2.5%/97.5% 분위수 = 95% 신뢰구간.
"""
import argparse
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

rng = np.random.default_rng(42)   # 시드 고정 → 재현 가능


# ── 부트스트랩 핵심 함수 (여기만 이해하면 됨) ─────────────────────────
def bootstrap_ci(data, stat_fn, B=2000, alpha=0.05):
    """data 배열을 복원추출로 B번 재추출하며 stat_fn(통계량)을 재계산 → 신뢰구간.
       stat_fn: 배열 하나를 받아 숫자 하나를 돌려주는 함수 (예: np.mean, np.median)."""
    data = np.asarray(data)
    n = len(data)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)        # ← 복원추출: 같은 크기 '가짜 새 표본'의 위치
        boots[i] = stat_fn(data[idx])      # ← 그 표본으로 통계량 다시 계산
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return stat_fn(data), lo, hi, boots


def bootstrap_diff_ci(a, b, stat_fn=np.mean, B=2000, alpha=0.05):
    """두 그룹 a, b를 각각 복원추출 → (b통계 - a통계) '차이'의 신뢰구간.
       차이 구간이 0을 안 걸치면 '두 그룹이 유의미하게 다르다'."""
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(B)
    for i in range(B):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        diffs[i] = stat_fn(sb) - stat_fn(sa)
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return stat_fn(b) - stat_fn(a), lo, hi, diffs


# ── 데이터 로딩 + CLI ────────────────────────────────────────────────
def load_column(args):
    import pandas as pd
    if args.excel:
        df = pd.read_excel(args.excel, sheet_name=args.sheet or 0, header=args.header)
    else:
        df = pd.read_csv(args.csv)
    if args.col not in df.columns:
        sys.exit(f"[!] '{args.col}' 열이 없습니다. 가능한 열: {list(df.columns)}")
    return df


def main():
    ap = argparse.ArgumentParser(description="로컬 부트스트랩 신뢰구간 도구")
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 시연")
    ap.add_argument("--csv"); ap.add_argument("--excel")
    ap.add_argument("--sheet"); ap.add_argument("--header", type=int, default=0)
    ap.add_argument("--col", help="구간을 낼 숫자 열 이름")
    ap.add_argument("--stat", default="mean", choices=["mean", "median"])
    ap.add_argument("--group", help="두 그룹 비교 시: 그룹을 나누는 열")
    ap.add_argument("--a"); ap.add_argument("--b", help="비교할 두 그룹 값")
    ap.add_argument("-B", type=int, default=2000)
    args = ap.parse_args()

    stat_fn = np.mean if args.stat == "mean" else np.median
    name = args.stat

    if args.demo:
        print("=" * 60)
        print("  [DEMO] 합성 공정 데이터로 부트스트랩 시연 (실데이터 아님)")
        print("=" * 60)
        # 예: 개선 전(평균 5.0) vs 개선 후(평균 4.3) 불량률 흉내
        before = rng.normal(5.0, 1.5, 80).clip(0)
        after = rng.normal(4.3, 1.5, 80).clip(0)
        pt, lo, hi, _ = bootstrap_ci(after, stat_fn, B=args.B)
        print(f"  개선 후 {name} {pt:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
        d, dlo, dhi, _ = bootstrap_diff_ci(before, after, stat_fn, B=args.B)
        print(f"  (개선후 - 개선전) 차이 {d:+.2f}  95% CI [{dlo:+.2f}, {dhi:+.2f}]")
        verdict = "0을 걸침 → 개선 단정 못함" if dlo < 0 < dhi else "0 안 걸침 → 유의미한 차이"
        print(f"  판정: {verdict}")
        print("\n  내 데이터로 하려면:")
        print("    python bootstrap_ci.py --csv 내파일.csv --col 열이름")
        print("    python bootstrap_ci.py --excel 내파일.xlsx --col GPC --group 장비 --a NEX-1 --b NEX-2")
        return

    if not (args.csv or args.excel) or not args.col:
        ap.error("실데이터 모드는 --csv/--excel 과 --col 이 필요합니다 (또는 --demo).")

    df = load_column(args)

    if args.group:   # 두 그룹 차이 모드
        if not (args.a and args.b):
            sys.exit("[!] --group 에는 --a, --b 두 그룹 값이 필요합니다.")
        a = df.loc[df[args.group].astype(str) == args.a, args.col].dropna().to_numpy()
        b = df.loc[df[args.group].astype(str) == args.b, args.col].dropna().to_numpy()
        if len(a) < 5 or len(b) < 5:
            sys.exit(f"[!] 표본이 너무 적습니다 (a={len(a)}, b={len(b)}).")
        print("=" * 60)
        print(f"  {args.col} 의 {name}: '{args.a}'(n={len(a)}) vs '{args.b}'(n={len(b)})")
        d, dlo, dhi, _ = bootstrap_diff_ci(a, b, stat_fn, B=args.B)
        print(f"  차이(b-a) {d:+.4g}  95% CI [{dlo:+.4g}, {dhi:+.4g}]")
        print(f"  판정: {'0을 걸침 → 차이 단정 못함' if dlo < 0 < dhi else '0 안 걸침 → 유의미한 차이'}")
    else:            # 단일 열 평균/중앙값 모드
        x = df[args.col].dropna().to_numpy()
        if len(x) < 5:
            sys.exit(f"[!] 표본이 너무 적습니다 (n={len(x)}).")
        pt, lo, hi, _ = bootstrap_ci(x, stat_fn, B=args.B)
        print("=" * 60)
        print(f"  {args.col} (n={len(x)}) 의 {name}: {pt:.4g}")
        print(f"  95% 신뢰구간 [{lo:.4g}, {hi:.4g}]  (B={args.B})")


if __name__ == "__main__":
    main()

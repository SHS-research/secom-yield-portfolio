"""프레임 PNG들을 애니메이션 GIF로 합침 (Pillow) — README 상단 데모용.
실행: python web/build_gif.py
입력: results/demo/frames/frame_*.png   출력: results/demo/portfolio-demo.gif
"""
import os, glob, sys
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FRAMES = os.path.join("results", "demo", "frames")
OUT = os.path.join("results", "demo", "portfolio-demo.gif")
WIDTH = 760          # README 표시용 다운스케일 폭
FRAME_MS = 90        # 프레임당 시간(ms)

files = sorted(glob.glob(os.path.join(FRAMES, "frame_*.png")))
if not files:
    sys.exit(f"[!] 프레임이 없습니다: {FRAMES} (먼저 node web/capture_frames.js 실행)")

imgs = []
for f in files:
    im = Image.open(f).convert("RGB")
    w, h = im.size
    im = im.resize((WIDTH, round(h * WIDTH / w)), Image.LANCZOS)
    imgs.append(im)

# 마지막 프레임 잠깐 정지 효과: 끝 프레임 몇 개 복제
imgs += [imgs[-1]] * 6

imgs[0].save(OUT, save_all=True, append_images=imgs[1:], loop=0,
             duration=FRAME_MS, optimize=True, disposal=2)
kb = os.path.getsize(OUT) / 1024
print(f"GIF 저장: {OUT}  ({len(imgs)} frames, {kb:.0f} KB)")

// README 상단용 데모 GIF 프레임 캡처 — web/index.html 을 스크롤·클릭하며 뷰포트 스냅샷 저장
// 실행: node web/capture_frames.js   (사전: web/ 에서 npm i -D playwright && npx playwright install chromium)
// 출력: results/demo/frames/frame_###.png  (이후 build_gif.py 가 GIF로 합침)
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const framesDir = path.resolve(__dirname, '..', 'results', 'demo', 'frames');
  fs.rmSync(framesDir, { recursive: true, force: true });
  fs.mkdirSync(framesDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  const fileUrl = 'file://' + path.resolve(__dirname, 'index.html').replace(/\\/g, '/');
  await page.goto(fileUrl);
  await page.waitForTimeout(400);

  let i = 0;
  const snap = async () => {
    await page.screenshot({ path: path.join(framesDir, `frame_${String(i).padStart(3, '0')}.png`) });
    i++;
  };

  // 상단에서 잠깐 머물기
  for (let k = 0; k < 4; k++) { await snap(); await page.waitForTimeout(60); }

  // 스크롤 다운 (전체 페이지 훑기)
  const height = await page.evaluate(() => document.body.scrollHeight - window.innerHeight);
  const steps = 26;
  for (let s = 1; s <= steps; s++) {
    await page.evaluate((y) => window.scrollTo(0, y), Math.round((height * s) / steps));
    await page.waitForTimeout(30);
    await snap();
  }

  // 발견(details) 펼치기
  const summaries = page.locator('details summary');
  const n = await summaries.count();
  for (let j = 1; j < n; j++) { await summaries.nth(j).click(); await page.waitForTimeout(120); await snap(); }
  await snap();

  // 다크모드 토글
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(120);
  await page.click('#theme-toggle');
  for (let k = 0; k < 6; k++) { await snap(); await page.waitForTimeout(80); }

  await context.close();
  await browser.close();
  console.log(`captured ${i} frames -> ${framesDir}`);
})();

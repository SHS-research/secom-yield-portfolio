// Playwright 자동 시연 녹화 — web/index.html 을 스크롤·클릭하며 영상(.webm)으로 저장
// 실행: node web/demo.js   (사전: npm install -D playwright && npx playwright install chromium)
// ★ OBS 같은 외부 녹화 프로그램 불필요 — Playwright 내장 recordVideo 가 영상을 직접 저장한다.
// 결과: results/demo/portfolio-demo.webm
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

(async () => {
  const demoDir = path.resolve(__dirname, '..', 'results', 'demo');
  fs.mkdirSync(demoDir, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: demoDir, size: { width: 1280, height: 800 } },
  });
  const page = await context.newPage();

  // 로컬 파일을 file:// URL 로 로드
  const fileUrl = 'file://' + path.resolve(__dirname, 'index.html').replace(/\\/g, '/');
  console.log('1. 포트폴리오 접속:', fileUrl);
  await page.goto(fileUrl);
  await page.waitForTimeout(1200);

  console.log('2. 스무스 스크롤 다운...');
  const height = await page.evaluate(() => document.body.scrollHeight);
  for (let y = 0; y <= height; y += 12) {
    await page.evaluate((v) => window.scrollTo(0, v), y);
    await page.waitForTimeout(16);
  }
  await page.waitForTimeout(800);

  console.log('3. 발견 상세(details) 펼치기...');
  const summaries = page.locator('details summary');
  const n = await summaries.count();
  for (let i = 0; i < n; i++) { await summaries.nth(i).click(); await page.waitForTimeout(500); }
  await page.waitForTimeout(600);

  console.log('4. 다크모드 토글 시연...');
  await page.click('#theme-toggle');
  await page.waitForTimeout(1500);

  console.log('5. 상단으로 복귀...');
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  await page.waitForTimeout(1500);

  // context.close() 가 있어야 영상이 파일로 flush 된다
  await context.close();
  await browser.close();

  // 생성된 .webm 을 고정 이름으로 정리
  const webm = fs.readdirSync(demoDir).find((f) => f.endsWith('.webm'));
  if (webm) {
    const finalPath = path.join(demoDir, 'portfolio-demo.webm');
    if (path.join(demoDir, webm) !== finalPath) fs.renameSync(path.join(demoDir, webm), finalPath);
    const kb = (fs.statSync(finalPath).size / 1024).toFixed(0);
    console.log(`6. 완료: ${finalPath} (${kb} KB)`);
  } else {
    console.log('6. [!] .webm 이 생성되지 않았습니다.');
  }
})();

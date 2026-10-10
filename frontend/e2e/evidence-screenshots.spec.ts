// e2e/evidence-screenshots.spec.ts — captures the final-state evidence
// screenshots into e2e/evidence/ (committed to the repo for the phase report).
// Run: npx playwright test e2e/evidence-screenshots.spec.ts
import { test } from "@playwright/test";
import fs from "fs";

const OUT = "e2e/evidence";
test.beforeAll(() => fs.mkdirSync(OUT, { recursive: true }));

const routes: [string, string][] = [
  ["home", "/"],
  ["simulation", "/simulation"],
  ["training", "/training"],
  ["behavior", "/behavior"],
  ["evaluation", "/evaluation"],
  ["experiments", "/experiments"],
  ["agents", "/agents"],
  ["network", "/network"],
];

for (const [name, route] of routes) {
  test(`screenshot ${name}`, async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto(route, { waitUntil: "networkidle" });
    if (route === "/simulation") {
      // Drive a real episode so the capture shows live state, not idle config.
      await page.getByRole("button", { name: "START" }).click();
      await page.waitForTimeout(4000);
    }
    await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
  });
}

test("screenshot simulation-mobile-375px", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/simulation", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "START" }).click();
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${OUT}/simulation-mobile.png`, fullPage: true });
});

// e2e/simulation.spec.ts — Live Battle acceptance against the REAL backend.
// Requires: backend on :8000 (started by CI/runner) and frontend on :3000.
import { test, expect } from "@playwright/test";

test.describe("live simulation", () => {
  test("backend is reachable before running live tests", async ({ request }) => {
    const resp = await request.get("http://127.0.0.1:8000/health");
    expect(resp.ok()).toBeTruthy();
  });

  test("start streams real state, events, and network from the backend", async ({ page }) => {
    await page.goto("/simulation");
    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();

    // Config loaded from the backend registry
    await expect(page.locator("select").first()).toContainText("V1 Enterprise");

    // START → episode_start → live state
    await page.getByRole("button", { name: "START" }).click();

    // The header must switch to LIVE within a few seconds
    await expect(page.getByText("LIVE", { exact: true })).toBeVisible({ timeout: 10_000 });

    // Real network snapshot arrives: 8 nodes in the topology SVG
    const nodeRects = page.locator("svg[aria-label='Live network topology'] g");
    await expect(nodeRects.first()).toBeVisible({ timeout: 10_000 });

    // Step counter advances beyond 0 (auto-play)
    await expect
      .poll(async () => {
        const txt = await page.locator("span.badge-brutal.mono").filter({ hasText: /STEP \d+/ }).first().innerText();
        const m = txt.match(/STEP (\d+)/);
        return m ? Number(m[1]) : 0;
      }, { timeout: 15_000, intervals: [500] })
      .toBeGreaterThan(0);

    // PAUSE halts auto-play
    await page.getByRole("button", { name: "PAUSE" }).click();
    await expect(page.getByText("PAUSED", { exact: true })).toBeVisible();
    const stepAtPause = Number(
      (
        await page
          .locator("span.badge-brutal.mono")
          .filter({ hasText: /STEP \d+/ })
          .first()
          .innerText()
      ).match(/STEP (\d+)/)![1],
    );
    await page.waitForTimeout(1500);
    const stepAfterWait = Number(
      (
        await page
          .locator("span.badge-brutal.mono")
          .filter({ hasText: /STEP \d+/ })
          .first()
          .innerText()
      ).match(/STEP (\d+)/)![1],
    );
    expect(stepAfterWait).toBe(stepAtPause);
  });

  test("STEP advances exactly one transition and RESET restores initial state", async ({ page }) => {
    await page.goto("/simulation");
    await page.getByRole("button", { name: "START" }).click();
    await expect(page.getByText("LIVE", { exact: true })).toBeVisible({ timeout: 10_000 });

    // Pause first, then verify discrete stepping
    await page.getByRole("button", { name: "PAUSE" }).click();
    await expect(page.getByText("PAUSED", { exact: true })).toBeVisible();

    const readStep = async () =>
      Number(
        (
          await page
            .locator("span.badge-brutal.mono")
            .filter({ hasText: /STEP \d+/ })
            .first()
            .innerText()
        ).match(/STEP (\d+)/)![1],
      );

    const before = await readStep();
    await page.getByRole("button", { name: "STEP", exact: true }).click();
    await expect
      .poll(readStep, { timeout: 10_000, intervals: [200] })
      .toBe(before + 1);

    // RESET returns to step 0, zero rewards, cleared events
    await page.getByRole("button", { name: "RESET" }).click();
    await expect(page.getByText("PAUSED", { exact: true })).toBeVisible();
    await expect
      .poll(readStep, { timeout: 10_000, intervals: [200] })
      .toBe(0);
    await expect(page.getByText("No events yet — press STEP or RESUME.")).toBeVisible();
  });

  test("agent panels show real reward numbers (not placeholder values)", async ({ page }) => {
    await page.goto("/simulation");
    await page.getByRole("button", { name: "START" }).click();
    await expect(page.getByText("LIVE", { exact: true })).toBeVisible({ timeout: 10_000 });
    await page.waitForTimeout(2500);

    // The old mock showed static "+21.4"; real backend values never equal that pair.
    const body = await page.locator("body").innerText();
    expect(body).not.toContain("+21.4");
    // Red panel must show a numeric reward readout (e.g. "Reward\n-14.50")
    const redPanel = page.locator("div", { has: page.getByText("🔴 RED — ATTACKER") }).filter({ has: page.getByText("Reward", { exact: true }) });
    await expect(redPanel.getByText(/^-?\d+\.\d{2}$/).first()).toBeVisible();
    const bluePanel = page.locator("div", { has: page.getByText("🔵 BLUE — DEFENDER") }).filter({ has: page.getByText("Reward", { exact: true }) });
    await expect(bluePanel.getByText(/^-?\d+\.\d{2}$/).first()).toBeVisible();
  });
});

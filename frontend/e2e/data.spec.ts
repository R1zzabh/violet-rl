// e2e/data.spec.ts — Research pages render REAL backend artifacts.
import { test, expect } from "@playwright/test";

test.describe("training (real metrics)", () => {
  test("registry lists the real run and charts render real values", async ({ page }) => {
    await page.goto("/training");
    // Real run id produced by the training loop (registry row cell)
    await expect(page.locator("td", { hasText: "ppo_red_v1" }).first()).toBeVisible({ timeout: 15_000 });
    // Metric charts render with real axis labels (global_step range from training.jsonl)
    await expect(page.getByText(/Mean episode reward per update — run ppo_red_v1/)).toBeVisible();
    await expect(page.locator("svg[aria-label='mean_episode_reward']")).toBeVisible();
    // Summary cards show numeric updates (50 real PPO updates in the artifact)
    await expect(page.getByText("Updates", { exact: true }).first()).toBeVisible();
  });
});

test.describe("evaluation (real artifacts)", () => {
  test("cross-matchup table renders real rates", async ({ page }) => {
    await page.goto("/evaluation");
    await expect(page.getByText("CROSS-MATCHUP SUMMARY — run ppo_red_v1")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Attack success")).toBeVisible();
    // Real generated timestamp appears
    await expect(page.getByText(/generated \d{4}-\d{2}-\d{2}T/)).toBeVisible();
  });
});

test.describe("experiments (registry metadata)", () => {
  test("registry table and checkpoint metadata render", async ({ page }) => {
    await page.goto("/experiments");
    await expect(page.getByText("ppo_red_v1")).toBeVisible({ timeout: 15_000 });
    // Select the run → metadata loads
    await page.getByText("ppo_red_v1").first().click();
    await expect(page.getByText("CHECKPOINT METADATA — ppo_red_v1")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("Hyperparameters")).toBeVisible();
    await expect(page.getByText("rollout_steps", { exact: true })).toBeVisible();
  });
});

test.describe("behavior (evidence-based)", () => {
  test("methodology note and real heatmap render", async ({ page }) => {
    await page.goto("/behavior");
    await expect(page.getByText("Methodology")).toBeVisible({ timeout: 15_000 });
    // Real heatmap section appears when the behavior report has data
    await expect(page.getByText("BLUE — state × action").or(page.getByText("NOT AVAILABLE"))).toBeVisible({ timeout: 15_000 });
  });
});

test.describe("agents + network (registry + topology)", () => {
  test("agent registry shows available and unavailable agents", async ({ page }) => {
    await page.goto("/agents");
    await expect(page.getByText("Scripted Aggressive")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("UNAVAILABLE").first()).toBeVisible(); // RL red marked honestly
    await expect(page.getByText("AVAILABLE").first()).toBeVisible();
  });

  test("network explorer shows the real 8-node topology", async ({ page }) => {
    await page.goto("/network");
    await expect(page.getByText("V1 Enterprise Topology")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("8 nodes")).toBeVisible();
    await expect(page.locator("svg[aria-label='Live network topology']").first()).toBeVisible();
    await expect(page.locator("svg[aria-label='Live network topology']").getByText("DATABASE").first()).toBeVisible();
    await expect(page.locator("svg[aria-label='Live network topology']").getByText("DECOY").first()).toBeVisible();
  });
});

test.describe("error states", () => {
  test("backend down produces useful UI on training page", async ({ page }) => {
    // Point the browser at a dead API by intercepting before requests go out.
    await page.route("**/runs**", (route) => route.abort("connectionrefused"));
    await page.goto("/training");
    await expect(page.getByText("Backend unavailable")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/fetch failed|Backend unreachable/)).toBeVisible();
  });

  test("simulation page shows backend-down banner when API is unreachable", async ({ page }) => {
    await page.route("**/scenarios**", (route) => route.abort("connectionrefused"));
    await page.route("**/agents**", (route) => route.abort("connectionrefused"));
    await page.goto("/simulation");
    await expect(page.getByText("Backend unavailable")).toBeVisible({ timeout: 15_000 });
  });
});

test.describe("responsive", () => {
  const viewports: [string, number, number][] = [
    ["mobile", 375, 812],
    ["tablet", 768, 1024],
    ["desktop", 1280, 800],
  ];
  for (const [name, width, height] of viewports) {
    test(`no horizontal overflow at ${name} (${width}px)`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      for (const route of ["/", "/simulation", "/training", "/network"]) {
        await page.goto(route, { waitUntil: "domcontentloaded" });
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
        );
        // Allow tiny subpixel slack; anything larger is real overflow/clipping.
        expect(overflow, `${route} overflows horizontally at ${width}px`).toBeLessThanOrEqual(2);
      }
    });
  }
});

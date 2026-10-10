// e2e/smoke.spec.ts — Homepage + all routes render without fatal errors.
import { test, expect } from "@playwright/test";

test("homepage renders with hero and nav", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Multi-Agent RL");
  await expect(page.locator("nav")).toBeVisible();
});

test("all routes open without page errors", async ({ page }) => {
  const routes = ["/simulation", "/training", "/behavior", "/evaluation", "/experiments", "/agents", "/network"];
  const pageErrors: string[] = [];
  page.on("pageerror", (e) => pageErrors.push(e.message));

  for (const route of routes) {
    await page.goto(route, { waitUntil: "domcontentloaded" });
    await expect(page.locator("h1")).toBeVisible();
  }
  expect(pageErrors).toEqual([]);
});

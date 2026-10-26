import { expect, test } from "@playwright/test";

const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

test("SQLi live run exposes measured charts and a replay that resumes after refresh", async ({
  page,
}) => {
  const response = await page.request.get(`${apiBase}/range/training-runs?limit=100`);
  expect(response.ok()).toBeTruthy();
  const { items } = await response.json();
  const run = items.find(
    (item: { status: string; config: { scenario_id?: string } }) =>
      item.status === "completed" && item.config.scenario_id === "sqli_lab_v1",
  );
  if (!run) throw new Error("a completed SQLi PPO run is required for this replay check");

  await page.goto("/sqli");
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "SQL injection learning lab",
  );
  await expect(page.getByText("sqli_target_v1").first()).toBeVisible();

  await page.goto(`/sqli/${run.training_run_id}`);
  await expect(page.getByRole("heading", { name: "Guided event replay" })).toBeVisible();
  const firstStep = page.getByText(/Step 1 \/ \d+/);
  await expect(firstStep).toBeVisible();
  const totalSteps = Number((await firstStep.textContent())?.match(/\/ (\d+)/)?.[1] ?? 0);
  expect(totalSteps).toBeGreaterThan(1);
  await expect(page.getByRole("heading", { name: "Per-episode returns" })).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Verified security and availability" }),
  ).toBeVisible();
  await expect(page.getByText("Same observation, before and after update")).toBeVisible();
  await expect(
    page.getByText("Both vectors use the identical stored reference observation.").first(),
  ).toBeVisible();

  await page.getByRole("button", { name: "Pause replay" }).click();
  await page.getByRole("button", { name: /Advance/ }).click();
  await expect(page.getByText(`Step 2 / ${totalSteps}`)).toBeVisible();
  await page.reload();
  await expect(page.getByText(`Step 1 / ${totalSteps}`)).toBeVisible();
  await page.getByRole("button", { name: /Advance/ }).click();
  await expect(page.getByText(`Step 2 / ${totalSteps}`)).toBeVisible();
});

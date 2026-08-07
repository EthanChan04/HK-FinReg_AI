import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";

const layout = fs.readFileSync("src/app/layout.tsx", "utf8");
const config = fs.readFileSync("next.config.ts", "utf8");
const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));

test("production build does not depend on remote Google fonts", () => {
  assert.doesNotMatch(layout, /next\/font\/google/);
});

test("Next.js config pins the frontend tracing root", () => {
  assert.match(config, /turbopack\s*:\s*\{[\s\S]*root\s*:\s*process\.cwd\(\)/);
});

test("production build has a native-binding-independent fallback", () => {
  assert.match(packageJson.scripts.build, /--webpack/);
});

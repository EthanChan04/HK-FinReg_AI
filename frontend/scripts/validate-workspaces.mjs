import fs from "node:fs";
import path from "node:path";

const sourcePath = path.join(process.cwd(), "src", "lib", "bankWorkspaces.ts");
const source = fs.readFileSync(sourcePath, "utf8");

const requiredBoardIds = [
  "customer-account",
  "transaction-payment",
  "product-launch",
  "regulatory-research",
  "human-review",
  "knowledge-base",
];

const requiredEngineModes = ["rag", "rag_kag", "deepresearch", "human_review"];

const errors = [];

for (const boardId of requiredBoardIds) {
  if (!source.includes(`id: "${boardId}"`)) {
    errors.push(`Missing board id: ${boardId}`);
  }
}

for (const engineMode of requiredEngineModes) {
  if (!source.includes(`engineMode: "${engineMode}"`)) {
    errors.push(`Missing engine mode: ${engineMode}`);
  }
}

for (const field of ["nameZh", "description", "primaryUsers", "defaultInput", "endpoint"]) {
  if (!source.includes(field)) {
    errors.push(`Missing required workflow field in source: ${field}`);
  }
}

if (errors.length > 0) {
  console.error(errors.join("\n"));
  process.exit(1);
}

console.log("Workspace configuration validation passed.");

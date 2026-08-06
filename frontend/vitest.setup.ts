// Vitest setup: 注册 jest-dom 自定义 matchers（toBeInTheDocument 等）
// 未启用 vitest globals，因此显式注册 RTL cleanup（每个用例后卸载组件）
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup());

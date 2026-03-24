"""
HK-FinReg AI — 后端 SSE 流式接口自动化测试脚本
使用 httpx 向 FastAPI 流式端点发送请求，并在终端以打字机效果实时渲染输出。

用法:
    1. 先在另一个终端启动后端:  uvicorn app.main:app --reload --port 8000
    2. 运行本脚本:  python auto_test.py
"""
import httpx
import json
import sys
import time

# ============ 配置区 ============
BASE_URL = "http://127.0.0.1:8000"
STREAM_ENDPOINT = "/api/v1/svf/analyze/stream"

TEST_PAYLOAD = {
    "application_data": (
        "SVF Compliance Inquiry:\n"
        "Company: FastPay HK Limited\n"
        "License Type: SVF\n"
        "Service: Stored value facility for retail payments\n"
        "Transaction Volume: HKD 50 million monthly\n"
        "AML Officer: Appointed\n"
        "KYC Procedure: eKYC with facial recognition\n"
        "Suspicious Transaction Reports: 2 filed in past year"
    ),
    "stream_agents_state": True
}

# ============ 颜色工具 ============
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
BOLD    = "\033[1m"
RESET   = "\033[0m"


def print_header():
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  HK-FinReg AI — SSE Stream Auto Test")
    print(f"  Target: {BASE_URL}{STREAM_ENDPOINT}")
    print(f"{'=' * 60}{RESET}\n")


def main():
    print_header()

    # Step 1: 健康检查
    print(f"{YELLOW}[1/3] 正在检查后端健康状态...{RESET}")
    health_url = f"{BASE_URL}/api/v1/health"
    try:
        resp = httpx.get(health_url, timeout=30.0, follow_redirects=True, trust_env=False)
        if resp.status_code != 200:
            print(f"  {RED}❌ 健康检查失败: HTTP {resp.status_code}{RESET}")
            print(f"     URL: {health_url}")
            print(f"     Response: {resp.text[:300]}")
            sys.exit(1)
        health = resp.json()
        print(f"  ✅ 后端状态: {health.get('status')}")
        engines = health.get("engines", {})
        for k, v in engines.items():
            color = GREEN if v not in ("missing", "disabled") else RED
            print(f"     {k}: {color}{v}{RESET}")
        print()
    except httpx.ConnectError:
        print(f"  {RED}❌ 无法连接到 {BASE_URL}，请先启动后端：{RESET}")
        print(f"     cd backend && uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    # Step 2: 发送 SSE 流式请求
    print(f"{YELLOW}[2/3] 正在发送 SSE 流式请求...{RESET}\n")
    start_time = time.time()
    agent_count = 0
    token_chars = 0

    try:
        with httpx.stream(
            "POST",
            f"{BASE_URL}{STREAM_ENDPOINT}",
            json=TEST_PAYLOAD,
            timeout=120.0,
            follow_redirects=True,
            trust_env=False
        ) as response:

            if response.status_code != 200:
                print(f"{RED}❌ HTTP {response.status_code}{RESET}")
                sys.exit(1)

            event_type = ""
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()

                elif line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_type == "agent_state":
                        agent_count += 1
                        agent = data.get("agent", "?")
                        msg = data.get("message", "")
                        print(f"  {MAGENTA}🤖 [{agent}]{RESET} {msg}")

                    elif event_type == "token":
                        text = data.get("text", "")
                        # 打字机效果：逐字符输出
                        for ch in text:
                            sys.stdout.write(ch)
                            sys.stdout.flush()
                            time.sleep(0.003)  # 3ms per char
                        token_chars += len(text)

                    elif event_type == "done":
                        pass  # 结束标记

                    event_type = ""

    except httpx.ReadTimeout:
        print(f"\n{RED}❌ 请求超时 (>120s)，可能是 LLM 响应过慢{RESET}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Step 3: 汇总结果
    print(f"\n\n{YELLOW}[3/3] 测试结果汇总{RESET}")
    print(f"  {'─' * 40}")
    print(f"  Agent 节点数:    {GREEN}{agent_count}{RESET}")
    print(f"  输出字符数:      {GREEN}{token_chars}{RESET}")
    print(f"  端到端耗时:      {GREEN}{elapsed:.2f}s{RESET}")
    print(f"  {'─' * 40}")

    # LangSmith 提醒
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  🔭 LangSmith Trace 检查提醒")
    print(f"{'=' * 60}{RESET}")
    print(f"""
  {BOLD}{GREEN}请立即前往 LangSmith 面板查看本次请求的完整 Trace 树：{RESET}

  {BOLD}https://smith.langchain.com{RESET}

  在左侧导航栏中找到项目 {CYAN}hk-finreg-ai{RESET}，
  点击最新一条 Trace，您应该能看到：
    ✦ 每个 Agent Node (Extractor → Retriever → Analyzer → Reviewer)
      的 Input/Output 详情
    ✦ Conditional Edge 的路由判断 (APPROVED / REJECTED)
    ✦ 每次 LLM .invoke() 的 Prompt 与 Completion 全文
    ✦ Token 用量统计与延迟分布

  {YELLOW}如果 Trace 列表为空，请检查 backend/.env 中的
  LANGSMITH_API_KEY 是否正确填入。{RESET}
""")


if __name__ == "__main__":
    main()

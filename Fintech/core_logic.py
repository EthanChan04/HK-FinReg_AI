"""
核心业务逻辑模块 (Core Logic)

本模块包含系统底层的主要业务代码。涵盖：
1. 数据隐私与脱敏处理 (Privacy Shield)
2. 基础 LLM 代理与 RAG 代理的初始化构建
3. 对接流式文本/结构化返回结果的规范化处理 (Formatters)
4. 多种合规审查场景的业务封装：
    - SVF 合规审查 (基于 RAG 的文档检索增强生成)
    - 银行开户资格审查 (纯 LLM)
    - 跨境汇款风险评估 (纯 LLM)
    - SME 企业贷款信用评估 (纯 LLM)
"""
import os
import re
import time
import streamlit as st 
from dotenv import load_dotenv
from typing import Tuple, Optional, Union, TypedDict, Annotated
from datetime import datetime

# LangGraph Multi-Agent 架构支持
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import json

# 引入 LangChain 框架所需依赖
from langchain_community.document_loaders import PyPDFLoader        # PDF 提取
from langchain_text_splitters import CharacterTextSplitter         # 文本字符拆分
from langchain_community.chat_models import ChatTongyi             # 通义千问大语言模型实现 (遗留)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings        # OpenAI 格式兼容大语言模型实现 (用于 LongCat & Zhipu)
from langchain_community.vectorstores import Chroma                # 向量存储引擎 ChromaDB
from langchain_classic.chains import RetrievalQA                     # RAG 检索生成问答链 (LangChain 1.x 兼容)
from langchain_core.messages import AIMessage                         # LangChain AI 返回的消息模型规范 (LangChain 1.x 兼容)

from performance_monitor import get_tracker

# 载入环境变量配置（如 API Key 等）
load_dotenv()

# 启动核心校验：确保 DashScope API 密钥已被正确配置，否则打印红色报错并阻止进一步的初始化
if not os.getenv("DASHSCOPE_API_KEY"):
    print("\033[91mCRITICAL ERROR: DASHSCOPE_API_KEY not found! Please check .env file.\033[0m")

def pii_scrubber(text: str) -> str:
    """
    隐私护盾层 (Layer 2: Privacy Shield)
    在敏感数据发送给大语言模型或第三方服务器之前，将个人可识别信息 (PII) 进行初步脱敏。
    当前支持利用正则匹配替换相关的香港身份证号(HKID)和联系电话号码。
    """
    if not text:
        return ""
    # 替换香港身份证号 (e.g., A123456(7) -> [HKID REDACTED])
    text = re.sub(r'[A-Z]\d{6}\([0-9A]\)', '[HKID REDACTED]', text)
    # 替换手机号 (e.g., 以5/6/9开头的8位数字)
    text = re.sub(r'\b[569]\d{7}\b', '[PHONE REDACTED]', text)
    return text

def extract_response_content(response) -> str:
    """
    辅助函数：从不同的响应数据类型中提取并合并出纯字符串结果。
    
    兼容性处理包括:
    - 纯字符串 (str): 直接返回原样
    - 字典对象 (dict): 寻找 'result' 键或 'content' 键返回
    - LangChain AIMessage: 返回其 content 属性
    - 其它任意类型对象: 尝试转化为普通字符串
    """
    if response is None:
        return ""
    
    if isinstance(response, str):
        return response
    
    if isinstance(response, dict):
        if 'result' in response:
            return str(response['result']) if response['result'] else ""
        if 'content' in response:
            return str(response['content']) if response['content'] else ""
        return str(response)
    
    if isinstance(response, AIMessage):
        return response.content if response.content else ""
    
    if hasattr(response, 'content'):
        return str(response.content) if response.content else ""
    
    return str(response)

def format_output(text: Union[str, dict, AIMessage, None]) -> str:
    """
    清洗并规范化系统的最终文本输出 (Format Output)。
    利用正则表达式去除不可见字符、异常的重复符号边界以及多余换行符，
    使得显示出来的 Markdown 更加美观整洁。
    """
    text = extract_response_content(text)
    
    if not text:
        return ""
    
    # 清除各种非打印的不可见控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 修正超出 markdown 标准的冗余水平分割线：比如将过多连续的***转回**
    text = re.sub(r'\*\*\*+', '**', text)
    text = re.sub(r'---+', '---', text)
    text = re.sub(r'===+', '===', text)
    
    # 合并连续大于等于 3 个换行符为合理的双换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 保证列表项前面板的空格对其标准 Markdown
    text = re.sub(r'^(\s*)[-*]\s*', r'\1- ', text, flags=re.MULTILINE)
    
    return text.strip()

def get_current_timestamp() -> str:
    """辅助函数：获取统一的带有 HKT (Hong Kong Time) 标识的时间戳文本"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S HKT")

@st.cache_resource
def build_rag_agent() -> object:
    """
    构建基于文档增强检索 (RAG) 的 AI 引擎。
    针对 SVF (Stored Value Facilities) 合规场景，主要进行 PDF 的读取切割、内嵌及建立向量问答。
    
    该函数已经被 Streamlit cache。这意味着向量库的加载和构建过程在整个应用生命周期中仅执行一次，避免严重的性能损耗。
    """
    print("Initializing RAG Compliance Engine (Loading PDF)...")
    
    # 本地法律法规合规条例文档位置
    pdf_path = "./AML Guideline for LCs_Eng_30 Sep 2021.pdf"
    
    # 检查 PDF 是否存在，如果不存在则提示缺失并中止 RAG agent 的加载
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return None

    try:
        # 使用 PyPDFLoader 工具加载 PDF 文本文档
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"PDF loaded successfully: {len(documents)} pages.")
    except Exception as e:
        print(f"PDF loading failed: {e}")
        return None

    # 将长篇幅的文本文档切分为最大字符数为 1500，且彼此覆盖 200 字符重叠的多个段落，以提升嵌入和检索准确性。
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Text split completed: {len(splits)} segments.")

    # 2. 将文档切片转换为向量嵌入并存入本地临时 Chroma 数据库
    print("Generating Vector Embeddings via Zhipu API...")
    if not os.getenv("ZHIPU_API_KEY"):
        print("\033[93mWarning: ZHIPU_API_KEY not found! Embedding might fail.\033[0m")
        
    embeddings = OpenAIEmbeddings(
        model="embedding-3",
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
        chunk_size=64  # 智谱 API 限制单次请求的 input 数组长度不得超过64
    )
    
    db = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        collection_name="zhipu_collection"
    )
    
    # 3. 初始化主干大模型为 Zhipu GLM-4.7-Flash
    print("Initializing General LLM Backend (Zhipu GLM-4.7-Flash)...")
    llm = ChatOpenAI(
        model_name="glm-4-flash", 
        temperature=0,
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    # 以 "stuff"（拼接式）生成 Retrieval QA 问答检索链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=db.as_retriever()
    )
    
    print("RAG Compliance Engine initialized successfully.")
    return qa_chain

@st.cache_resource
def build_llm_only_agent() -> object:
    """
    构建单纯依赖大型语言模型内置知识进行回答引擎 (LLM-only Agent)。
    用于虚拟银行、跨境汇款、中小微及 SME 等依赖常识性合规审查的业务场景，跳过了本地文档检索层。
    """
    print("Initializing LLM-only Engine (Zhipu GLM-4.7-Flash)...")
    
    # 初始化智谱 API 并绑定 langchain_openai
    if not os.getenv("ZHIPU_API_KEY"):
        print("\033[91mCRITICAL ERROR: ZHIPU_API_KEY not found!\033[0m")
        
    llm = ChatOpenAI(
        model_name="glm-4-flash", 
        temperature=0,
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    
    print("LLM-only Engine initialized successfully.")
    return llm

@st.cache_resource
def build_thinking_agent() -> object:
    """
    构建基于 LongCat-Flash-Thinking-2601 的深度思考引擎 (Thinking Agent)。
    用于处理如中小企业信贷审查、复杂跨境汇款反洗钱审查等需要深度逻辑拆解的场景。
    """
    print("Initializing Thinking Engine...")
    
    # 检查 API Key
    if not os.getenv("LONGCAT_API_KEY"):
        print("\033[91mCRITICAL ERROR: LONGCAT_API_KEY not found!\033[0m")
        
    llm = ChatOpenAI(
        model_name="LongCat-Flash-Thinking-2601",
        temperature=0,
        openai_api_key=os.getenv("LONGCAT_API_KEY"),
        openai_api_base="https://api.longcat.chat/openai/v1"
    )
    
    print("Thinking Engine initialized successfully.")
    return llm

def build_compliance_agent():
    """兼容旧版本的默认调用，用于外部对统一 RAG agent 的简化初始化"""
    return build_rag_agent()

# ==========================================
# 业务功能 1: SVF 多智能体合规审查 (Agentic Workflow + RAG)
# ==========================================
class SVFState(TypedDict, total=False):
    original_input: str
    extracted_entities: str
    retrieved_docs: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str

def generate_risk_report(agent, user_input: str) -> Tuple[str, str, float]:
    """
    通过多智能体协作(Multi-Agent System)生成储值支付工具(SVF)合规风险报告。
    流程：Extractor -> Retriever -> Analyzer -> Reviewer -> (循环修正) -> END
    """
    tracker = get_tracker()
    start_time = time.time()
    safe_input = pii_scrubber(user_input)
    
    if agent is None:
        return safe_input, "❌ RAG 引擎未初始化", time.time() - start_time
        
    # 初始化共享的大语言模型实例
    llm = ChatOpenAI(
        model_name="glm-4-flash", 
        temperature=0,
        openai_api_key=os.getenv("ZHIPU_API_KEY"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    # 取出 RAG 的原生检索器
    retriever = agent.retriever
    
    # ---------------- 智能体节点 (Agents) 定义 ----------------
    def extractor_node(state: SVFState):
        """节点 1：信息提取专员"""
        prompt = f"Extract the key transaction details (Sender, Country, Amount, Purpose) from this text and return it as a concise summary:\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"extracted_entities": resp.content}
        
    def retriever_node(state: SVFState):
        """节点 2：法务检索专员"""
        query = f"SVF compliance rules, money laundering thresholds, and CDD requirement guidelines regarding: {state.get('extracted_entities', '')}"
        docs = retriever.invoke(query)
        doc_text = "\\n\\n".join([d.page_content for d in docs])
        return {"retrieved_docs": doc_text}
        
    def analyzer_node(state: SVFState):
        """节点 3：风控审计官"""
        prompt = f'''You are a Senior Compliance Officer for a Hong Kong FinTech/SVF Company.
Review the transaction profile against HKMA guidelines and generate a professional compliance report.

CUSTOMER/TRANSACTION DATA:
{state['original_input']}

HKMA GUIDELINES EXTRACTED:
{state.get('retrieved_docs', '')}

Reviewer Feedback to Address (if any):
{state.get('reviewer_feedback', 'None')}

Generate a structured Risk Assessment Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# SVF Compliance Risk Assessment Report

**Report Date:** {get_current_timestamp()}

---

## 1. Risk Classification

| Category | Assessment |
|----------|------------|
| **Risk Level** | [Low / Medium / High] |
| **CDD Type Required** | [No CDD / Simplified CDD / Standard CDD / Enhanced CDD] |
| **Account Type** | [Device-based / Network-based] |

---

## 2. Transaction Analysis

**Transaction Amount:** [Amount and currency]

**Threshold Assessment:**
- Stored Value Limit: [Relevant threshold]
- Assessment: [Within limit / Exceeds limit]

---

## 3. Customer Profile Review

**Identity Verification:**
- Name: [Customer name or REDACTED]
- Identification: [HKID status]
- Contact: [Phone status]

**Source of Funds:** [Assessment of fund source]

**Occupation:** [Customer occupation]

---

## 4. Regulatory Reference

**HKMA Guideline Reference:**
- Chapter/Paragraph: [Specific reference]
- Key Requirements: [Relevant requirements]

---

## 5. Risk Factors Identified

[List any risk factors or concerns]

---

## 6. Recommendation

**Final Decision:** [Approve / Reject / Request Enhanced Due Diligence]

**Rationale:** [Brief explanation of the decision]

**Next Steps:** [Any required follow-up actions]

---

*This report is generated by HK-FinReg Multi-Agent System for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}
        
    def reviewer_node(state: SVFState):
        """节点 4：首席合规官 (负责红蓝对抗与驳回)"""
        draft = state.get("draft_report", "")
        prompt = f"""You are the Chief Compliance Officer. Review this draft report:
{draft}
Does it clearly cite HKMA rules and make a definitive risk classification?
If it is good, reply exactly with 'APPROVED'.
If it is lacking details or logic, reply exactly with 'REJECTED: [reason]'."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        rev_count = state.get("revision_count", 0)
        # 上限 1 次自我驳回修正，防止死循环
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}
            
    def should_continue(state: SVFState):
        """图状态分发路由条件"""
        if "final_report" in state and state["final_report"]:
            return "end"
        else:
            return "revise"

    # ---------------- 编排基于 LangGraph 的有限状态机 ----------------
    workflow = StateGraph(SVFState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("reviewer", reviewer_node)
    
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "retriever")
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "end": END,
            "revise": "analyzer"
        }
    )
    
    # 编译图形
    app_graph = workflow.compile()
    
    try:
        # 执行图系统
        initial_state = {
            "original_input": safe_input, 
            "revision_count": 0, 
            "final_report": "", 
            "reviewer_feedback": ""
        }
        final_state = app_graph.invoke(initial_state)
        
        result_text = final_state.get("final_report", "❌ 终态生成失败")
        formatted_result = format_output(result_text)
        
        processing_time = time.time() - start_time
        tracker.log_query("SVF Multi-Agent (RAG)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("SVF Multi-Agent (RAG)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"Agentic Workflow Error: {str(e)}", processing_time

# ==========================================
# 业务功能 2: 银行开户资格多智能体审查 (Agentic Workflow - 纯 LLM 模式)
# ==========================================
class VBState(TypedDict, total=False):
    original_input: str
    extracted_kyc_data: str
    cdd_assessment: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str

def check_virtual_bank_eligibility(agent, user_input: str) -> Tuple[str, str, float]:
    """
    通过多智能体协作(Multi-Agent System)检查新客户注册信息是否符合银行(Virtual Bank)开户合规限制。
    全程跳过检索层，由四个独立微观 Agent 节点构成：KYC分析 -> CDD定级 -> 审批撰写 -> CRO复核
    """
    tracker = get_tracker()
    start_time = time.time()
    
    safe_input = pii_scrubber(user_input)
    
    if agent is None:
        return safe_input, "❌ LLM 引擎未初始化", time.time() - start_time
    
    # 获取传入的基础大模型 (Zhipu GLM)    
    llm = agent
    
    # ---------------- 智能体节点 (Agents) 定义 ----------------
    def kyc_node(state: VBState):
        """节点 1：KYC 专员"""
        prompt = f"Extract critical KYC entities (Name, Identity Type, Occupation, Income Source) from this application text into a concise framework:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"extracted_kyc_data": resp.content}
        
    def cdd_node(state: VBState):
        """节点 2：CDD 尽职调查专员"""
        prompt = f"Review this KYC data:\\n{state.get('extracted_kyc_data', '')}\\nDetermine the ML/TF Risk Level (Low/Medium/High) and specify the appropriate CDD Level (Simplified/Standard/Enhanced CDD). Briefly justify."
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"cdd_assessment": resp.content}
        
    def approval_node(state: VBState):
        """节点 3：开户审批官"""
        prompt = f'''You are a Senior Compliance Officer at a Hong Kong Bank.
Assess the account opening application and generate a professional eligibility report.

APPLICATION DATA:
{state['original_input']}

KYC STRUCTURAL DATA:
{state.get('extracted_kyc_data', '')}

CDD PROFILE:
{state.get('cdd_assessment', '')}

Reviewer Feedback to Address (if any):
{state.get('reviewer_feedback', 'None')}

Generate a structured Account Opening Eligibility Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# Bank Account Opening Eligibility Report

**Report Date:** {get_current_timestamp()}

---

## 1. Application Summary

| Field | Details |
|-------|---------|
| **Applicant Name** | [Name or REDACTED] |
| **Account Type** | [Individual / Joint / Corporate] |
| **Application Status** | [Eligible / Conditional / Rejected] |

---

## 2. Identity Verification Assessment

**Document Verification:**
- Identity Document: [HKID / Passport / Other - Status]
- Address Proof: [Verified / Pending / Not Provided]
- Document Completeness: [Complete / Incomplete]

**Address Verification Status:**
- Residential Address: [Verification status]
- Document Type: [Utility bill / Bank statement / Other]
- Document Age: [Within 3 months / Older than 3 months]

---

## 3. CDD Level Assessment

**Required CDD Level:** [Simplified / Standard / Enhanced]

**Justification:**
- Risk Profile: [Low / Medium / High]
- Customer Type: [Individual / PEP / High-risk business]
- Source of Funds: [Clear / Requires verification]

---

## 4. Risk Classification

| Risk Factor | Level | Notes |
|-------------|-------|-------|
| **Overall Risk** | [Low / Medium / High] | [Brief explanation] |
| **Identity Risk** | [Low / Medium / High] | [Assessment] |
| **Financial Risk** | [Low / Medium / High] | [Assessment] |

---

## 5. Regulatory Concerns

[List any regulatory concerns or red flags, or state "No regulatory concerns identified"]

---

## 6. Recommendation

**Decision:** [Eligible / Conditional / Rejected]

**Required Actions:**
- [List any additional documents required]
- [List any verification steps needed]

**Next Steps:**
- [Specific actions for account opening process]

---

*This report is generated by HK-FinReg Multi-Agent System for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}
        
    def cro_node(state: VBState):
        """节点 4：首席风险官 (负责红蓝对抗审查逻辑自洽性)"""
        draft = state.get("draft_report", "")
        prompt = f"""You are the Chief Risk Officer (CRO). Review this draft account opening report:
{draft}
If the 'Required CDD Level' correctly matches the 'Risk Level' and the 'Decision' is logically sound, reply exactly with 'APPROVED'.
If the logic is contradictory (e.g. High Risk profile but immediately Eligible without Enhanced CDD), reply exactly with 'REJECTED: [detailed reason]'."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}

    def should_continue(state: VBState):
        """条件分发节点路由"""
        if "final_report" in state and state["final_report"]:
            return "end"
        else:
            return "revise"

    # ---------------- 编排基于 LangGraph 的有限状态机图 ----------------
    workflow = StateGraph(VBState)
    workflow.add_node("kyc_node", kyc_node)
    workflow.add_node("cdd_node", cdd_node)
    workflow.add_node("approval_node", approval_node)
    workflow.add_node("cro_node", cro_node)
    
    workflow.set_entry_point("kyc_node")
    workflow.add_edge("kyc_node", "cdd_node")
    workflow.add_edge("cdd_node", "approval_node")
    workflow.add_edge("approval_node", "cro_node")
    workflow.add_conditional_edges(
        "cro_node",
        should_continue,
        {
            "end": END,
            "revise": "approval_node"
        }
    )
    
    app_graph = workflow.compile()
    
    try:
        initial_state = {
            "original_input": safe_input, 
            "revision_count": 0, 
            "final_report": "", 
            "reviewer_feedback": ""
        }
        final_state = app_graph.invoke(initial_state)
        
        result_text = final_state.get("final_report", "❌ 终态报告生成失败")
        formatted_result = format_output(result_text)
        
        processing_time = time.time() - start_time
        tracker.log_query("Bank Onboarding Multi-Agent (LLM)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("Bank Onboarding Multi-Agent (LLM)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"Agentic Workflow Error: {str(e)}", processing_time

# ==========================================
# 业务功能 3: 跨境资金汇款多智能体合规判定 (Agentic Workflow - Thinking Model)
# ==========================================
class CBState(TypedDict, total=False):
    original_input: str
    parsed_funds: str
    sanctions_screening: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str

def assess_cross_border_transaction(agent, user_input: str) -> Tuple[str, str, float]:
    """
    通过多智能体协作(Multi-Agent System)判断单笔跨境资金记录是否触发制裁或反洗钱预警。
    调用 LongCat 的深度思考模型构建 Extractor -> Sanctions -> Investigator -> QA 的 Agentic Loop。
    """
    tracker = get_tracker()
    start_time = time.time()
    
    safe_input = pii_scrubber(user_input)
    
    # 强制获取深度思考模型实例 (LongCat)
    llm = build_thinking_agent()
    
    # ---------------- 智能体节点 (Agents) 定义 ----------------
    def extractor_node(state: CBState):
        """节点 1：信息提取专员"""
        prompt = f"Analyze this remittance log and extract the exact Sender, Beneficiary, Amount, Currency, Destination Country, and Purpose into a clean summary:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"parsed_funds": resp.content}
        
    def sanctions_node(state: CBState):
        """节点 2：制裁筛查专员"""
        prompt = f"Screen these entities and destination against global sanction frameworks (UN, OFAC SDN, EU, HK). Determine strict Confirm/Clear status:\\n{state.get('parsed_funds', '')}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"sanctions_screening": resp.content}
        
    def investigator_node(state: CBState):
        """节点 3：反洗钱调查专家"""
        prompt = f'''You are a Senior Compliance Officer specializing in Cross-Border Remittance at a Hong Kong Virtual Bank.
Assess the transaction logic based on the extracted data and sanction results.

ORIGINAL LOG: 
{state['original_input']}

PARSED DATA: 
{state.get('parsed_funds', '')}

SANCTIONS RESULT: 
{state.get('sanctions_screening', '')}

Reviewer Feedback to Address (if any): 
{state.get('reviewer_feedback', 'None')}

Generate a structured Cross-Border Transaction Risk Assessment Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# Cross-Border Remittance Risk Assessment Report

**Report Date:** {get_current_timestamp()}

---

## 1. Transaction Summary

| Field | Details |
|-------|---------|
| **Sender** | [Name or REDACTED] |
| **Beneficiary** | [Name or REDACTED] |
| **Destination Country** | [Country name] |
| **Amount** | [Amount and currency] |
| **Purpose** | [Stated purpose] |

---

## 2. Risk Score Assessment

**Overall Risk Score:** [1-10] out of 10

| Risk Category | Score | Assessment |
|---------------|-------|------------|
| **Country Risk** | [1-10] | [Low / Medium / High / Prohibited] |
| **Transaction Risk** | [1-10] | [Assessment] |
| **Customer Risk** | [1-10] | [Assessment] |

---

## 3. Sanctions Screening

**Screening Result:** [Clear / Potential Match / Confirmed Match]

**Screening Lists Checked:**
- UN Sanctions List: [Clear / Match]
- US OFAC SDN List: [Clear / Match]
- EU Sanctions List: [Clear / Match]
- Hong Kong Sanctions List: [Clear / Match]

---

## 4. AML/CFT Compliance Check

**Record Keeping Requirements:**
- Transaction >= HKD 8,000: [Yes / No]
- Records Required: [Yes / No]

**Suspicious Transaction Indicators:**
- [List any ML/TF red flags identified, or "No suspicious indicators identified"]

**Enhanced Due Diligence:**
- Required: [Yes / No]
- Reason: [If applicable]

---

## 5. Regulatory Framework Reference

**Applicable Regulations:**
- HKMA AML/CFT Guidelines
- FATF Recommendation 16 (Wire Transfers)
- FATF Recommendation 13 (Correspondent Banking)

---

## 6. Recommendation

**Decision:** [Approve / Approve with Monitoring / Hold for Review / Reject]

**Conditions (if applicable):**
- [List any conditions or monitoring requirements]

**Next Steps:**
- [Specific actions for transaction processing]

---

*This report is generated by HK-FinReg Multi-Agent System (LongCat Engine) for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}
        
    def qa_node(state: CBState):
        """节点 4：合规总监 (QA Red Team)"""
        draft = state.get("draft_report", "")
        prompt = f"""You are the Compliance Director. Review this Draft Remittance Risk Report:
{draft}
If the Sanctions Screening indicates a 'Match' or 'Potential Match', but the Final Decision is 'Approve', it is a FATAL logic error.
Reply exactly with 'APPROVED' if the decision is logically safe and matches the risk indicators.
Reply exactly with 'REJECTED: [detailed reason]' if contradictory or lacking conditions."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}
            
    def should_continue(state: CBState):
        if "final_report" in state and state["final_report"]:
            return "end"
        else:
            return "revise"

    # ---------------- 编排基于 LangGraph 的有限状态机图 ----------------
    workflow = StateGraph(CBState)
    workflow.add_node("extractor_node", extractor_node)
    workflow.add_node("sanctions_node", sanctions_node)
    workflow.add_node("investigator_node", investigator_node)
    workflow.add_node("qa_node", qa_node)
    
    workflow.set_entry_point("extractor_node")
    workflow.add_edge("extractor_node", "sanctions_node")
    workflow.add_edge("sanctions_node", "investigator_node")
    workflow.add_edge("investigator_node", "qa_node")
    workflow.add_conditional_edges(
        "qa_node",
        should_continue,
        {
            "end": END,
            "revise": "investigator_node"
        }
    )
    
    app_graph = workflow.compile()
    
    try:
        initial_state = {
            "original_input": safe_input, 
            "revision_count": 0, 
            "final_report": "", 
            "reviewer_feedback": ""
        }
        final_state = app_graph.invoke(initial_state)
        
        result_text = final_state.get("final_report", "❌ 终态生成失败")
        formatted_result = format_output(result_text)
        
        processing_time = time.time() - start_time
        tracker.log_query("Cross-Border Multi-Agent (LLM)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("Cross-Border Multi-Agent (LLM)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"Agentic Workflow Error: {str(e)}", processing_time

# ==========================================
# 业务功能 4: 中小企业信贷多智能体融资审查 (Agentic Workflow - Thinking Model)
# ==========================================
class SMEState(TypedDict, total=False):
    original_input: str
    parsed_financials: str
    risk_analysis: str
    draft_report: str
    reviewer_feedback: str
    revision_count: int
    final_report: str

def assess_sme_credit(agent, user_input: str) -> Tuple[str, str, float]:
    """
    负责对中小企业递交的额度审批、资金流转等业务背景，进行中小企业商户借贷层面的风险评价估算。
    构建 Data Processor -> Financial Analyst -> Credit Officer -> Credit Committee 的多智能体流水线。
    """
    tracker = get_tracker()
    start_time = time.time()
    
    # 数据的 PII 脱敏过滤
    safe_input = pii_scrubber(user_input)
    
    # 获取思考引擎
    llm = build_thinking_agent()

    # ---------------- 智能体节点 (Agents) 定义 ----------------
    def data_node(state: SMEState):
        """节点 1：数据整备专员"""
        prompt = f"Extract Company Name, Business Type, Operating Years, Revenue, Margins, and requested Amount into a structured format:\\n{state['original_input']}"
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"parsed_financials": resp.content}

    def analyst_node(state: SMEState):
        """节点 2：财务分析师"""
        prompt = f"Analyze this financial data:\\n{state.get('parsed_financials', '')}\\nDetermine the Business Viability (Stable/Volatile) and highlight explicitly any severe Industry or Currency Risks. Provide a short, rigorous quantitative logic chain."
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"risk_analysis": resp.content}

    def officer_node(state: SMEState):
        """节点 3：信贷审批官"""
        prompt = f'''You are a Senior Credit Risk Analyst at a Hong Kong Virtual Bank specializing in SME Lending.
Draft the SME Credit Assessment Report based on the application data and the financial analyst's evaluation.

APPLICATION DATA:
{state['original_input']}

FINANCIAL ANALYSIS:
{state.get('risk_analysis', '')}

Committee Feedback to Address (if any):
{state.get('reviewer_feedback', 'None')}

Generate a structured SME Credit Assessment Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# SME Credit Assessment Report

**Report Date:** {get_current_timestamp()}

---

## 1. Application Summary

| Field | Details |
|-------|---------|
| **Company Name** | [Company name] |
| **Business Type** | [Industry/Business type] |
| **Operating Years** | [Number of years] |
| **Loan Amount Requested** | [Amount and currency] |
| **Loan Purpose** | [Stated purpose] |

---

## 2. Credit Rating

**Final Credit Rating:** [A / B / C / D / E]

| Rating Scale | Description |
|--------------|-------------|
| A | Excellent - Low Risk |
| B | Good - Moderate Risk |
| C | Average - Acceptable Risk |
| D | Below Average - Elevated Risk |
| E | High Risk - Not Recommended |

---

## 3. Business Viability Assessment

| Factor | Assessment | Details |
|--------|------------|---------|
| **Revenue Stability** | [Stable / Moderate / Volatile] | [Analysis] |
| **Profit Margin Trend** | [Growing / Stable / Declining] | [Analysis] |
| **Operating History** | [Strong / Adequate / Limited] | [Analysis] |

**Platform Performance (if e-commerce):**
- Primary Platforms: [List platforms]
- Sales Volume Trend: [Growing / Stable / Declining]
- Customer Ratings: [Score if available]

---

## 4. Risk Factors Analysis

| Risk Factor | Level | Mitigation |
|-------------|-------|------------|
| **Industry Risk** | [Low / Medium / High] | [Mitigation strategy] |
| **Currency Exposure** | [Low / Medium / High] | [Mitigation strategy] |
| **Supply Chain Risk** | [Low / Medium / High] | [Mitigation strategy] |
| **Key Person Risk** | [Low / Medium / High] | [Mitigation strategy] |

---

## 5. Financial Analysis

**Key Financial Indicators:**
- Annual Revenue: [Amount]
- Net Profit Margin: [Percentage]
- Debt Service Coverage Ratio (Est.): [Ratio]
- Working Capital Assessment: [Adequate / Tight / Insufficient]

**Growth Sustainability:**
- Assessment: [Sustainable / Moderate / Concerning]
- Key Drivers: [List key growth drivers]

---

## 6. Credit Recommendation

**Decision:** [Approved / Conditional Approval / Declined]

**Approved Amount:** [Amount or N/A]

**Terms (if approved):**
- Interest Rate Range: [Rate range]
- Loan Tenor: [Period]
- Repayment Schedule: [Schedule type]

**Conditions (if conditional):**
- [List conditions to be met]

**Reasons (if declined):**
- [List reasons for decline]

---

## 7. Risk-Adjusted Pricing

**Recommended Interest Rate:** [Base rate + Risk premium]

**Additional Requirements:**
- Guarantor: [Required / Not Required]
- Collateral: [Required / Not Required]
- Insurance: [Required / Not Required]

---

*This report is generated by HK-FinReg Multi-Agent System (LongCat Engine) for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.'''
        resp = llm.invoke([HumanMessage(content=prompt)])
        return {"draft_report": resp.content}

    def committee_node(state: SMEState):
        """节点 4：风控最终委员会 (QA)"""
        draft = state.get("draft_report", "")
        prompt = f"""You are the Credit Committee Chair. Review this Draft Credit Report:
{draft}
If the 'Final Credit Rating' (e.g. A/B) logically contradicts significant risks highlighted in the 'Risk Factors Analysis' or 'Financial Analysis' (e.g. Volatile Revenue + High Industry Risk), it is a red flag.
Reply exactly with 'APPROVED' if the Rating accurately reflects the stated risks.
Reply exactly with 'REJECTED: [detailed reason]' if the rating is too lenient given the highlighted risks."""
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        
        rev_count = state.get("revision_count", 0)
        if content.startswith("APPROVED") or rev_count >= 1:
            return {"final_report": draft, "revision_count": rev_count}
        else:
            return {"reviewer_feedback": content, "revision_count": rev_count + 1}
            
    def should_continue(state: SMEState):
        if "final_report" in state and state["final_report"]:
            return "end"
        else:
            return "revise"

    # ---------------- 编排基于 LangGraph 的有限状态机图 ----------------
    workflow = StateGraph(SMEState)
    workflow.add_node("data_node", data_node)
    workflow.add_node("analyst_node", analyst_node)
    workflow.add_node("officer_node", officer_node)
    workflow.add_node("committee_node", committee_node)
    
    workflow.set_entry_point("data_node")
    workflow.add_edge("data_node", "analyst_node")
    workflow.add_edge("analyst_node", "officer_node")
    workflow.add_edge("officer_node", "committee_node")
    workflow.add_conditional_edges(
        "committee_node",
        should_continue,
        {
            "end": END,
            "revise": "officer_node"
        }
    )
    
    app_graph = workflow.compile()
    
    try:
        initial_state = {
            "original_input": safe_input, 
            "revision_count": 0, 
            "final_report": "", 
            "reviewer_feedback": ""
        }
        final_state = app_graph.invoke(initial_state)
        
        result_text = final_state.get("final_report", "❌ 终态生成失败")
        formatted_result = format_output(result_text)
        
        processing_time = time.time() - start_time
        tracker.log_query("SME Credit Multi-Agent (LLM)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("SME Credit Multi-Agent (LLM)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"Agentic Workflow Error: {str(e)}", processing_time

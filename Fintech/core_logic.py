"""
核心业务逻辑模块 (Core Logic)

本模块包含系统底层的主要业务代码。涵盖：
1. 数据隐私与脱敏处理 (Privacy Shield)
2. 基础 LLM 代理与 RAG 代理的初始化构建
3. 对接流式文本/结构化返回结果的规范化处理 (Formatters)
4. 多种合规审查场景的业务封装：
    - SVF 合规审查 (基于 RAG 的文档检索增强生成)
    - 虚拟银行开户资格审查 (纯 LLM)
    - 跨境汇款风险评估 (纯 LLM)
    - SME 企业贷款信用评估 (纯 LLM)
"""
import os
import re
import time
import streamlit as st 
from dotenv import load_dotenv
from typing import Tuple, Optional, Union
from datetime import datetime

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
# 业务功能 1: SVF 合规及交易审查 (依赖 RAG 模式)
# ==========================================
def generate_risk_report(agent, user_input: str) -> Tuple[str, str, float]:
    """
    根据给定的交易数据生成关于储值支付工具用户(SVF)合规的评审风险报告。
    本流程执行时调用本地基于《有关储值支付工具的打击洗钱指南》的 RAG Agent 检索链。
    
    返回 (处理后脱敏信息, 格式化后的生成报告, 耗时秒数)
    """
    tracker = get_tracker()
    start_time = time.time()
    
    # 1. 启动脱敏处理
    safe_input = pii_scrubber(user_input)
    
    # 2. 构建发送给 AI 模型的固定指令前置（Prompt），严格规定了 Markdown 报告结构输出格式
    prompt = """You are a Senior Compliance Officer for a Hong Kong FinTech/SVF Company.

Review the following transaction/customer profile against HKMA guidelines and generate a professional compliance report.

CUSTOMER/TRANSACTION DATA:
{safe_input}

Generate a structured Risk Assessment Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# SVF Compliance Risk Assessment Report

**Report Date:** {timestamp}

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
- Stored Value Limit: [Relevant threshold, e.g., HKD 3,000 / HKD 8,000 / HKD 25,000]
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
- Chapter/Paragraph: [Specific reference from the guideline]
- Key Requirements: [Relevant regulatory requirements]

---

## 5. Risk Factors Identified

[List any risk factors or concerns, or state "No significant risk factors identified"]

---

## 6. Recommendation

**Final Decision:** [Approve / Reject / Request Enhanced Due Diligence]

**Rationale:** [Brief explanation of the decision]

**Next Steps:** [Any required follow-up actions]

---

*This report is generated by HK-FinReg AI Compliance System for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.""".format(
        safe_input=safe_input,
        timestamp=get_current_timestamp()
    )
    
    try:
        # 3. 阻塞调用大语言模型以及检索相关知识点进行推理响应
        response = agent.invoke(prompt)
        processing_time = time.time() - start_time
        
        # 4. 抽取结果并记录成功指标
        result_text = extract_response_content(response)
        formatted_result = format_output(result_text)
        
        tracker.log_query("SVF Compliance Review (RAG)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("SVF Compliance Review (RAG)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"AI Processing Error: {str(e)}", processing_time

# ==========================================
# 业务功能 2: 虚拟银行开户资格审查 (纯 LLM 模式)
# ==========================================
def check_virtual_bank_eligibility(agent, user_input: str) -> Tuple[str, str, float]:
    """
    检查提供的新客户注册信息是否符合虚拟银行(Virtual Bank)远程在线开户的合法合规限制。
    全程跳过检索层，通过 LLM 底层的监管体系通识及常识自动推演进行反馈。
    
    返回 (处理后脱敏信息, 格式化后的生成报告, 耗时秒数)
    """
    tracker = get_tracker()
    start_time = time.time()
    
    # 1. 对开户人的关键身份标识(如身份证等)行进处理
    safe_input = pii_scrubber(user_input)
    
    # 2. 组装虚拟银行开户合规专向的提示词体系
    prompt = """You are a Senior Compliance Officer at a Hong Kong Virtual Bank.

Assess the following account opening application against HKMA Guidelines for Virtual Banks and generate a professional eligibility report.

APPLICATION DATA:
{safe_input}

Generate a structured Account Opening Eligibility Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# Virtual Bank Account Opening Eligibility Report

**Report Date:** {timestamp}

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

*This report is generated by HK-FinReg AI Compliance System for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.""".format(
        safe_input=safe_input,
        timestamp=get_current_timestamp()
    )
    
    try:
        # 调用大语言模型（不带 RAG，直连 ChatTongyi 引擎调用）
        response = agent.invoke(prompt)
        processing_time = time.time() - start_time
        
        result_text = extract_response_content(response)
        formatted_result = format_output(result_text)
        
        tracker.log_query("Virtual Bank Onboarding (LLM)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("Virtual Bank Onboarding (LLM)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"AI Processing Error: {str(e)}", processing_time

# ==========================================
# 业务功能 3: 跨境资金汇款合规判定 (纯 LLM 模式)
# ==========================================
def assess_cross_border_transaction(agent, user_input: str) -> Tuple[str, str, float]:
    """
    判断单笔跨境资金拨汇记录是否可能触发特定的风险条件、受制裁人名单或者 AML/CFT 敏感预警。
    
    返回 (处理后脱敏信息, 格式化后的生成报告, 耗时秒数)
    """
    tracker = get_tracker()
    start_time = time.time()
    
    # 脱敏处理
    safe_input = pii_scrubber(user_input)
    
    # 填充提示词模版
    prompt = """You are a Senior Compliance Officer specializing in Cross-Border Remittance at a Hong Kong Virtual Bank.

Assess the following cross-border transaction against HKMA AML/CFT Guidelines and generate a professional risk assessment report.

TRANSACTION DATA:
{safe_input}

Generate a structured Cross-Border Transaction Risk Assessment Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# Cross-Border Remittance Risk Assessment Report

**Report Date:** {timestamp}

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

*This report is generated by HK-FinReg AI Compliance System for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.""".format(
        safe_input=safe_input,
        timestamp=get_current_timestamp()
    )
    
    try:
        # 1. 初始化 Thinking Agent 进行深度逻辑推理
        llm_agent = build_thinking_agent()
        # 2. 调用跨境汇款风险评估方法
        response = llm_agent.invoke(prompt)
        processing_time = time.time() - start_time
        
        result_text = extract_response_content(response)
        formatted_result = format_output(result_text)
        
        tracker.log_query("Cross-Border Remittance (LLM)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("Cross-Border Remittance (LLM)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"AI Processing Error: {str(e)}", processing_time

# ==========================================
# 业务功能 4: 中小企业信贷融资 (纯 LLM 模式)
# ==========================================
def assess_sme_credit(agent, user_input: str) -> Tuple[str, str, float]:
    """
    负责对中小企业递交的额度审批、资金流转等业务背景，进行中小企业商户借贷层面的风险评价估算。
    
    返回 (处理后脱敏信息, 格式化后的生成报告, 耗时秒数)
    """
    tracker = get_tracker()
    start_time = time.time()
    
    # 数据的 PII 脱敏过滤
    safe_input = pii_scrubber(user_input)
    
    # 构建专属的中小型企业级信贷审查 AI 指令语并格式化时间
    prompt = """You are a Senior Credit Risk Analyst at a Hong Kong Virtual Bank specializing in SME Lending.

Assess the following SME loan application against HKMA Supervisory Policy Manual and generate a professional credit assessment report.

APPLICATION DATA:
{safe_input}

Generate a structured SME Credit Assessment Report using EXACTLY the following format. Do not add any decorative characters or symbols outside the specified format.

---

# SME Credit Assessment Report

**Report Date:** {timestamp}

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

*This report is generated by HK-FinReg AI Credit Assessment System for reference purposes only.*

---

Answer in English. Use only standard Markdown formatting. Do not use special Unicode characters or decorative symbols.""".format(
        safe_input=safe_input,
        timestamp=get_current_timestamp()
    )
    
    try:
        # 1. 初始化 Thinking Agent 进行复杂的信用逻辑推断
        llm_agent = build_thinking_agent()
        # 2. 调用中小企业融资审查方法
        response = llm_agent.invoke(prompt)
        processing_time = time.time() - start_time
        
        # 文本正则清洗
        result_text = extract_response_content(response)
        formatted_result = format_output(result_text)
        
        tracker.log_query("SME Credit Assessment (LLM)", processing_time, len(user_input), "success")
        return safe_input, formatted_result, processing_time
    except Exception as e:
        processing_time = time.time() - start_time
        tracker.log_query("SME Credit Assessment (LLM)", processing_time, len(user_input), "error", str(e))
        return safe_input, f"AI Processing Error: {str(e)}", processing_time

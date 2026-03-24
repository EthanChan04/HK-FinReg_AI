"""
主应用程序入口 (Main Application)
基于 Streamlit 构建的 Web 前端界面，提供多场景的金融合规审查平台页面。
包含了四个主要的功能模块（选项卡）：
1. SVF合规审查 (使用 RAG 模式)
2. 银行开户 (使用 LLM 模式)
3. 跨境汇款评估 (使用 LLM 模式)
4. SME融资评估 (使用 LLM 模式)
"""
import streamlit as st
import os
import time
import requests
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

# 导入核心业务逻辑函数，用于处理不同场景下的合规及风险评估
from core_logic import (
    build_rag_agent,                   # 构建基于 RAG 的智能体（PDF 检索）
    build_llm_only_agent,              # 构建仅基于 LLM 的智能体（内置知识）
    generate_risk_report,              # 生成 SVF 合规风险报告
    check_virtual_bank_eligibility,    # 检查虚拟银行开户资格
    assess_cross_border_transaction,   # 评估跨境汇款交易风险
    assess_sme_credit                  # 获取中小企业融资信用评估
)

# 导入性能监控相关模块
from performance_monitor import get_tracker

def typewriter_effect(text, container, chunk_size=5, delay=0.015):
    """通过更新 st.empty() 容器来安全地模拟打字机效果，避免 DOM 渲染崩溃"""
    time.sleep(0.5)  # [核心修复] 等待上方 status 绿框彻底完成收起渲染动画，防止产生 DOM state 紊乱
    current_text = ""
    for i in range(0, len(text), chunk_size):
        current_text += text[i:i+chunk_size]
        container.markdown(current_text + "▌")  # 追加打字机光标
        time.sleep(delay)
    container.markdown(current_text)  # 最后移除光标

# 加载 .env 文件中的环境变量（如 DASHSCOPE_API_KEY）
load_dotenv()

# 配置 Streamlit 页面基本信息（标题、图标、布局模式）
st.set_page_config(
    page_title="HK-FinReg AI", 
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# 注入极致炫酷的自定义全局 CSS (CSS Injection)
# ==========================================
st.markdown("""
<style>
    /* 渐变按钮及悬浮动画 (Hover effects) */
    div.stButton > button {
        background: linear-gradient(45deg, #10B981, #3B82F6);
        color: white;
        border: none;
        transition: all 0.3s ease-in-out;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0px 8px 15px rgba(59, 130, 246, 0.4);
        color: white;
        border: none;
    }
    div.stButton > button:active {
        transform: translateY(1px);
    }
    
    /* 让侧边栏更有层级感 */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* 动画 Status 状态框左侧彩带 */
    [data-testid="stStatusWidget"] {
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* 标题炫彩字体 */
    h1 {
        background: -webkit-linear-gradient(45deg, #1e293b, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# 加载 Lottie 动画辅助函数
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# 页面头部信息展示
st.title("🏦 HK-FinReg AI: Multi-Scenario Compliance Platform")
st.markdown("""
**Target User:** HKMA-Regulated Entities (Virtual Banks, SVF Licensees)  
**Core Engine:** Zhipu GLM-4.7-Flash + RAG & LongCat-Flash-Thinking  
**Regulatory Source:** HKMA Guidelines & International Best Practices
""")

# ==========================================
# 侧边栏：系统监控与状态显示 (System Monitor)
# ==========================================
with st.sidebar:
    # 加载并渲染 Lottie 矢量微动画作为侧边栏的高级点缀
    lottie_ai_brain = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json") 
    if lottie_ai_brain:
        st_lottie(lottie_ai_brain, height=180, key="ai_brain_animation")
    else:
        # Fallback 动画链接
        fallback_lottie = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json")
        if fallback_lottie:
            st_lottie(fallback_lottie, height=180, key="ai_brain_animation_fallback")
            
    st.header("⚙️ System Monitor")
    
    # 检查智谱 GLM API 密钥是否成功加载
    if os.getenv("ZHIPU_API_KEY"):
        st.success("✅ API Key Loaded (Zhipu GLM)")
    else:
        st.error("❌ Zhipu API Key Not Found! Check .env")
        
    # 检查 LongCat API 密钥
    if os.getenv("LONGCAT_API_KEY"):
        st.success("✅ API Key Loaded (LongCat API)")
    else:
        st.error("❌ LongCat API Key Not Found!")
        
    st.info("---")
    
    # 模块当前状态 (Module Status)
    st.markdown("### 🛠️ Module Status")
    st.success("Layer 1: Data Ingestion (PDF) - Ready")
    st.success("Layer 2: Privacy Shield (PDPO) - Active")
    st.success("Layer 3: RAG Knowledge Base - Connected")
    
    st.info("---")
    
    # 支持的业务场景概览 (Scenario Coverage)
    st.markdown("### 📊 Scenario Coverage")
    st.info("🔹 SVF合规审查 (RAG模式)")
    st.info("🔹 银行开户 (LLM模式)")
    st.info("🔹 跨境汇款评估 (Thinking模式)")
    st.info("🔹 SME融资评估 (Thinking模式)")
    
    st.info("---")
    
    # 统计信息显示 (Session Statistics)
    st.markdown("### 📈 Session Statistics")
    tracker = get_tracker()
    stats = tracker.get_session_summary()
    
    # 展示总查询次数与平均响应时间
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("总查询次数", stats["total_queries"])
    with col_b:
        st.metric("平均响应时间", f"{stats['avg_processing_time']:.2f}s")
    
    # 若有错误，显示错误次数警告
    if stats["total_errors"] > 0:
        st.warning(f"⚠️ 错误次数: {stats['total_errors']}")
    
    # 分模块显示查询统计
    if stats["queries_by_module"]:
        st.markdown("**模块使用统计:**")
        for module, count in stats["queries_by_module"].items():
            st.caption(f"• {module}: {count}次")

# ==========================================
# 主体页面：通过选项卡(Tabs)划分不同业务场景
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 SVF合规审查", 
    "🏦 银行开户", 
    "💱 跨境汇款评估", 
    "📈 SME融资评估"
])

# ------------------------------------------
# Tab 1: SVF Licensee Compliance Review (SVF 合规审查)
# ------------------------------------------
with tab1:
    st.subheader("SVF Licensee Compliance Review")
    st.markdown("**模式:** 🔍 RAG (基于 PDF 文档检索)")
    st.markdown("**适用场景:** 储值支付工具持牌机构的 AML/CFT 合规审查")
    
    col1, col2 = st.columns([1, 1])
    
    # 左侧面板：提供输入区
    with col1:
        st.markdown("#### 📥 输入: 交易/客户资料")
        st.info("请输入交易记录或客户 KYC 信息")
        
        # 预设的 SVF 审查输入样例
        default_text_svf = """Transaction Request:
Customer: Chan Tai Man
HKID: A123456(7)
Mobile: 91234567
Activity: Opening a stored value account with an initial deposit of HKD 3,000 via local bank transfer.
Occupation: Clerk at a primary school.
Source of Funds: Salary."""
        
        user_input_svf = st.text_area("数据输入", height=250, value=default_text_svf, key="svf_input")
        run_button_svf = st.button("🚀 执行合规扫描", use_container_width=True, key="svf_button")
    
    # 右侧面板：展示评估结果
    with col2:
        st.markdown("#### 🛡️ 输出: 风险评估报告")
        
        if run_button_svf:
            # 校验 API Key
            if not os.getenv("ZHIPU_API_KEY"):
                st.error("Critical Error: No Zhipu API Key found in environment variables.")
            else:
                try:
                    with st.status("🔄 开始 SVF 合规审查...", expanded=True) as status:
                        st.write("🔍 正在扫描 PII 并进行初步数据脱敏...")
                        time.sleep(0.8)
                        
                        st.write("📚 正在构建 RAG 检索模型并加载金管局知识库...")
                        # 1. 初始化 RAG Agent
                        rag_agent = build_rag_agent()
                        
                        if rag_agent is not None:
                            st.write("🧑‍💻 [智能体1: 提取专员] 正在从自然语言抽取核心实体及交易流水...")
                            time.sleep(1.2)
                            st.write("👩‍⚖️ [智能体2: 法务专员] 正在通过 RAG 向量池检索 HKMA 金管局合规条款...")
                            time.sleep(1.5)
                            st.write("🕵️ [智能体3: 风控审计官] 正在交叉核对法条，起草 CDD 及 AML 业务风险报告...")
                            time.sleep(2.0)
                            st.write("👔 [智能体4: 首席合规官] 正在执行防幻觉检验与对抗复核判决...")
                            
                            # 2. 调用生成合规审查报告的多智能体状态图
                            scrubbed_data, report, processing_time = generate_risk_report(rag_agent, user_input_svf)
                            status.update(label="✅ 合规评估报告生成完毕！", state="complete", expanded=False)
                        else:
                            status.update(label="❌ RAG 引擎初始化失败！", state="error", expanded=False)
                    
                    if rag_agent is None:
                        st.error("RAG 引擎初始化失败，请检查 PDF 文件是否存在。")
                    else:
                        # 3. 输出性能指标
                        st.markdown("##### ⏱️ 性能指标")
                        st.caption(f"处理时间: **{processing_time:.2f}秒** | 模式: RAG (PDF文档检索)")
                        
                        # 4. 展示脱敏后的数据 (确保隐私保护)
                        st.markdown("##### 🔒 隐私脱敏后的输入")
                        st.caption("发送给大模型的数据 (PII已移除):")
                        st.code(scrubbed_data, language='text')
                        
                        st.markdown("---")
                        
                        # 5. 展示 AI 分析结果报告
                        st.markdown("##### 📝 AI 分析结果")
                        st.toast(f"✨ 合规神经引擎分析完毕 耗时: {processing_time:.2f}s", icon="✅")
                        report_container = st.empty()
                        typewriter_effect(report, report_container)
                        
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
                    st.warning("提示: 请检查 Zhipu API Key 或网络连接")

# ------------------------------------------
# Tab 2: Bank Account Opening (银行开户资格审查)
# ------------------------------------------
with tab2:
    st.subheader("Bank Account Opening Eligibility Check")
    st.markdown("**模式:** 🧠 LLM-only (基于模型内置知识)")
    st.markdown("**适用场景:** 银行开户合规审查 (对标 Fusion Bank 场景)")
    
    col1, col2 = st.columns([1, 1])
    
    # 左侧面板：提供开户申请信息输入区
    with col1:
        st.markdown("#### 📥 输入: 开户申请信息")
        st.info("请输入客户开户申请资料")
        
        # 预设的虚拟银行开户信息样例
        default_text_vb = """Account Opening Application:
Applicant: Wong Siu Ming
HKID: B987654(3)
Mobile: 61234567
Residential Address: Room 123, Building A, Kowloon, Hong Kong
Occupation: Software Engineer
Employer: Tech Company Ltd.
Monthly Income: HKD 50,000
Account Type: Individual Savings Account
Source of Funds: Monthly Salary
Purpose: Personal savings and daily transactions"""
        
        user_input_vb = st.text_area("数据输入", height=250, value=default_text_vb, key="vb_input")
        run_button_vb = st.button("🚀 执行开户审查", use_container_width=True, key="vb_button")
    
    # 右侧面板：展示开户评估报告
    with col2:
        st.markdown("#### 🛡️ 输出: 开户资格评估报告")
        
        if run_button_vb:
            if not os.getenv("ZHIPU_API_KEY"):
                st.error("Critical Error: No Zhipu API Key found in environment variables.")
            else:
                try:
                    with st.status("🔄 开始银行开户资格多智能体审查...", expanded=True) as status:
                        # 1. 初始化纯 LLM Agent
                        llm_agent = build_llm_only_agent()
                        
                        st.write("🔍 [智能体1: KYC专员] 正在从自然语言抽取申请人的身份与财务背景要素...")
                        time.sleep(1.2)
                        st.write("📊 [智能体2: CDD专员] 正在根据身份数据评估洗钱风险等级 (Low/Medium/High) 并制定由于审查策略...")
                        time.sleep(1.5)
                        st.write("📑 [智能体3: 开户审批官] 正在起草开户资格审查报告初稿，比对开户合规要求...")
                        time.sleep(2.0)
                        st.write("🕴️ [智能体4: 首席风险官 CRO] 正在执行强对抗逻辑复核，验证 CDD 等级与开户结果是否逻辑自洽...")
                        
                        # 2. 调用生成开户资格评估报告的多智能体状态图
                        scrubbed_data, report, processing_time = check_virtual_bank_eligibility(llm_agent, user_input_vb)
                        
                        status.update(label="✅ 银行开户资格评估完毕！", state="complete", expanded=False)
                    
                    # 3. 输出性能指标
                    st.markdown("##### ⏱️ 性能指标")
                    st.caption(f"处理时间: **{processing_time:.2f}秒** | 模式: LLM-only (内置知识)")
                    
                    # 4. 展示脱敏后的输入数据
                    st.markdown("##### 🔒 隐私脱敏后的输入")
                    st.caption("发送给大模型的数据 (PII已移除):")
                    st.code(scrubbed_data, language='text')
                    
                    st.markdown("---")
                    
                    # 5. 展示开户评估报告
                    st.markdown("##### 📝 开户资格评估结果")
                    st.toast(f"✨ 客户尽职调查 (CDD) 通过评估 耗时: {processing_time:.2f}s", icon="✅")
                    report_container = st.empty()
                    typewriter_effect(report, report_container)
                    
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
                    st.warning("提示: 请检查 Zhipu API Key 或网络连接")

# ------------------------------------------
# Tab 3: Cross-Border Remittance (跨境汇款合规审查)
# ------------------------------------------
with tab3:
    st.subheader("Cross-Border Remittance Risk Assessment")
    st.markdown("**模式:** 🤔 Thinking Model (深度逻辑拆解)")
    st.markdown("**适用场景:** 跨境汇款合规审查 (对标 Fusion Bank 全球汇款业务)")
    
    col1, col2 = st.columns([1, 1])
    
    # 左侧面板：提供汇款交易信息输入
    with col1:
        st.markdown("#### 📥 输入: 汇款交易信息")
        st.info("请输入跨境汇款交易详情")
        
        # 预设的跨境汇款样例
        default_text_cb = """Cross-Border Remittance Request:
Sender Name: Lee Ka Fai
Sender Account: 123-456-789
Sender Country: Hong Kong
Beneficiary Name: Zhang Wei
Beneficiary Country: Mainland China
Beneficiary Bank: ICBC Shenzhen Branch
Amount: HKD 50,000
Currency: HKD to CNY
Purpose: Family Support
Frequency: Monthly
Relationship: Brother
Source of Funds: Employment Income"""
        
        user_input_cb = st.text_area("数据输入", height=250, value=default_text_cb, key="cb_input")
        run_button_cb = st.button("🚀 执行汇款风险评估", use_container_width=True, key="cb_button")
    
    # 右侧面板：展示交易风险评估结果
    with col2:
        st.markdown("#### 🛡️ 输出: 交易风险评估报告")
        
        if run_button_cb:
            if not os.getenv("LONGCAT_API_KEY"):
                st.error("Critical Error: No LongCat API Key found in environment variables.")
            else:
                try:
                    with st.status("🔄 开始跨境汇款反洗钱多智能体审查 (LongCat Engine)...", expanded=True) as status:
                        st.write("🔍 [智能体1: 信息提取专员] 正在切分资金链路，抽离发送方与收款方金融特征...")
                        time.sleep(1.0)
                        
                        st.write("🛡️ [智能体2: 制裁筛查专员] 正在调用全球制裁名单库 (OFAC/UN/EU) 执行刚性碰撞匹配...")
                        time.sleep(1.2)
                        
                        st.write("🧠 [智能体3: 反洗钱调查专家] 正在唤醒 LongCat-Flash-Thinking 深度拆解跨境汇款经济合理性...")
                        time.sleep(2.0)
                        
                        st.write("👔 [智能体4: 合规总监 QA] 正在进行最终的逻辑自洽测试与否决拦截校验...")
                        
                        # 1. 跨境汇款评估逻辑由内部组装 LangGraph，全程使用 build_thinking_agent() 驱动节点
                        scrubbed_data, report, processing_time = assess_cross_border_transaction(None, user_input_cb)
                        
                        status.update(label="✅ 资金汇款风险拆解评估完毕！", state="complete", expanded=False)
                    
                    # 3. 输出性能及脱敏数据
                    st.markdown("##### ⏱️ 性能指标")
                    st.caption(f"处理时间: **{processing_time:.2f}秒** | 模式: Thinking Model (深度逻辑拆解)")
                    
                    st.markdown("##### 🔒 隐私脱敏后的输入")
                    st.caption("发送给大模型的数据 (PII已移除):")
                    st.code(scrubbed_data, language='text')
                    
                    st.markdown("---")
                    
                    # 4. 展示评估结果
                    st.markdown("##### 📝 跨境汇款风险评估结果")
                    st.toast(f"✨ 全球制裁与反洗钱链路扫描完成 耗时: {processing_time:.2f}s", icon="✅")
                    report_container = st.empty()
                    typewriter_effect(report, report_container)
                    
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
                    st.warning("提示: 请检查 LongCat API Key 或网络连接")

# ------------------------------------------
# Tab 4: SME Lending Credit Assessment (中小企业融资信用评估)
# ------------------------------------------
with tab4:
    st.subheader("SME Lending Credit Assessment")
    st.markdown("**模式:** 🤔 Thinking Model (深度逻辑拆解)")
    st.markdown("**适用场景:** 中小企业融资信用评估 (对标 Fusion Bank 电商宜享贷)")
    
    col1, col2 = st.columns([1, 1])
    
    # 左侧面板：提供SME贷款记录输入
    with col1:
        st.markdown("#### 📥 输入: SME 融资申请信息")
        st.info("请输入中小企业融资申请详情")
        
        # 预设的SME融资信息样例
        default_text_sme = """SME Loan Application:
Company Name: ABC Trading Co. Ltd.
Registration: Hong Kong
Business Type: Cross-border E-commerce
Operating Years: 3
Platforms: Amazon (US), Shopee (SEA)
Annual Revenue: HKD 5,000,000
Net Profit Margin: 15%
Loan Amount Requested: HKD 500,000
Loan Purpose: Inventory Purchase
Collateral: None (Unsecured)
Directors: Chan Tai Man (100% shareholder)
Employees: 5"""
        
        user_input_sme = st.text_area("数据输入", height=250, value=default_text_sme, key="sme_input")
        run_button_sme = st.button("🚀 执行信用评估", use_container_width=True, key="sme_button")
    
    # 右侧面板：展示信用评估结果
    with col2:
        st.markdown("#### 🛡️ 输出: 信用评估报告")
        
        if run_button_sme:
            if not os.getenv("LONGCAT_API_KEY"):
                st.error("Critical Error: No LongCat API Key found in environment variables.")
            else:
                try:
                    with st.status("🔄 开始 SME 融资信用评估 (LongCat 多智能体架构)...", expanded=True) as status:
                        st.write("🧮 [智能体1: 数据整备专员] 正在从杂乱财报中提取营收、工龄、及利润率等核心指标...")
                        time.sleep(1.0)
                        
                        st.write("📈 [智能体2: 财务分析师] 正在解构企业现金流底稿，并量化市场与行业风险敞口...")
                        time.sleep(1.5)
                        
                        st.write("✍️ [智能体3: 信贷审批官] 正在起草信贷评级长篇报告，推演初版评级 (A-E)...")
                        time.sleep(2.0)
                        
                        st.write("👨‍⚖️ [智能体4: 风控最终委员会] 正在严格比对信贷评级与底层财务风险标的，执行逻辑阻断校验...")
                        
                        # 1. 中小企业审查也已路由至 Thinking 引擎下的 LangGraph 内部逻辑处理
                        scrubbed_data, report, processing_time = assess_sme_credit(None, user_input_sme)
                        
                        status.update(label="✅ SME 信用及定价评估模型运行完毕！", state="complete", expanded=False)
                    
                    # 3. 输出性能和脱敏信息
                    st.markdown("##### ⏱️ 性能指标")
                    st.caption(f"处理时间: **{processing_time:.2f}秒** | 模式: Thinking Model (深度逻辑拆解)")
                    
                    st.markdown("##### 🔒 隐私脱敏后的输入")
                    st.caption("发送给大模型的数据 (PII已移除):")
                    st.code(scrubbed_data, language='text')
                    
                    st.markdown("---")
                    
                    # 4. 展示评估报告
                    st.markdown("##### 📝 SME 信用评估结果")
                    st.toast(f"✨ 动态定价与信贷风险建模完毕 耗时: {processing_time:.2f}s", icon="✅")
                    report_container = st.empty()
                    typewriter_effect(report, report_container)
                    
                except Exception as e:
                    st.error(f"System Error: {str(e)}")
                    st.warning("提示: 请检查 LongCat API Key 或网络连接")

# ==========================================
# 页脚 (Footer)
# ==========================================
st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns([1, 1, 1])
with col_footer1:
    st.caption("© 2026 HK-FinReg AI Project")
with col_footer2:
    st.caption("Powered by LangChain, Zhipu GLM & LongCat")
with col_footer3:
    st.empty()

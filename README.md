# HK-FinReg AI: Multi-Scenario Compliance Platform

🏦 **香港金融科技合规AI平台** - 基于大语言模型、RAG技术与深度逻辑推理模型的多场景金融合规审查系统

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![ZhipuAI](https://img.shields.io/badge/Zhipu-GLM--4.7--Flash-blue.svg)](https://open.bigmodel.cn/)
[![LongCat](https://img.shields.io/badge/LongCat-Thinking--2601-purple.svg)](https://longcat.chat/)

---

## 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [交互与视觉体验](#交互与视觉体验)
- [技术栈选型](#技术栈选型)
- [环境配置要求](#环境配置要求)
- [安装与部署步骤](#安装与部署步骤)
- [使用说明](#使用说明)
- [项目架构](#项目架构)
- [日志与监控](#日志与监控)
- [许可证信息](#许可证信息)

---

## 项目概述

HK-FinReg AI 是一个专为香港金融科技行业设计的多场景合规审查平台，采用先进的 **RAG（检索增强生成）** 架构、**GLM 大语言模型引擎** 和 **长逻辑 Thinking 深度推理模型**，为储值支付工具（SVF）持牌机构、虚拟银行等提供高度自动化且极具深度的智能审查解决方案。

### 项目背景

- **监管合规需求**：香港金融管理局（HKMA）对金融机构的AML/CFT（反洗钱/反恐怖融资）合规要求日益严格
- **人工审查瓶颈**：传统合规审查依赖人工，效率低、成本高、容易出错
- **AI技术赋能**：利用多模态大模型的长文本推理与外挂知识库检索能力，实现合规审查流程的降本增效

### 目标用户

- 🎯 香港商业银行与虚拟银行（Virtual Banks）
- 🎯 储值支付工具（SVF）持牌机构
- 🎯 跨境支付服务提供商与汇款代理
- 🎯 中小企业（SME）供应链融资服务机构

---

## 核心功能

系统由**双核异构 AI 引擎**驱动，根据业务复杂度智能漏斗分发任务：

### 1. 📋 SVF合规审查（RAG 模式）
- **功能描述**：基于 HKMA 官方文档《Guideline on AML/CFT for SVF》提供文件检索级合规审查。
- **底层引擎**：**Zhipu GLM-4.7-Flash** + `embedding-3` (Chroma本地向量库)
- **技术特点**：PDF文档分块解析 + 2048维语义检索 + 实时对账推理

### 2. 🏦 银行开户审查（LLM-only 模式）
- **功能描述**：银行开户资格自动核准，包含身份验证、KYC、以及CDD（客户尽职调查）风险分类。
- **底层引擎**：**Zhipu GLM-4.7-Flash** (内置逻辑网)
- **输出内容**：结构化的开户决断评估报告与处理流

### 3. 💱 跨境汇款风险评估（Thinking 深度拆解模式）
- **功能描述**：针对复杂的高风险汇款路径（如跨境亲属汇款、可疑交易），进行全球制裁名单筛查与 AML/CFT 合规链路推导。
- **底层引擎**：**LongCat-Flash-Thinking-2601** (深度推理引擎)
- **技术特点**：赋予大语言模型自我纠错与长逻辑链验证能力

### 4. 📈 SME融资信用评估（Thinking 深度拆解模式）
- **功能描述**：针对中小企业进出口贸易及电商流水，构建财务模型并分析信贷风险、动态定价。
- **底层引擎**：**LongCat-Flash-Thinking-2601** (深度推理引擎)
- **输出内容**：A-E 五级信用量化评级 + 风险溢价调整

---

## 交互与视觉体验

项目为终端展示注入了 Apple 级别的流式高级前端组件，打破传统数据后台的刻板印象：
- **✨ CSS Glassmorphism**：全栈全局注入现代玻璃拟态渐变按钮、悬浮悬停阻尼框与柔和高斯阴影。
- **🧠 矢量微动画 (Lottie)**：侧边栏无缝集成高性能 `streamlit-lottie` AI 核心全息动画，呈现赛博极客质感。
- **🔄 沉浸式状态机 (`st.status`)**：抛弃简陋的 Spinner，改为交互式嵌套步进进度条，全景揭秘系统思考全链路。
- **✍️ 打字机流式输出**：集成自定义 Python 生成器，将最终的合规研判报告以拟人化的打字机流式特效（Streaming）逐字输出。
- **🎉 轻量级 Toast 提示**：摒弃占用屏幕空间的色块提示框，重构为 macOS 风格的右下角 Toast 性能反馈微操交互。

---

## 技术栈选型

### 核心技术框架
| 组件 | 选型 | 用途 |
|------|------|----------|
| **核心语言** | Python 3.10+ | 主业务应用容器开发 |
| **基础框架** | LangChain Core | 规范化生成式 AI 调用图谱与消息结构 |
| **前端交互** | Streamlit 1.x | Python原生 Web UI 响应式开发框架 |
| **本地存储** | ChromaDB | 轻量级本地进程嵌入级向量数据库 |

### AI 多模型调度网 (Dual-Engine)
| 服务提供商 | 接入模型 | 职责映射 |
|------|------|------|
| **智谱大模型** (ZhipuAI) | `glm-4-flash` | 处理结构相对标准的文字生成与RAG整合输出 |
| **智谱大模型** (ZhipuAI) | `embedding-3` | 负责提取 PDF 的高维高精度知识向量嵌入 |
| **LongCat API** | `LongCat-Flash-Thinking-2601` | 承接具有深水逻辑计算、反洗钱推理与图计算需求的复杂请求 |

---

## 环境配置要求

- **操作系统**：Windows 10/11、macOS、Linux
- **依赖环境**：Python 3.10+
- **API 密钥申请**：
  - [智谱 AI 开放平台](https://open.bigmodel.cn/) (提供 OpenAI 兼容格式接入 ZHIPU_API_KEY)
  - [LongCat 开放平台](https://longcat.chat/) (获取 LONGCAT_API_KEY)

---

## 安装与部署步骤

### 1. 克隆或下载项目
```bash
git clone <repository-url>
cd Fintech
```

### 2. 创建并激活专属虚拟环境 (HKFinReg)
*强烈建议配置专属环境以隔离项目依赖的各类 langchain 版本模块，避免本地安装污染。*

**Windows:**
```bash
python -m venv HKFinReg
.\HKFinReg\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv HKFinReg
source HKFinReg/bin/activate
```

### 3. 安装依赖包
```bash
pip install -r requirements.txt
```
*(主要包含 `langchain`, `langchain-openai`, `streamlit`, `streamlit-lottie`, `chromadb`, `pypdf` 等)*

### 4. 设置环境变量
在 `Fintech` 根目录下创建一个完整的 `.env` 文件，并存入模型密钥：

```env
# 智谱AI API Key（处理日常 LLM 与 RAG Embedding 层）
ZHIPU_API_KEY=your_zhipu_api_key

# LongCat API Key（处理高深推理任务）
LONGCAT_API_KEY=your_longcat_api_key
```

### 5. 组装本地合规基准知识库
确认项目下存放有针对 RAG 检索使用的权威 PDF 源文件：
- 默认读取路径：`Fintech/AML Guideline for LCs_Eng_30 Sep 2021.pdf`

### 6. 运行应用
```bash
python -m streamlit run app.py
```
*(启动后将自动在浏览器中打开：http://localhost:8501 )*

---

## 隐私脱敏与安全性

合规数据红线不可侵犯，项目内置了严格的：
- **PII（个人隐私）清洗层 (`pii_scrubber`)**：在明文交互上游，所有暴露的 香港身份证(HKID)、手机号 将在抵达 LLM 的传输信道前被无感替换。
- **状态不落盘**：除了合规的日志审计脱敏快照 `compliance_audit.log` 留存，应用不保存任何未授权的用户记录于 Chroma DB。

---

## 项目架构

```text
Fintech/
├── app.py                      # Streamlit 视觉前端流、控制组件与 CSS
├── core_logic.py               # RAG检索逻辑、LLM初始化路由配置、脱敏探针
├── performance_monitor.py      # I/O 监控、LLM接口耗费时长探针追踪
├── requirements.txt            # 项目依赖
├── .env                        # [Git已忽略] API Key管理配置
├── .gitignore                  # 安全屏蔽配置
├── AML Guideline for LCs...pdf # RAG 本地知识源
└── compliance_audit.log        # 系统合规运转及错误诊断审计日志
```

---

## 许可证信息

本项目采用 **MIT License**。

---

> **Design Notice**：  
> HK-FinReg AI 是面向香港金融科创及监管语境深度定制的展演项目工程。  
> **Built with ⚡ Python + LangChain + Zhipu GLM + LongCat** 

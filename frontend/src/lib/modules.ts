// 模块配置：4大业务线的端点、默认测试数据
import type { ModuleConfig } from "@/types";

export const modules: ModuleConfig[] = [
  {
    id: "svf",
    name: "SVF Compliance",
    nameZh: "SVF 合规审查",
    endpoint: "/api/v1/svf/analyze/stream",
    icon: "📋",
    defaultInput: `SVF Compliance Inquiry:
Company: FastPay HK Limited
License Type: SVF
Service: Stored value facility for retail payments
Transaction Volume: HKD 50 million monthly
AML Officer: Appointed
KYC Procedure: eKYC with facial recognition
Suspicious Transaction Reports: 2 filed in past year`,
  },
  {
    id: "bank",
    name: "Bank Account",
    nameZh: "银行开户审查",
    endpoint: "/api/v1/bank-account/verify/stream",
    icon: "🏦",
    defaultInput: `Account Opening Application:
Name: Chan Tai Man
ID Type: HKID
Occupation: Restaurant Owner
Monthly Income: HKD 35,000
Source of Wealth: Business Income
Purpose of Account: Business Operations
PEP Status: No
Country of Tax Residence: Hong Kong`,
  },
  {
    id: "crossborder",
    name: "Cross-Border",
    nameZh: "跨境汇款评估",
    endpoint: "/api/v1/cross-border/assess/stream",
    icon: "💱",
    defaultInput: `Transaction Log:
Sender: Li Wei, PRC Passport E12345678
Beneficiary: Li Jun (brother), HKID A987654(3)
Amount: USD 48,000
Destination: Hong Kong
Origin: Shenzhen, China
Purpose: Family Support
Frequency: Monthly
Bank: China Construction Bank -> HSBC HK`,
  },
  {
    id: "sme",
    name: "SME Lending",
    nameZh: "SME 信贷评估",
    endpoint: "/api/v1/sme/credit-rating/stream",
    icon: "📈",
    defaultInput: `SME Loan Application:
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
Employees: 5`,
  },
];

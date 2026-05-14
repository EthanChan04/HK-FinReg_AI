"""Map query + product profile into cross-regulator obligations."""

from __future__ import annotations

from app.schemas.kag import ObligationItem, ObligationMapResponse, ProductProfile
from app.services.kag.graph_retriever import GraphRetriever
from app.services.retrieval.retrieval_service import RetrievalService


class ObligationMapper:
    """KAG-led mapper with retrieval evidence backfill."""

    _REGULATOR_RULES: dict[str, tuple[str, ...]] = {
        "HKMA": (
            "svf",
            "bank",
            "banking",
            "lending",
            "loan",
            "payment",
            "remittance",
            "aml",
            "kyc",
            "cdd",
        ),
        "SFC": (
            "securit",
            "investment",
            "robo",
            "advis",
            "suitability",
            "portfolio",
        ),
        "PCPD": (
            "privacy",
            "personal data",
            "pdpo",
            "biometric",
            "facial",
            "chatbot",
            "genai",
            "ai",
            "model",
            "offshore",
        ),
        "IA": ("insurance", "underwriting"),
        "MPFA": ("mpf",),
        "JFIU": ("suspicious", "str", "cross-border", "cross border", "aml utility", "anomaly"),
    }

    _RISK_RULES: dict[str, tuple[str, ...]] = {
        "AML/CFT": ("aml", "cft", "kyc", "cdd", "edd", "suspicious", "str", "monitoring"),
        "Fraud": ("fraud", "anomaly", "suspicious"),
        "Data Privacy": ("privacy", "personal data", "pdpo", "biometric", "facial", "data"),
        "Model Risk": ("ai", "model", "genai", "algorithm", "scoring", "robo"),
        "Cybersecurity": ("cyber", "security", "offshore", "third-party", "third party", "cloud"),
        "Conduct Risk": ("suitability", "investment", "advisor", "conduct"),
    }

    _OBLIGATION_RULES: dict[str, tuple[str, ...]] = {
        "CDD": ("cdd", "kyc", "onboarding", "identity", "sanctions", "wallet"),
        "EDD": ("edd", "enhanced due diligence"),
        "STR": ("str", "suspicious", "anomaly"),
        "Record Keeping": ("record", "retention", "audit trail", "sanctions", "compliance drafting"),
        "Suitability": ("suitability", "investment", "advisor", "recommendation"),
        "Ongoing Monitoring": ("monitoring", "transaction", "anomaly"),
        "Human Oversight": ("ai", "chatbot", "genai", "manual review", "human", "robo", "advisor", "biometric"),
        "Model Governance": ("model", "scoring", "algorithm", "validation", "underwriting"),
        "Data Protection": ("privacy", "personal data", "pdpo", "biometric", "facial", "customer profile", "segments", "data"),
        "Third-party Risk Control": ("third-party", "third party", "vendor", "offshore", "cloud"),
    }

    @staticmethod
    def _normalize_text(query: str, profile: ProductProfile) -> str:
        parts = [
            query or "",
            profile.product_type or "",
            " ".join(profile.business_activity),
            " ".join(profile.data_used),
            " ".join(profile.target_customers),
            " ".join(profile.regulated_entities),
        ]
        return " ".join(parts).lower()

    @staticmethod
    def _matched_rules(text: str, rules: dict[str, tuple[str, ...]]) -> set[str]:
        matches: set[str] = set()
        for label, keywords in rules.items():
            if any(keyword in text for keyword in keywords):
                matches.add(label)
        return matches

    def map_obligations(
        self,
        query: str,
        product_profile: ProductProfile | None,
        graph_retriever: GraphRetriever,
        retrieval_service: RetrievalService,
    ) -> ObligationMapResponse:
        profile = product_profile or ProductProfile()
        graph_paths = graph_retriever.retrieve_paths(query, limit=8)
        evidence = retrieval_service.retrieve(query, retrieval_mode="kag", top_k=8)
        normalized = self._normalize_text(query, profile)

        regulators: set[str] = set()
        products: set[str] = set()
        risks: set[str] = set()
        obligations: list[ObligationItem] = []
        seen_obligations: set[tuple[str, str, str]] = set()
        evidence_ids = [chunk.evidence_id for chunk in evidence[:3]]

        for path in graph_paths:
            path_regulators: list[str] = []
            for node in path.get("path", []):
                upper = str(node).upper()
                if upper in {"HKMA", "SFC", "PCPD", "IA", "MPFA", "JFIU"}:
                    regulator = str(node)
                    regulators.add(regulator)
                    path_regulators.append(regulator)
            products.update(path.get("matched_topics", []))
            for risk in path.get("matched_risks", []):
                risks.add(risk)

            primary_regulator = path_regulators[0] if path_regulators else "Unknown"
            matched_risks = path.get("matched_risks") or ["Operational Risk"]
            primary_risk = matched_risks[0]
            for obligation_name in path.get("matched_obligations", []):
                key = (obligation_name, primary_regulator, primary_risk)
                if key in seen_obligations:
                    continue
                seen_obligations.add(key)
                obligations.append(
                    ObligationItem(
                        obligation=obligation_name,
                        regulator=primary_regulator,
                        risk=primary_risk,
                        controls=["manual review", "audit trail"],
                        evidence_ids=evidence_ids,
                    )
                )

        regulators.update(self._matched_rules(normalized, self._REGULATOR_RULES))
        risks.update(self._matched_rules(normalized, self._RISK_RULES))
        inferred_obligations = self._matched_rules(normalized, self._OBLIGATION_RULES)
        if profile.ai_used:
            inferred_obligations.update({"Human Oversight", "Model Governance"})
        if profile.cross_border:
            inferred_obligations.update({"STR"})
        for obligation_name in sorted(inferred_obligations):
            mapped_risk = "Operational Risk"
            if obligation_name in {"CDD", "EDD", "STR", "Ongoing Monitoring"}:
                mapped_risk = "AML/CFT"
            elif obligation_name in {"Data Protection"}:
                mapped_risk = "Data Privacy"
            elif obligation_name in {"Human Oversight", "Model Governance"}:
                mapped_risk = "Model Risk"
            elif obligation_name in {"Suitability"}:
                mapped_risk = "Conduct Risk"
            elif obligation_name in {"Third-party Risk Control"}:
                mapped_risk = "Cybersecurity"
            risks.add(mapped_risk)
            key = (obligation_name, sorted(regulators)[0] if regulators else "HKMA", mapped_risk)
            if key in seen_obligations:
                continue
            seen_obligations.add(key)
            obligations.append(
                ObligationItem(
                    obligation=obligation_name,
                    regulator=key[1],
                    risk=mapped_risk,
                    controls=["manual review", "audit trail"],
                    evidence_ids=evidence_ids,
                )
            )

        # fallback when graph has no rich obligation nodes yet
        if not obligations:
            fallback_obligation = "Human Oversight" if profile.ai_used else "CDD"
            fallback_risk = "Model Risk" if profile.ai_used else "AML/CFT"
            obligations.append(
                ObligationItem(
                    obligation=fallback_obligation,
                    regulator=sorted(regulators)[0] if regulators else "HKMA",
                    risk=fallback_risk,
                    controls=["manual review"] if profile.ai_used else ["identity verification"],
                    evidence_ids=evidence_ids,
                )
            )
            risks.add(fallback_risk)

        if not regulators:
            regulators.update({"HKMA"})
            if profile.ai_used:
                regulators.add("PCPD")
            if profile.cross_border:
                regulators.add("JFIU")

        if profile.product_type:
            products.add(profile.product_type)
        for activity in profile.business_activity:
            products.add(activity)
        if "cross-border" in normalized or "cross border" in normalized:
            regulators.add("JFIU")
            risks.add("AML/CFT")

        return ObligationMapResponse(
            applicable_regulators=sorted(regulators),
            applicable_products=sorted(products),
            risks=sorted(risks),
            obligations=obligations,
            graph_paths=graph_paths,
        )

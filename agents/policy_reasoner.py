from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    logger.warning("AutoGen not installed — falling back to rule-based reasoning")

LLM_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }
    ],
    "temperature": 0.1,
    "timeout": 60,
    "cache_seed": 42,
}

HF_LLM_CONFIG = {
    "config_list": [
        {
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "api_base": "http://localhost:8000/v1",
            "api_key": "not-needed",
            "api_type": "open_ai",
        }
    ],
    "temperature": 0.1,
}

# Data Models

@dataclass
class PolicyReason:
    policy_id:      str
    policy_title:   str
    policy_text:    str
    change_type:    str
    severity:       str
    billing_codes:  List[str]
    diagnoses:      List[str]
    procedures:     List[str]
    impact_score:   float
    rejection_risk: float
    source_url:     str


@dataclass
class PolicyDecision:

    action:             str
    confidence:         float

    compliance_rules:   List[Dict[str, Any]]
    rejection_codes:    List[str]
    approval_conditions: List[str]

    workflow_updates:   List[Dict[str, str]]
    billing_edits:      List[Dict[str, Any]]

    # Documentation
    rationale:          str
    agent_transcript:   List[Dict[str, str]]
    recommended_actions: List[str]

    metadata: Dict[str, Any] = field(default_factory=dict)

# Agent System Prompts

POLICY_ANALYST_PROMPT = """You are an expert CMS Healthcare Policy Analyst with 20 years of experience 
in Medicare/Medicaid compliance. Your role is to:

1. Analyze policy change text and identify ALL compliance rules
2. Extract specific billing code impacts (CPT, HCPCS, ICD-10)
3. Identify effective dates and implementation timelines
4. Flag prior authorization requirements
5. Note documentation requirements

Always respond in structured JSON format:
{
  "compliance_rules": [{"rule_id": "R001", "description": "...", "applies_to": ["CPT:99213"]}],
  "effective_date": "YYYY-MM-DD",
  "prior_auth_required": true/false,
  "documentation_required": ["..."],
  "key_restrictions": ["..."]
}
Never add "```json" wrappers. Output raw JSON only."""

BILLING_VALIDATOR_PROMPT = """You are a Certified Professional Coder (CPC) and billing compliance expert.
Your role is to:

1. Validate billing codes against the policy analyst's findings
2. Identify code conflicts, bundling issues (NCCI edits), and modifier requirements
3. Check for frequency limitations and place-of-service restrictions
4. Compute rejection risk score for each identified code
5. Suggest correct crosswalk codes when codes are deleted

Always respond in structured JSON:
{
  "code_validations": [
    {
      "code": "99213",
      "code_type": "CPT",
      "valid": true/false,
      "rejection_risk": 0.0-1.0,
      "issues": ["..."],
      "modifiers_required": ["..."],
      "crosswalk": "99214"
    }
  ],
  "bundling_issues": ["..."],
  "ncci_edits": ["..."]
}"""

CLAIMS_ADJUDICATOR_PROMPT = """You are a Senior Claims Adjudication Expert at a major US health insurer.
Based on the policy analysis and billing validation, you must:

1. Make a FINAL DECISION: APPROVE | FLAG_REVIEW | REJECT | AUTO_UPDATE
2. Generate CMS-standard rejection reason codes if applicable
3. Create specific workflow update instructions for the billing system
4. Define what automatic billing system changes are needed

CMS Rejection Codes:
- CO-4:  Procedure code inconsistent with modifier
- CO-11: Diagnosis inconsistent with procedure  
- CO-15: Missing/invalid authorization
- CO-22: This care may be covered by another payer
- CO-50: Non-covered service
- CO-97: Procedure/service not paid separately
- PR-1:  Deductible amount
- PR-2:  Coinsurance amount

Respond in JSON:
{
  "action": "APPROVE|FLAG_REVIEW|REJECT|AUTO_UPDATE",
  "confidence": 0.0-1.0,
  "rejection_codes": ["CO-50"],
  "approval_conditions": ["..."],
  "workflow_updates": [
    {
      "update_type": "ADD_PA_CHECK|UPDATE_CODE|ADD_MODIFIER|FLAG_CODE|DISABLE_CODE",
      "code": "99213",
      "rule": "require PA for >2 units",
      "description": "..."
    }
  ],
  "rationale": "...",
  "recommended_actions": ["..."]
}"""


# Multi-Agent Orchestrator

class PolicyReasonerAgent:

    def __init__(self):
        self._agents_initialized = False
        self._policy_analyst = None
        self._billing_validator = None
        self._claims_adjudicator = None
        self._orchestrator = None

    def _init_agents(self):
        if self._agents_initialized or not AUTOGEN_AVAILABLE:
            return

        llm_cfg = LLM_CONFIG
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not set — agents will use rule-based fallback")
            return

        self._policy_analyst = AssistantAgent(
            name="PolicyAnalyst",
            system_message=POLICY_ANALYST_PROMPT,
            llm_config=llm_cfg,
        )

        self._billing_validator = AssistantAgent(
            name="BillingValidator",
            system_message=BILLING_VALIDATOR_PROMPT,
            llm_config=llm_cfg,
        )

        self._claims_adjudicator = AssistantAgent(
            name="ClaimsAdjudicator",
            system_message=CLAIMS_ADJUDICATOR_PROMPT,
            llm_config=llm_cfg,
        )

        self._orchestrator = UserProxyAgent(
            name="Orchestrator",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
            code_execution_config=False,
        )

        self._agents_initialized = True
        logger.success("AutoGen agents initialized")

    # ── Public API

    async def reason(self, context: PolicyReason) -> PolicyDecision:

        self._init_agents()

        if self._agents_initialized and AUTOGEN_AVAILABLE and self._policy_analyst:
            return await self._autogen_reason(context)
        else:
            return self._rule_based_reason(context)

    # AutoGen Path

    async def _autogen_reason(self, ctx: PolicyReason) -> PolicyDecision:
        try:
            initial_message = self._build_initial_message(ctx)
            transcript: List[Dict[str, str]] = []

            group_chat = GroupChat(
                agents=[
                    self._orchestrator,
                    self._policy_analyst,
                    self._billing_validator,
                    self._claims_adjudicator,
                ],
                messages=[],
                max_round=6,
                speaker_selection_method="round_robin",
            )

            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=LLM_CONFIG,
            )

            await self._orchestrator.a_initiate_chat(
                manager,
                message=initial_message,
            )

            analyst_output = {}
            validator_output = {}
            adjudicator_output = {}

            for msg in group_chat.messages:
                agent = msg.get("name", "")
                content = msg.get("content", "")
                transcript.append({"agent": agent, "message": content[:500]})

                parsed = self._safe_parse_json(content)
                if agent == "PolicyAnalyst" and parsed:
                    analyst_output = parsed
                elif agent == "BillingValidator" and parsed:
                    validator_output = parsed
                elif agent == "ClaimsAdjudicator" and parsed:
                    adjudicator_output = parsed

            return self._build_decision(
                ctx, analyst_output, validator_output, adjudicator_output, transcript
            )

        except Exception as e:
            logger.error(f"AutoGen reasoning failed: {e}")
            return self._rule_based_reason(ctx)

    # Rule-Based Fallback

    def _rule_based_reason(self, ctx: PolicyReason) -> PolicyDecision:

        from nlp.policy_classifier import ChangeCategory, Severity

        compliance_rules = []
        rejection_codes = []
        workflow_updates = []
        billing_edits = []
        approval_conditions = []

        # Rule Application

        if "PRIOR_AUTH" in ctx.change_type.upper():
            for code in ctx.billing_codes:
                compliance_rules.append({
                    "rule_id": f"PA_{code}",
                    "description": f"Prior authorization required for {code}",
                    "applies_to": [code],
                    "mandatory": True,
                })
                workflow_updates.append({
                    "update_type": "ADD_PA_CHECK",
                    "code": code,
                    "rule": "require_prior_auth",
                    "description": f"Add PA verification gate for {code}",
                })
            rejection_codes.append("CO-15")

        if "COVERAGE" in ctx.change_type.upper() and ctx.rejection_risk > 0.5:
            for code in ctx.billing_codes[:10]:
                compliance_rules.append({
                    "rule_id": f"COV_{code}",
                    "description": f"Coverage change detected for {code}",
                    "applies_to": [code],
                    "mandatory": True,
                })
                workflow_updates.append({
                    "update_type": "FLAG_CODE",
                    "code": code,
                    "rule": "coverage_restriction",
                    "description": f"Flag {code} for manual review before submission",
                })
            rejection_codes.extend(["CO-50", "CO-97"])

        if "CODING" in ctx.change_type.upper():
            for code in ctx.billing_codes[:5]:
                billing_edits.append({
                    "code": code,
                    "old_value": code,
                    "new_value": f"{code}_UPDATED",
                    "reason": "Policy coding update requires code revision",
                })

        if "DOCUMENTATION" in ctx.change_type.upper():
            compliance_rules.append({
                "rule_id": "DOC_001",
                "description": "Updated documentation requirements must be met",
                "applies_to": ctx.billing_codes,
                "mandatory": True,
            })
            approval_conditions.append(
                "Clinical documentation must be attached at time of submission"
            )

        # Decision Logic
        if ctx.severity in ("CRITICAL", "HIGH") and ctx.rejection_risk > 0.70:
            action = "REJECT"
            confidence = 0.85
        elif ctx.rejection_risk > 0.50 or ctx.impact_score > 0.60:
            action = "FLAG_REVIEW"
            confidence = 0.78
        elif workflow_updates:
            action = "AUTO_UPDATE"
            confidence = 0.80
        else:
            action = "APPROVE"
            confidence = 0.90

        rationale = (
            f"Policy change '{ctx.change_type}' classified as {ctx.severity} severity. "
            f"Rejection risk: {ctx.rejection_risk:.0%}. Impact score: {ctx.impact_score:.0%}. "
            f"Decision: {action}. {len(workflow_updates)} workflow updates required."
        )

        recommended_actions = [
            f"Update billing system for {len(workflow_updates)} code(s)",
            f"Notify billing team of {ctx.severity} priority change",
            f"Review {len(ctx.billing_codes)} affected billing codes",
        ]
        if action == "REJECT":
            recommended_actions.insert(0, "⚠️ IMMEDIATE ACTION: Halt claims with affected codes")

        return PolicyDecision(
            action=action,
            confidence=confidence,
            compliance_rules=compliance_rules,
            rejection_codes=rejection_codes,
            approval_conditions=approval_conditions,
            workflow_updates=workflow_updates,
            billing_edits=billing_edits,
            rationale=rationale,
            agent_transcript=[{"agent": "RuleEngine", "message": rationale}],
            recommended_actions=recommended_actions,
        )

    # Helpers

    @staticmethod
    def _build_initial_message(ctx: PolicyReason) -> str:
        codes_str = ", ".join(ctx.billing_codes[:20]) or "None identified"
        return f"""
New CMS Policy Change Requires Compliance Analysis:

POLICY ID: {ctx.policy_id}
TITLE: {ctx.policy_title}
CHANGE TYPE: {ctx.change_type}
SEVERITY: {ctx.severity}
IMPACT SCORE: {ctx.impact_score:.0%}
REJECTION RISK: {ctx.rejection_risk:.0%}

AFFECTED BILLING CODES: {codes_str}
DIAGNOSES: {', '.join(ctx.diagnoses[:10]) or 'None'}
PROCEDURES: {', '.join(ctx.procedures[:10]) or 'None'}
SOURCE URL: {ctx.source_url}

POLICY TEXT (excerpt):
{ctx.policy_text[:1000]}

PolicyAnalyst: Please analyze this policy change and extract all compliance rules.
BillingValidator: Validate billing codes and identify risks.
ClaimsAdjudicator: Make final adjudication decision and specify workflow updates.
Reply with TERMINATE when analysis is complete.
"""

    @staticmethod
    def _safe_parse_json(text: str) -> Optional[Dict]:
        try:
            text = text.strip()
            if "```" in text:
                text = re.sub(r"```(?:json)?", "", text).strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        return None

    @staticmethod
    def _build_decision(
        ctx: PolicyReason,
        analyst: Dict,
        validator: Dict,
        adjudicator: Dict,
        transcript: List[Dict],
    ) -> PolicyDecision:
        return PolicyDecision(
            action=adjudicator.get("action", "FLAG_REVIEW"),
            confidence=float(adjudicator.get("confidence", 0.70)),
            compliance_rules=analyst.get("compliance_rules", []),
            rejection_codes=adjudicator.get("rejection_codes", []),
            approval_conditions=adjudicator.get("approval_conditions", []),
            workflow_updates=adjudicator.get("workflow_updates", []),
            billing_edits=validator.get("code_validations", []),
            rationale=adjudicator.get("rationale", "Agent-based analysis complete"),
            agent_transcript=transcript,
            recommended_actions=adjudicator.get("recommended_actions", []),
            metadata={
                "analyst_output": analyst,
                "validator_output": validator,
            },
        )


import re

_reasoner_instance: Optional[PolicyReasonerAgent] = None


def get_reasoner() -> PolicyReasonerAgent:
    global _reasoner_instance
    if _reasoner_instance is None:
        _reasoner_instance = PolicyReasonerAgent()
    return _reasoner_instance

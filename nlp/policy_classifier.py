from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import pipeline as hf_pipeline

# Constants

ZSC_MODEL   = "facebook/bart-large-mnli"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHANGE_LABELS = [
    "coverage expansion",
    "coverage restriction or exclusion",
    "reimbursement rate change",
    "prior authorization requirement",
    "documentation requirement update",
    "billing code addition or deletion",
    "administrative policy update",
    "medical necessity criteria change",
    "frequency or quantity limitation",
    "place of service restriction",
]

SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH":     0.60,
    "MEDIUM":   0.40,
    "LOW":      0.00,
}

# Keywords that escalate severity
CRITICAL_KEYWORDS = [
    "immediate", "retroactive", "penalty", "overpayment", "fraud",
    "mandatory", "all claims", "effective immediately", "corrective action",
    "program integrity",
]

HIGH_KEYWORDS = [
    "prior authorization", "medical necessity", "non-covered",
    "excluded", "denial", "appeal", "suspended",
]



# Enums


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


class ChangeCategory(str, Enum):
    COVERAGE        = "COVERAGE"
    REIMBURSEMENT   = "REIMBURSEMENT"
    PRIOR_AUTH      = "PRIOR_AUTH"
    CODING          = "CODING"
    DOCUMENTATION   = "DOCUMENTATION"
    BILLING_RULE    = "BILLING_RULE"
    ADMINISTRATIVE  = "ADMINISTRATIVE"
    MEDICAL_NECESS  = "MEDICAL_NECESSITY"
    FREQUENCY_LIMIT = "FREQUENCY_LIMIT"
    PLACE_SERVICE   = "PLACE_OF_SERVICE"

# Output Model

@dataclass
class ClassificationResult:
    # Primary classification
    primary_category:   ChangeCategory
    label_scores:       Dict[str, float]        
    change_type:        str                     

    # Risk assessment
    severity:           Severity
    impact_score:       float                    
    rejection_risk:     float                    
    financial_impact:   str                     

    # Billing-specific
    affected_code_types: List[str]               
    action_required:    str                      
    recommended_action: str                      
    urgency_hours:      int                      

    # Metadata
    confidence:         float
    rationale:          str
    raw_label:          str
    metadata:           Dict[str, Any] = field(default_factory=dict)

# Classifier

class PolicyClassifier:

    def __init__(self, use_gpu: bool = False):
        self.device = 0 if (use_gpu and torch.cuda.is_available()) else -1
        self._zsc_pipe = None
        logger.info("PolicyClassifier initialized")

    def _load_zsc(self):
        if self._zsc_pipe is None:
            logger.info(f"Loading {ZSC_MODEL}...")
            self._zsc_pipe = hf_pipeline(
                "zero-shot-classification",
                model=ZSC_MODEL,
                device=self.device,
                multi_label=True,
            )
            logger.success(f"Zero-shot classifier ready: {ZSC_MODEL}")
        return self._zsc_pipe

    # Public API

    def classify(self, text: str, billing_codes: Optional[List[str]] = None) -> ClassificationResult:

        text = text[:2000]
        billing_codes = billing_codes or []

        # Stage 1: Zero-shot classification
        label, scores, confidence = self._zero_shot_classify(text)

        # Stage 2: Map label → ChangeCategory
        category = self._map_label_to_category(label)

        # Stage 3: Severity scoring
        severity, impact_score = self._score_severity(text, category, scores)

        # Stage 4: Rejection risk
        rejection_risk = self._estimate_rejection_risk(category, severity, billing_codes)

        # Stage 5: Action recommendations
        action_required, recommendation, urgency = self._recommend_action(
            category, severity, billing_codes
        )

        # Stage 6: Financial impact label
        financial_impact = self._label_financial_impact(impact_score, text)

        return ClassificationResult(
            primary_category=category,
            label_scores=scores,
            change_type=label,
            severity=severity,
            impact_score=round(impact_score, 3),
            rejection_risk=round(rejection_risk, 3),
            financial_impact=financial_impact,
            affected_code_types=self._affected_code_types(billing_codes),
            action_required=action_required,
            recommended_action=recommendation,
            urgency_hours=urgency,
            confidence=round(confidence, 3),
            rationale=self._build_rationale(label, severity, billing_codes),
            raw_label=label,
        )

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[ClassificationResult]:

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    results.append(self.classify(text))
                except Exception as e:
                    logger.error(f"Classify failed: {e}")
                    results.append(self._fallback_result())
        return results

    # Stage Implementations

    def _zero_shot_classify(self, text: str) -> Tuple[str, Dict[str, float], float]:
        """Returns (top_label, label_scores, top_confidence)."""
        try:
            zsc = self._load_zsc()
            result = zsc(text, CHANGE_LABELS, hypothesis_template="This policy change involves {}.")
            scores = dict(zip(result["labels"], result["scores"]))
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            return top_label, scores, top_score
        except Exception as e:
            logger.warning(f"ZSC failed: {e} — falling back to keyword classification")
            return self._keyword_classify(text)

    @staticmethod
    def _keyword_classify(text: str) -> Tuple[str, Dict[str, float], float]:
        text_lower = text.lower()
        scores: Dict[str, float] = {lbl: 0.1 for lbl in CHANGE_LABELS}

        keyword_map = {
            "coverage expansion": ["now covered", "benefit added", "covered service"],
            "coverage restriction or exclusion": ["non-covered", "excluded", "no longer covered"],
            "reimbursement rate change": ["fee schedule", "RVU", "payment rate", "reimbursement"],
            "prior authorization requirement": ["prior authorization", "pre-auth", "PA required"],
            "documentation requirement update": ["documentation", "medical record", "clinical note"],
            "billing code addition or deletion": ["new code", "deleted code", "code change", "CPT"],
        }

        for label, keywords in keyword_map.items():
            score = sum(0.3 for kw in keywords if kw.lower() in text_lower)
            scores[label] = min(0.95, scores[label] + score)

        top_label = max(scores, key=lambda k: scores[k])
        return top_label, scores, scores[top_label]

    @staticmethod
    def _map_label_to_category(label: str) -> ChangeCategory:
        mapping = {
            "coverage expansion":                 ChangeCategory.COVERAGE,
            "coverage restriction or exclusion":  ChangeCategory.COVERAGE,
            "reimbursement rate change":           ChangeCategory.REIMBURSEMENT,
            "prior authorization requirement":     ChangeCategory.PRIOR_AUTH,
            "documentation requirement update":    ChangeCategory.DOCUMENTATION,
            "billing code addition or deletion":   ChangeCategory.CODING,
            "administrative policy update":        ChangeCategory.ADMINISTRATIVE,
            "medical necessity criteria change":   ChangeCategory.MEDICAL_NECESS,
            "frequency or quantity limitation":    ChangeCategory.FREQUENCY_LIMIT,
            "place of service restriction":        ChangeCategory.PLACE_SERVICE,
        }
        for key, cat in mapping.items():
            if key in label.lower():
                return cat
        return ChangeCategory.ADMINISTRATIVE

    @staticmethod
    def _score_severity(
        text: str,
        category: ChangeCategory,
        scores: Dict[str, float],
    ) -> Tuple[Severity, float]:
        text_lower = text.lower()
        base_score = max(scores.values()) if scores else 0.5

        # Keyword boosters
        if any(kw in text_lower for kw in CRITICAL_KEYWORDS):
            base_score = min(1.0, base_score + 0.30)
        elif any(kw in text_lower for kw in HIGH_KEYWORDS):
            base_score = min(1.0, base_score + 0.15)

        # Category base weights
        category_weights = {
            ChangeCategory.COVERAGE:        0.20,
            ChangeCategory.REIMBURSEMENT:   0.25,
            ChangeCategory.PRIOR_AUTH:      0.20,
            ChangeCategory.MEDICAL_NECESS:  0.25,
            ChangeCategory.CODING:          0.15,
            ChangeCategory.DOCUMENTATION:   0.10,
            ChangeCategory.ADMINISTRATIVE:  0.05,
        }
        base_score = min(1.0, base_score + category_weights.get(category, 0.10))

        # Map to severity
        for sev, threshold in SEVERITY_THRESHOLDS.items():
            if base_score >= threshold:
                return Severity(sev), base_score

        return Severity.LOW, base_score

    @staticmethod
    def _estimate_rejection_risk(
        category: ChangeCategory,
        severity: Severity,
        billing_codes: List[str],
    ) -> float:
        base_risk = {
            ChangeCategory.COVERAGE:       0.65,
            ChangeCategory.PRIOR_AUTH:     0.70,
            ChangeCategory.MEDICAL_NECESS: 0.75,
            ChangeCategory.CODING:         0.60,
            ChangeCategory.REIMBURSEMENT:  0.30,
            ChangeCategory.DOCUMENTATION:  0.55,
            ChangeCategory.ADMINISTRATIVE: 0.20,
            ChangeCategory.FREQUENCY_LIMIT: 0.65,
            ChangeCategory.PLACE_SERVICE:  0.60,
        }.get(category, 0.40)

        severity_multiplier = {
            Severity.CRITICAL: 1.30,
            Severity.HIGH:     1.15,
            Severity.MEDIUM:   1.00,
            Severity.LOW:      0.80,
        }.get(severity, 1.0)

        # More codes = higher aggregate risk
        code_factor = min(1.0, 1.0 + len(billing_codes) * 0.02)
        return min(0.99, base_risk * severity_multiplier * code_factor)

    @staticmethod
    def _recommend_action(
        category: ChangeCategory,
        severity: Severity,
        billing_codes: List[str],
    ) -> Tuple[str, str, int]:
        base_actions = {
            ChangeCategory.COVERAGE: (
                "Update coverage determination rules",
                "Add pre-claim coverage check for affected codes",
                24,
            ),
            ChangeCategory.PRIOR_AUTH: (
                "Implement mandatory PA validation step",
                "Flag all claims with affected codes for prior authorization verification",
                12,
            ),
            ChangeCategory.REIMBURSEMENT: (
                "Update fee schedule in billing system",
                "Reprocess pending claims with corrected rates",
                48,
            ),
            ChangeCategory.CODING: (
                "Update billing code database",
                "Replace deprecated codes; apply crosswalk mapping",
                24,
            ),
            ChangeCategory.DOCUMENTATION: (
                "Revise documentation checklists",
                "Update EHR templates to capture new required fields",
                72,
            ),
            ChangeCategory.MEDICAL_NECESS: (
                "Update medical necessity criteria",
                "Add pre-submission validation against new criteria",
                24,
            ),
        }

        default = ("Review policy change", "Monitor claims for rejection pattern", 72)
        action, recommendation, urgency = base_actions.get(category, default)

        # Escalate urgency for high severity
        if severity == Severity.CRITICAL:
            urgency = max(4, urgency // 6)
        elif severity == Severity.HIGH:
            urgency = max(8, urgency // 3)

        return action, recommendation, urgency

    @staticmethod
    def _label_financial_impact(impact_score: float, text: str) -> str:
        amounts = re.findall(r'\$\s*([\d,]+)', text)
        dollar_sum = sum(
            int(a.replace(",", "")) for a in amounts if a.replace(",", "").isdigit()
        )

        if dollar_sum > 1_000_000 or impact_score > 0.80:
            return f"HIGH (>${dollar_sum:,})" if dollar_sum else "HIGH"
        elif dollar_sum > 100_000 or impact_score > 0.50:
            return f"MEDIUM (${dollar_sum:,})" if dollar_sum else "MEDIUM"
        return "LOW"

    @staticmethod
    def _affected_code_types(codes: List[str]) -> List[str]:
        types = set()
        for code in codes:
            if re.match(r'^\d{5}', code):
                types.add("CPT")
            elif re.match(r'^[A-V]\d{4}', code):
                types.add("HCPCS")
            elif re.match(r'^[A-Z]\d{2}', code):
                types.add("ICD-10")
        return sorted(types)

    @staticmethod
    def _build_rationale(label: str, severity: Severity, codes: List[str]) -> str:
        code_str = f" affecting codes {', '.join(codes[:5])}" if codes else ""
        return (
            f"Classified as '{label}' with {severity.value} severity{code_str}. "
            f"Immediate review required to prevent claim rejections."
        )

    @staticmethod
    def _fallback_result() -> ClassificationResult:
        return ClassificationResult(
            primary_category=ChangeCategory.ADMINISTRATIVE,
            label_scores={},
            change_type="unknown",
            severity=Severity.LOW,
            impact_score=0.0,
            rejection_risk=0.0,
            financial_impact="UNKNOWN",
            affected_code_types=[],
            action_required="Manual review required",
            recommended_action="Review policy change manually",
            urgency_hours=72,
            confidence=0.0,
            rationale="Classification failed — manual review required",
            raw_label="",
        )


_classifier_instance: Optional[PolicyClassifier] = None

def get_classifier(use_gpu: bool = False) -> PolicyClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = PolicyClassifier(use_gpu=use_gpu)
    return _classifier_instance


def classify_policy(text: str, billing_codes: Optional[List[str]] = None) -> ClassificationResult:
    return get_classifier().classify(text, billing_codes)

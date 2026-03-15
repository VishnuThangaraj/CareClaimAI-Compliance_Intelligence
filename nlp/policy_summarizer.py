from __future__ import annotations

import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

# Output model

@dataclass
class PolicyChangeSummary:
    policy_id:    str
    headline:     str          # ≤ 20 words
    overview:     str          # 2-3 sentences
    what_changed: str          # one-liner: "Coverage restriction"
    codes_affected: List[str]  # top 5 billing codes
    severity:     str          # CRITICAL / HIGH / MEDIUM / LOW
    action:       str          # what billing staff must do right now
    urgency:      str          # "Immediate" / "Within 24h" / "Within 48h" / "Routine"
    change_type:  str
    source_url:   str
    effective_date: Optional[str]
    method:       str = "template"   # "model" or "template"

# Change-type → plain-English mappings

_CHANGE_PLAIN: Dict[str, str] = {
    "coverage expansion":                "Coverage expansion",
    "coverage restriction or exclusion": "Coverage restriction",
    "reimbursement rate change":         "Reimbursement rate update",
    "prior authorization requirement":   "Prior authorization added",
    "documentation requirement update":  "Documentation requirement change",
    "billing code addition or deletion": "Billing code update",
    "administrative policy update":      "Administrative update",
    "medical necessity criteria change": "Medical necessity criteria change",
    "frequency or quantity limitation":  "Frequency/quantity limit change",
    "place of service restriction":      "Place-of-service restriction",
}

_SEVERITY_URGENCY: Dict[str, str] = {
    "CRITICAL": "Immediate — stop affected claims now",
    "HIGH":     "Within 24 hours",
    "MEDIUM":   "Within 48 hours",
    "LOW":      "Routine — next billing cycle",
}

_ACTION_TEMPLATES: Dict[str, str] = {
    "coverage restriction or exclusion":
        "Halt submission of claims containing {codes}. Verify coverage status "
        "before resubmitting. Attach medical necessity documentation.",
    "prior authorization requirement":
        "Obtain prior authorization for {codes} before submitting claims. "
        "Add PA number field to claim form and validate at pre-submission.",
    "reimbursement rate change":
        "Update fee schedule for {codes} in the billing system. "
        "Reprocess any pending claims to avoid underpayment or overpayment.",
    "billing code addition or deletion":
        "Replace deprecated codes with approved crosswalk codes. "
        "Update charge master and EHR code sets for {codes}.",
    "documentation requirement update":
        "Update clinical documentation templates to capture new required fields "
        "for {codes}. Train clinical staff on revised requirements.",
    "medical necessity criteria change":
        "Add pre-submission medical necessity validation for {codes}. "
        "Review and update order templates to reflect new criteria.",
    "coverage expansion":
        "New billable services available for {codes}. "
        "Update charge master and notify clinical teams of expanded coverage.",
    "frequency or quantity limitation":
        "Enforce new frequency/quantity limits for {codes} at claim entry. "
        "Flag claims exceeding the limit for manual review.",
    "place of service restriction":
        "Validate place-of-service codes for {codes} against new policy rules. "
        "Incorrect POS codes will trigger automatic claim rejection.",
    "administrative policy update":
        "Review updated administrative requirements for {codes}. "
        "No immediate claim changes required — monitor for follow-up guidance.",
}

# Model-based summariser (lazy load)

class _ModelSummariser:
    _pipe = None

    @classmethod
    def _load(cls):
        if cls._pipe is None:
            try:
                from transformers import pipeline
                logger.info("Loading distilbart summarisation model...")
                cls._pipe = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1,
                    max_length=80,
                    min_length=20,
                    truncation=True,
                )
                logger.success("Summarisation model loaded")
            except Exception as e:
                logger.warning(f"Summarisation model unavailable: {e}")
                cls._pipe = False   # sentinel: don't try again
        return cls._pipe if cls._pipe is not False else None

    @classmethod
    def summarise(cls, text: str) -> Optional[str]:
        pipe = cls._load()
        if not pipe:
            return None
        try:
            # Truncate to ~1024 chars (model sweet-spot)
            snippet = text[:1024].strip()
            if len(snippet.split()) < 30:
                return None
            result = pipe(snippet, do_sample=False)
            return result[0]["summary_text"].strip() if result else None
        except Exception as e:
            logger.warning(f"Summarisation failed: {e}")
            return None

# Template-based fallback (zero dependencies)

def _template_headline(policy: Dict[str, Any]) -> str:
    change_plain = _CHANGE_PLAIN.get(
        policy.get("change_type", "").lower(),
        "Policy update"
    )
    codes = policy.get("billing_codes", [])
    code_str = (
        f"for {', '.join(codes[:3])}" + (" and more" if len(codes) > 3 else "")
        if codes else "affecting billing operations"
    )
    sev = policy.get("severity", "")
    sev_prefix = "CRITICAL: " if sev == "CRITICAL" else "HIGH: " if sev == "HIGH" else ""

    title = policy.get("title", "")
    if title and len(title) <= 80:
        return f"{sev_prefix}{title}"
    return f"{sev_prefix}{change_plain} {code_str}"


def _template_overview(policy: Dict[str, Any]) -> str:
    change_type  = policy.get("change_type", "administrative policy update").lower()
    change_plain = _CHANGE_PLAIN.get(change_type, "Policy update")
    codes        = policy.get("billing_codes", [])
    severity     = policy.get("severity", "LOW")
    risk_pct     = int(policy.get("rejection_risk", 0) * 100)
    impact_pct   = int(policy.get("impact_score", 0) * 100)
    source_type  = policy.get("source_type", "CMS")
    eff_date     = policy.get("effective_date", "")
    if eff_date and hasattr(eff_date, "strftime"):
        eff_date = eff_date.strftime("%B %d, %Y")
    elif isinstance(eff_date, str) and eff_date:
        eff_date = eff_date[:10]

    code_str = (
        ", ".join(codes[:5]) + (" (+{} more)".format(len(codes)-5) if len(codes) > 5 else "")
        if codes else "multiple billing codes"
    )

    # Sentence 1: what happened
    s1_map = {
        "coverage restriction or exclusion":
            f"CMS has restricted or excluded coverage affecting {code_str}.",
        "coverage expansion":
            f"CMS has expanded coverage to include services under {code_str}.",
        "prior authorization requirement":
            f"Prior authorization is now required before submitting claims for {code_str}.",
        "reimbursement rate change":
            f"Reimbursement rates have been updated for {code_str} under the {source_type} policy.",
        "billing code addition or deletion":
            f"Billing code changes affect {code_str} — additions or deletions are now in effect.",
        "documentation requirement update":
            f"New documentation requirements apply to claims containing {code_str}.",
        "medical necessity criteria change":
            f"Medical necessity criteria have been revised for services billed under {code_str}.",
        "frequency or quantity limitation":
            f"Frequency or quantity limits have been updated for {code_str}.",
        "place of service restriction":
            f"Place-of-service restrictions now apply to {code_str}.",
        "administrative policy update":
            f"An administrative policy update from {source_type} affects {code_str}.",
    }
    s1 = s1_map.get(change_type, f"A {change_plain.lower()} from {source_type} affects {code_str}.")
    if eff_date:
        s1 = s1.rstrip(".") + f", effective {eff_date}."

    # Sentence 2: financial/clinical risk
    if risk_pct >= 70:
        s2 = (f"Claims containing these codes carry a {risk_pct}% rejection risk if submitted "
              f"without addressing this change — immediate billing system updates are required.")
    elif risk_pct >= 40:
        s2 = (f"This change introduces a {risk_pct}% rejection risk on affected claims "
              f"and a {impact_pct}% financial impact score — review pending submissions.")
    else:
        s2 = (f"The estimated rejection risk is {risk_pct}% with a {impact_pct}% impact score "
              f"— routine billing updates are sufficient.")

    # Sentence 3: action
    action_tmpl = _ACTION_TEMPLATES.get(
        change_type,
        "Review the updated policy and adjust billing workflows for {codes} accordingly."
    )
    s3 = action_tmpl.format(codes=code_str)

    return f"{s1} {s2} {s3}"


def _urgency_label(severity: str, hours: int) -> str:
    if severity == "CRITICAL":
        return "Immediate — halt affected claims"
    if hours <= 12:
        return f"Urgent — within {hours} hours"
    if hours <= 48:
        return f"Within {hours} hours"
    return "Routine — next billing cycle"



# Public API


def summarise_policy(policy: Dict[str, Any]) -> PolicyChangeSummary:
    """
    Generate a short overview for a policy change document.
    Uses the distilbart model when available, template engine as fallback.
    """
    codes    = policy.get("billing_codes", [])[:8]
    severity = policy.get("severity", "LOW")
    urgency  = _urgency_label(severity, policy.get("urgency_hours", 72))
    change_plain = _CHANGE_PLAIN.get(
        policy.get("change_type", "").lower(), "Policy update"
    )

    # Try model-based overview on the raw_text
    raw   = policy.get("raw_text", "") or policy.get("summary", "")
    model_overview = _ModelSummariser.summarise(raw) if raw else None
    method = "model" if model_overview else "template"
    overview = model_overview or _template_overview(policy)

    headline = _template_headline(policy)

    # Action string
    ct = policy.get("change_type", "").lower()
    code_str = ", ".join(codes[:3]) or "affected codes"
    action = _ACTION_TEMPLATES.get(
        ct,
        "Review policy and update billing workflows for {codes}."
    ).format(codes=code_str).split(".")[0] + "."   # just first sentence

    eff = policy.get("effective_date")
    eff_str = None
    if eff:
        if hasattr(eff, "strftime"):
            eff_str = eff.strftime("%Y-%m-%d")
        elif isinstance(eff, str):
            eff_str = eff[:10]

    return PolicyChangeSummary(
        policy_id     = policy.get("policy_id", ""),
        headline      = headline,
        overview      = overview,
        what_changed  = change_plain,
        codes_affected= codes[:5],
        severity      = severity,
        action        = action,
        urgency       = urgency,
        change_type   = policy.get("change_type", ""),
        source_url    = policy.get("source_url", ""),
        effective_date= eff_str,
        method        = method,
    )


def summarise_batch(policies: List[Dict[str, Any]]) -> List[PolicyChangeSummary]:
    """Summarise a list of policy documents, sorted by severity."""
    _sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    policies_sorted = sorted(
        policies,
        key=lambda p: _sev_order.get(p.get("severity", "LOW"), 4)
    )
    results = []
    for pol in policies_sorted:
        try:
            results.append(summarise_policy(pol))
        except Exception as e:
            logger.warning(f"Summarise failed for {pol.get('policy_id')}: {e}")
    return results

from __future__ import annotations

import sys as _sys, os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from database.mongo import (
    AuditLog,
    AlertsRepository,
    ClaimsRepository,
    MongoManager,
    PolicyRepository,
)
from nlp.policy_classifier import classify_policy
from nlp.policy_ner import extract_policy_entities
from workers.policy_monitor import CH_BILLING_ALERTS, CH_CLAIM_GATE, CH_CRITICAL_ALERTS

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# App Lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    await MongoManager.connect()
    try:
        app.state.redis = aioredis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=True
        )
        await app.state.redis.ping()
        logger.success("Redis connected for billing API")
    except Exception:
        app.state.redis = None
        logger.warning("Redis unavailable — WebSocket alerts disabled")

    logger.success("CareClaimAI Billing API ready")
    yield

    await MongoManager.disconnect()
    if app.state.redis:
        await app.state.redis.aclose()
    logger.info("Billing API shutdown")


app = FastAPI(
    title="CareClaimAI — Billing Validation API",
    description="Real-time CMS policy compliance engine for claim adjudication",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Models

class ServiceLine(BaseModel):
    cpt_code: str
    icd10_codes: List[str] = []
    units: int = 1
    modifier: Optional[str] = None
    charge_amount: float = 0.0
    place_of_service: Optional[str] = None


class ClaimRequest(BaseModel):
    patient_id: str
    provider_npi: str
    service_date: str
    service_lines: List[ServiceLine]
    diagnosis_codes: List[str] = []
    prior_auth_number: Optional[str] = None
    payer_id: Optional[str] = None
    notes: Optional[str] = None


class ValidationResult(BaseModel):
    code: str
    valid: bool
    risk_score: float
    issues: List[str]
    warnings: List[str]
    policy_references: List[str]
    rejection_risk: float
    recommendations: List[str]


class ClaimValidationResponse(BaseModel):
    claim_id: str
    overall_status: str
    overall_risk_score: float
    service_line_results: List[ValidationResult]
    policy_flags: List[Dict[str, Any]]
    rejection_risk: float
    recommended_actions: List[str]
    active_alerts: List[Dict[str, Any]]
    validated_at: str


class ClaimSubmitResponse(BaseModel):
    claim_id: str
    status: str
    risk_score: float
    validation: ClaimValidationResponse
    submitted_at: str


# Core Validation Engine

class BillingValidator:

    @staticmethod
    async def validate_claim(claim: ClaimRequest) -> ClaimValidationResponse:
        claim_id = f"CLM-{uuid.uuid4().hex[:12].upper()}"
        all_codes = [sl.cpt_code for sl in claim.service_lines]
        all_codes += claim.diagnosis_codes

        service_results: List[ValidationResult] = []
        policy_flags: List[Dict] = []
        max_risk = 0.0

        for sl in claim.service_lines:
            result = await BillingValidator._validate_service_line(sl, claim)
            service_results.append(result)
            max_risk = max(max_risk, result.rejection_risk)
            if result.rejection_risk > 0.5 or not result.valid:
                policy_flags.append({
                    "code": sl.cpt_code,
                    "risk": result.rejection_risk,
                    "issues": result.issues,
                })

        # Fetch active alerts for affected codes
        active_alerts = []
        for code in all_codes[:10]:
            alerts = await PolicyRepository.get_policies_by_code(code)
            for alert in alerts[:2]:
                if alert.get("severity") in ("CRITICAL", "HIGH"):
                    active_alerts.append({
                        "code": code,
                        "policy_id": alert.get("policy_id"),
                        "severity": alert.get("severity"),
                        "change_type": alert.get("change_type"),
                        "action_required": alert.get("action_required"),
                    })

        # Overall status
        if max_risk >= 0.75:
            overall_status = "HIGH_RISK"
        elif max_risk >= 0.45:
            overall_status = "REVIEW_REQUIRED"
        else:
            overall_status = "APPROVED"

        recommended_actions = []
        if overall_status == "HIGH_RISK":
            recommended_actions.append("⚠️ Hold submission — manual review required")
        if active_alerts:
            recommended_actions.append(
                f"Active policy alerts for {len(active_alerts)} code(s)"
            )
        if not claim.prior_auth_number and any(r.issues for r in service_results):
            recommended_actions.append("Verify prior authorization requirements")

        return ClaimValidationResponse(
            claim_id=claim_id,
            overall_status=overall_status,
            overall_risk_score=round(max_risk, 3),
            service_line_results=service_results,
            policy_flags=policy_flags,
            rejection_risk=round(max_risk, 3),
            recommended_actions=recommended_actions,
            active_alerts=active_alerts,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    async def _validate_service_line(
        sl: ServiceLine,
        claim: ClaimRequest,
    ) -> ValidationResult:
        issues: List[str] = []
        warnings: List[str] = []
        policy_refs: List[str] = []
        recommendations: List[str] = []
        risk_score = 0.10  # baseline

        # Check CPT code against active policies
        policies = await PolicyRepository.get_policies_by_code(sl.cpt_code)

        for pol in policies:
            severity = pol.get("severity", "LOW")
            change_type = pol.get("change_type", "")
            rejection_risk = pol.get("rejection_risk", 0.0)
            policy_refs.append(pol.get("policy_id", ""))

            # Escalate risk based on policy findings
            if severity == "CRITICAL":
                risk_score = max(risk_score, 0.90)
                issues.append(
                    f"CRITICAL policy change for {sl.cpt_code}: {change_type}"
                )
            elif severity == "HIGH":
                risk_score = max(risk_score, 0.70)
                issues.append(
                    f"HIGH severity policy update for {sl.cpt_code}"
                )
            elif severity == "MEDIUM":
                risk_score = max(risk_score, 0.45)
                warnings.append(f"Policy change may affect {sl.cpt_code}")

            # Prior auth check
            if "PRIOR_AUTH" in change_type.upper():
                if not claim.prior_auth_number:
                    issues.append(f"Prior authorization required for {sl.cpt_code}")
                    risk_score = max(risk_score, 0.80)
                    recommendations.append("Obtain prior authorization before submission")

            # Coverage restriction
            if "COVERAGE_RESTRICTION" in pol.get("change_types", []):
                issues.append(f"{sl.cpt_code} may be non-covered per latest policy")
                risk_score = max(risk_score, 0.75)

        # Diagnosis ↔ Procedure consistency check
        if claim.diagnosis_codes and sl.icd10_codes:
            combined_codes = claim.diagnosis_codes + sl.icd10_codes
            # Simple heuristic: more codes = more risk of inconsistency
            if len(combined_codes) > 8:
                warnings.append("High code count — verify diagnosis-procedure linkage")
                risk_score = min(0.95, risk_score + 0.05)

        # Units check
        if sl.units > 4:
            warnings.append(f"High unit count ({sl.units}) — verify frequency limitation")
            risk_score = min(0.95, risk_score + 0.10)

        # Modifier validation
        if sl.modifier and sl.modifier not in {"25", "59", "GT", "GQ", "95", "76", "77"}:
            warnings.append(f"Verify modifier {sl.modifier} applicability")

        valid = risk_score < 0.60 and not issues

        if not recommendations:
            if risk_score > 0.70:
                recommendations.append("Manual pre-authorization review recommended")
            elif risk_score > 0.40:
                recommendations.append("Verify coverage before submission")

        return ValidationResult(
            code=sl.cpt_code,
            valid=valid,
            risk_score=round(risk_score, 3),
            issues=issues,
            warnings=warnings,
            policy_references=list(set(policy_refs)),
            rejection_risk=round(risk_score, 3),
            recommendations=recommendations,
        )

# WebSocket

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connected | total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# API Route

@app.post("/claims/validate", response_model=ClaimValidationResponse, tags=["Claims"])
async def validate_claim(claim: ClaimRequest):

    result = await BillingValidator.validate_claim(claim)
    await AuditLog.log(
        action="CLAIM_VALIDATED",
        entity_type="claim",
        entity_id=result.claim_id,
        payload={"status": result.overall_status, "risk": result.overall_risk_score},
    )
    return result


@app.post("/claims/submit", response_model=ClaimSubmitResponse, tags=["Claims"])
async def submit_claim(claim: ClaimRequest):

    validation = await BillingValidator.validate_claim(claim)

    # claim status
    if validation.overall_status == "HIGH_RISK":
        status = "flagged"
    elif validation.overall_status == "REVIEW_REQUIRED":
        status = "pending_review"
    else:
        status = "submitted"

    # Persist claim
    claim_doc = {
        "claim_id": validation.claim_id,
        "patient_id": claim.patient_id,
        "provider_npi": claim.provider_npi,
        "service_date": claim.service_date,
        "billing_codes": [sl.cpt_code for sl in claim.service_lines],
        "diagnosis_codes": claim.diagnosis_codes,
        "risk_score": validation.overall_risk_score,
        "status": status,
        "policy_flags": validation.policy_flags,
        "rejection_risk": validation.rejection_risk,
        "prior_auth_number": claim.prior_auth_number,
    }
    await ClaimsRepository.insert_claim(claim_doc)

    if status in ("flagged", "pending_review"):
        await ws_manager.broadcast(json.dumps({
            "type": "CLAIM_FLAGGED",
            "claim_id": validation.claim_id,
            "status": status,
            "risk_score": validation.overall_risk_score,
            "codes": claim_doc["billing_codes"][:5],
        }))

    await AuditLog.log(
        action="CLAIM_SUBMITTED",
        entity_type="claim",
        entity_id=validation.claim_id,
        payload={"status": status, "risk": validation.overall_risk_score},
    )

    return ClaimSubmitResponse(
        claim_id=validation.claim_id,
        status=status,
        risk_score=validation.overall_risk_score,
        validation=validation,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/claims/{claim_id}", tags=["Claims"])
async def get_claim(claim_id: str):
    claim = await ClaimsRepository.get_claim(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


@app.get("/claims/high-risk/list", tags=["Claims"])
async def get_high_risk_claims(
    threshold: float = Query(0.70, ge=0.0, le=1.0),
    limit: int = Query(50, le=200),
):
    claims = await ClaimsRepository.get_high_risk_claims(threshold, limit)
    return {"count": len(claims), "claims": claims}


@app.get("/policies/search", tags=["Policies"])
async def search_policies(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, le=100),
):
    results = await PolicyRepository.search_policies(q, limit)
    return {"count": len(results), "policies": results}


@app.get("/policies/recent", tags=["Policies"])
async def get_recent_changes(hours: int = Query(24, ge=1, le=168)):
    results = await PolicyRepository.get_recent_changes(hours)
    return {"count": len(results), "hours": hours, "policies": results}


@app.get("/policies/{policy_id}", tags=["Policies"])
async def get_policy(policy_id: str):
    policy = await PolicyRepository.get_latest_policy(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


@app.get("/billing-codes/{code}/check", tags=["Billing Codes"])
async def check_billing_code(code: str):
    policies = await PolicyRepository.get_policies_by_code(code)
    alerts = await AlertsRepository.get_active_alerts()
    code_alerts = [a for a in alerts if code in a.get("affected_codes", [])]

    risk = 0.0
    issues = []
    for pol in policies:
        risk = max(risk, pol.get("rejection_risk", 0.0))
        if pol.get("severity") in ("CRITICAL", "HIGH"):
            issues.append(f"{pol.get('severity')} policy change: {pol.get('change_type')}")

    return {
        "code": code,
        "policies_found": len(policies),
        "active_alerts": len(code_alerts),
        "max_rejection_risk": round(risk, 3),
        "status": "HIGH_RISK" if risk > 0.70 else "REVIEW" if risk > 0.40 else "CLEAR",
        "issues": issues,
        "recent_policies": policies[:3],
    }


@app.get("/alerts/active", tags=["Alerts"])
async def get_active_alerts(severity: Optional[str] = None):
    """Get all unresolved policy alerts, optionally filtered by severity."""
    alerts = await AlertsRepository.get_active_alerts(severity)
    return {"count": len(alerts), "alerts": alerts}


@app.post("/alerts/{alert_id}/resolve", tags=["Alerts"])
async def resolve_alert(alert_id: str, resolution_note: str = "Resolved by user"):
    """Mark an alert as resolved."""
    success = await AlertsRepository.resolve_alert(alert_id, resolution_note)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"resolved": True, "alert_id": alert_id}


@app.get("/dashboard/stats", tags=["Dashboard"])
async def get_dashboard_stats():
    claim_stats = await ClaimsRepository.get_claims_dashboard_stats()
    recent_policies = await PolicyRepository.get_recent_changes(hours=24)
    active_alerts = await AlertsRepository.get_active_alerts()

    critical_alerts = [a for a in active_alerts if a.get("severity") == "CRITICAL"]
    high_alerts     = [a for a in active_alerts if a.get("severity") == "HIGH"]

    return {
        "claim_stats":        claim_stats,
        "policy_changes_24h": len(recent_policies),
        "active_alerts":      len(active_alerts),
        "critical_alerts":    len(critical_alerts),
        "high_alerts":        len(high_alerts),
        "top_affected_codes": _extract_top_codes(recent_policies),
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health", tags=["System"])
async def health_check():
    try:
        await MongoManager.db().command("ping")
        db_status = "ok"
    except Exception:
        db_status = "error"

    redis_status = "ok"
    try:
        if app.state.redis:
            await app.state.redis.ping()
        else:
            redis_status = "unavailable"
    except Exception:
        redis_status = "error"

    return {
        "status": "healthy" if db_status == "ok" else "degraded",
        "mongodb": db_status,
        "redis": redis_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# WebSocket — Real-time Alert Stream

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):

    await ws_manager.connect(websocket)
    try:
        if app.state.redis:
            pubsub = app.state.redis.pubsub()
            await pubsub.subscribe(CH_BILLING_ALERTS, CH_CRITICAL_ALERTS, CH_CLAIM_GATE)

            async def _redis_listener():
                async for msg in pubsub.listen():
                    if msg["type"] == "message":
                        await websocket.send_text(msg["data"])

            listener_task = asyncio.create_task(_redis_listener())
            try:
                while True:
                    await websocket.receive_text()
            finally:
                listener_task.cancel()
                await pubsub.unsubscribe()
        else:
            while True:
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({"type": "heartbeat"}))

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

# Helpers

def _extract_top_codes(policies: List[Dict]) -> List[Dict[str, Any]]:
    code_counts: Dict[str, int] = {}
    for pol in policies:
        for code in pol.get("billing_codes", []):
            code_counts[code] = code_counts.get(code, 0) + 1
    return [
        {"code": k, "policy_count": v}
        for k, v in sorted(code_counts.items(), key=lambda x: -x[1])[:10]
    ]

# Policy Change Overview Endpoint

@app.get("/policies/changes/overview", tags=["Policies"])
async def get_policy_changes_overview(
    hours: int = Query(48, ge=1, le=168),
    limit: int = Query(20, le=50),
    severity: Optional[str] = None,
):

    from nlp.policy_summarizer import summarise_batch

    recent = await PolicyRepository.get_recent_changes(hours)
    if severity:
        recent = [p for p in recent if p.get("severity") == severity.upper()]
    recent = recent[:limit]
    summaries = summarise_batch(recent)

    return {
        "period_hours":  hours,
        "total_changes": len(summaries),
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "changes": [
            {
                "policy_id":      s.policy_id,
                "headline":       s.headline,
                "overview":       s.overview,
                "what_changed":   s.what_changed,
                "codes_affected": s.codes_affected,
                "severity":       s.severity,
                "action":         s.action,
                "urgency":        s.urgency,
                "change_type":    s.change_type,
                "source_url":     s.source_url,
                "effective_date": s.effective_date,
                "summary_method": s.method,
            }
            for s in summaries
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.billing_api:app", host="0.0.0.0", port=8000, reload=True)

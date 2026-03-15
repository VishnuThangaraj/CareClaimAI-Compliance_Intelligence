from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from loguru import logger

from database.mongo import (
    AuditLog,
    AlertsRepository,
    MongoManager,
    PolicyRepository,
)
from scraper.cms_scraper import CMSScraper, ScrapedPolicy
from nlp.policy_ner import extract_policy_entities
from nlp.policy_classifier import classify_policy
from agents.policy_reasoner import PolicyReason, get_reasoner

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Redis channels
CH_BILLING_ALERTS  = "billing_alerts"
CH_POLICY_UPDATES  = "policy_updates"
CH_CRITICAL_ALERTS = "critical_alerts"
CH_CLAIM_GATE      = "claim_gate_updates"   

# Policy Processing Pipeline

class PolicyProcessingPipeline:

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.reasoner = get_reasoner()

    async def process(self, policy: ScrapedPolicy) -> Dict[str, Any]:

        logger.info(f"Processing policy: {policy.policy_id}")
        summary: Dict[str, Any] = {
            "policy_id": policy.policy_id,
            "title": policy.title,
            "status": "processing",
            "changes_detected": False,
        }

        try:
            # Step 1: NER 
            ner_result = extract_policy_entities(policy.raw_text)
            logger.debug(
                f"NER: {len(ner_result.entities)} entities, "
                f"{len(ner_result.billing_codes)} billing codes"
            )

            # Step 2: Classify 
            codes = [b["code"] for b in ner_result.billing_codes]
            classification = classify_policy(policy.raw_text[:2000], codes)
            logger.info(
                f"[{policy.policy_id}] Classified: {classification.change_type} "
                f"| {classification.severity} | risk={classification.rejection_risk:.2f}"
            )

            #  Step 3: Build policy document for DB 
            policy_doc = {
                "policy_id":       policy.policy_id,
                "title":           policy.title,
                "source_url":      policy.source_url,
                "source_type":     policy.source_type,
                "raw_text":        policy.raw_text[:10_000],
                "billing_codes":   codes,
                "diagnoses":       ner_result.diagnoses,
                "procedures":      ner_result.procedures,
                "change_type":     classification.change_type,
                "severity":        classification.severity.value,
                "impact_score":    classification.impact_score,
                "rejection_risk":  classification.rejection_risk,
                "financial_impact": classification.financial_impact,
                "action_required": classification.action_required,
                "urgency_hours":   classification.urgency_hours,
                "effective_date":  policy.effective_date,
                "content_hash":    policy.content_hash,
                "ner_summary":     ner_result.summary_entities,
                "change_types":    ner_result.change_types,
            }

            #  Step 4: Store in MongoDB ─
            doc_id = await PolicyRepository.upsert_policy(policy_doc)
            summary["db_id"] = doc_id
            summary["changes_detected"] = True

            #  Step 5: Agent Reasoning (only for HIGH/CRITICAL) ─
            decision = None
            if classification.severity.value in ("CRITICAL", "HIGH"):
                reason_ctx = PolicyReason(
                    policy_id=policy.policy_id,
                    policy_title=policy.title,
                    policy_text=policy.raw_text[:1500],
                    change_type=classification.change_type,
                    severity=classification.severity.value,
                    billing_codes=codes[:20],
                    diagnoses=ner_result.diagnoses[:10],
                    procedures=ner_result.procedures[:10],
                    impact_score=classification.impact_score,
                    rejection_risk=classification.rejection_risk,
                    source_url=policy.source_url,
                )
                decision = await self.reasoner.reason(reason_ctx)
                policy_doc["agent_decision"] = {
                    "action":           decision.action,
                    "confidence":       decision.confidence,
                    "workflow_updates": decision.workflow_updates,
                    "rejection_codes":  decision.rejection_codes,
                    "rationale":        decision.rationale,
                }
                summary["agent_action"] = decision.action

            #  Step 6: Create Alert ─
            alert_id = await AlertsRepository.create_alert({
                "policy_id":       policy.policy_id,
                "title":           f"Policy Change: {policy.title[:100]}",
                "severity":        classification.severity.value,
                "change_type":     classification.change_type,
                "affected_codes":  codes[:20],
                "rejection_risk":  classification.rejection_risk,
                "source_url":      policy.source_url,
                "action_required": classification.action_required,
                "urgency_hours":   classification.urgency_hours,
                "agent_action":    decision.action if decision else "PENDING_REVIEW",
                "workflow_updates": decision.workflow_updates if decision else [],
            })
            summary["alert_id"] = alert_id

            #  Step 7: Publish to Redis ─
            await self._publish_alerts(policy, classification, decision)

            #  Step 8: Audit Log ─
            await AuditLog.log(
                action="POLICY_PROCESSED",
                entity_type="policy",
                entity_id=policy.policy_id,
                payload={
                    "severity": classification.severity.value,
                    "codes_affected": len(codes),
                    "agent_action": decision.action if decision else "N/A",
                },
            )

            summary["status"] = "completed"
            logger.success(
                f"✓ Policy processed: {policy.policy_id} | "
                f"{classification.severity} | {decision.action if decision else 'QUEUED'}"
            )

        except Exception as e:
            logger.error(f"Pipeline failed for {policy.policy_id}: {e}")
            summary["status"] = "failed"
            summary["error"] = str(e)

        return summary

    async def _publish_alerts(
        self,
        policy: ScrapedPolicy,
        classification: Any,
        decision: Optional[Any],
    ) -> None:
        """Publish structured alerts to Redis channels."""
        payload = {
            "policy_id":      policy.policy_id,
            "title":          policy.title[:120],
            "severity":       classification.severity.value,
            "change_type":    classification.change_type,
            "rejection_risk": classification.rejection_risk,
            "billing_codes":  [b["code"] for b in []],  # resolved below
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "action":         decision.action if decision else "REVIEW",
            "workflow_updates": decision.workflow_updates if decision else [],
        }

        payload_str = json.dumps(payload)

        # Always publish to general updates
        await self.redis.publish(CH_POLICY_UPDATES, payload_str)

        # Severity-based channels
        if classification.severity.value in ("CRITICAL", "HIGH"):
            await self.redis.publish(CH_BILLING_ALERTS, payload_str)

        if classification.severity.value == "CRITICAL":
            await self.redis.publish(CH_CRITICAL_ALERTS, payload_str)

        # Billing gate updates (workflow changes)
        if decision and decision.workflow_updates:
            gate_payload = json.dumps({
                "policy_id": policy.policy_id,
                "workflow_updates": decision.workflow_updates,
                "rejection_codes": decision.rejection_codes,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await self.redis.publish(CH_CLAIM_GATE, gate_payload)
            logger.info(f"Published {len(decision.workflow_updates)} billing gate updates")



# Monitor Worker


class PolicyMonitorWorker:

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.scraper = CMSScraper(concurrency=5)
        self._redis: Optional[aioredis.Redis] = None
        self._pipeline: Optional[PolicyProcessingPipeline] = None
        self._is_running = False
        self._stats = {
            "total_scans": 0,
            "policies_processed": 0,
            "alerts_created": 0,
            "errors": 0,
            "last_scan": None,
        }

    async def start(self) -> None:
        logger.info("Starting PolicyMonitorWorker...")

        # Connect DB
        await MongoManager.connect()

        # Connect Redis
        try:
            self._redis = aioredis.from_url(
                REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await self._redis.ping()
            logger.success("Redis connected")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e} — alerts will be DB-only")
            self._redis = None

        self._pipeline = PolicyProcessingPipeline(
            self._redis or _NullRedis()
        )

        # Register scheduled jobs
        self.scheduler.add_job(
            self._scan_cms_job,
            trigger=IntervalTrigger(minutes=30),
            id="scan_cms",
            name="CMS Policy Scanner",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self.scheduler.add_job(
            self._health_check_job,
            trigger=IntervalTrigger(minutes=10),
            id="health_check",
            name="Health Check",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._cleanup_job,
            trigger=IntervalTrigger(hours=24),
            id="cleanup",
            name="Daily Cleanup",
            replace_existing=True,
        )

        self.scheduler.start()
        self._is_running = True
        logger.success("PolicyMonitorWorker started | Jobs: scan(30m), health(10m), cleanup(24h)")

        # Run first scan immediately
        await self._scan_cms_job()

    async def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        await MongoManager.disconnect()
        if self._redis:
            await self._redis.aclose()
        self._is_running = False
        logger.info("PolicyMonitorWorker stopped")

    #  Scheduled Jobs ─

    async def _scan_cms_job(self) -> None:
        """Main CMS scraping + processing job."""
        logger.info(f"[SCAN] Starting CMS scan #{self._stats['total_scans'] + 1}")
        self._stats["total_scans"] += 1
        self._stats["last_scan"] = datetime.now(timezone.utc).isoformat()

        try:
            # Scrape all CMS targets
            new_policies = await self.scraper.scrape_all()
            logger.info(f"[SCAN] Found {len(new_policies)} new/changed policies")

            if not new_policies:
                logger.info("[SCAN] No policy changes detected")
                return

            # Process concurrently (cap at 3 to avoid overloading NLP)
            sem = asyncio.Semaphore(3)

            async def _process_one(p: ScrapedPolicy):
                async with sem:
                    return await self._pipeline.process(p)

            summaries = await asyncio.gather(
                *[_process_one(p) for p in new_policies],
                return_exceptions=True,
            )

            # Tally stats
            for s in summaries:
                if isinstance(s, Exception):
                    self._stats["errors"] += 1
                elif isinstance(s, dict):
                    if s.get("status") == "completed":
                        self._stats["policies_processed"] += 1
                    if s.get("alert_id"):
                        self._stats["alerts_created"] += 1

            logger.success(
                f"[SCAN] Complete: {self._stats['policies_processed']} processed, "
                f"{self._stats['alerts_created']} alerts"
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"[SCAN] Job failed: {e}")

    async def _health_check_job(self) -> None:
        """Verify all connections are alive."""
        try:
            if self._redis:
                await self._redis.ping()
            # Quick DB ping
            await MongoManager.db().command("ping")
            logger.debug("Health check: OK")
        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def _cleanup_job(self) -> None:
        """Remove stale unresolved low-severity alerts older than 7 days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        try:
            col = MongoManager.db().policy_alerts
            result = await col.delete_many({
                "severity": "LOW",
                "resolved": True,
                "detected_at": {"$lt": cutoff},
            })
            logger.info(f"Cleanup: removed {result.deleted_count} stale alerts")
        except Exception as e:
            logger.warning(f"Cleanup job error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "is_running": self._is_running}



# Null Redis (fallback when Redis is unavailable)


class _NullRedis:
    """No-op Redis substitute for environments without Redis."""

    async def publish(self, channel: str, message: str) -> None:
        logger.debug(f"[NullRedis] Would publish to '{channel}': {message[:80]}")

    async def ping(self) -> bool:
        return True



# Entry point for standalone worker


async def run_worker():
    worker = PolicyMonitorWorker()
    try:
        await worker.start()
        # Keep alive
        while True:
            await asyncio.sleep(60)
            stats = worker.get_stats()
            logger.info(
                f"Worker stats | Scans: {stats['total_scans']} | "
                f"Processed: {stats['policies_processed']} | "
                f"Alerts: {stats['alerts_created']} | "
                f"Errors: {stats['errors']}"
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker interrupted — shutting down")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(run_worker())

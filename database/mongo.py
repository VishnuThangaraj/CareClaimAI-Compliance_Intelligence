from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

load_dotenv()

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME: str = os.getenv("MONGO_DB", "careclaim_ai")

class _MongoManager:
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    async def connect(cls) -> None:
        if cls._client is None:
            cls._client = AsyncIOMotorClient(
                MONGO_URI,
                maxPoolSize=50,
                minPoolSize=5,
                serverSelectionTimeoutMS=5_000,
            )
            cls._db = cls._client[DB_NAME]
            await cls._ensure_indexes()
            logger.success(f"MongoDB connected → {DB_NAME}")

    @classmethod
    async def disconnect(cls) -> None:
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            logger.info("MongoDB disconnected")

    @classmethod
    def db(cls) -> AsyncIOMotorDatabase:
        if cls._db is None:
            raise RuntimeError("MongoDB not connected. Call MongoManager.connect() first.")
        return cls._db

    @classmethod
    async def _ensure_indexes(cls) -> None:
        db = cls._db

        # policies 
        await db.policies.create_indexes([
            IndexModel([("policy_id", ASCENDING), ("version", DESCENDING)], unique=True),
            IndexModel([("source_url", ASCENDING)]),
            IndexModel([("effective_date", DESCENDING)]),
            IndexModel([("billing_codes", ASCENDING)]),          
            IndexModel([("change_type", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ])

        # billing_codes 
        await db.billing_codes.create_indexes([
            IndexModel([("code", ASCENDING)], unique=True),
            IndexModel([("code_type", ASCENDING)]),              
            IndexModel([("active", ASCENDING)]),
        ])

        # claims 
        await db.claims.create_indexes([
            IndexModel([("claim_id", ASCENDING)], unique=True),
            IndexModel([("patient_id", ASCENDING)]),
            IndexModel([("billing_codes", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("submitted_at", DESCENDING)]),
            IndexModel([("risk_score", DESCENDING)]),
        ])

        # policy_alerts 
        await db.policy_alerts.create_indexes([
            IndexModel([("severity", ASCENDING)]),
            IndexModel([("resolved", ASCENDING)]),
            IndexModel([("affected_codes", ASCENDING)]),
            IndexModel([("detected_at", DESCENDING)]),
        ])

        # audit_log 
        await db.audit_log.create_indexes([
            IndexModel([("entity_type", ASCENDING), ("entity_id", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("action", ASCENDING)]),
        ])

        logger.debug("MongoDB indexes ensured")


MongoManager = _MongoManager

# Policy Repository

class PolicyRepository:

    @staticmethod
    def _col():
        return MongoManager.db().policies

    @staticmethod
    async def upsert_policy(policy: Dict[str, Any]) -> str:

        col = PolicyRepository._col()
        policy_id = policy["policy_id"]

        # Find latest version
        latest = await col.find_one(
            {"policy_id": policy_id},
            sort=[("version", DESCENDING)],
        )
        new_version = (latest["version"] + 1) if latest else 1

        doc = {
            **policy,
            "version": new_version,
            "created_at": datetime.now(timezone.utc),
            "is_latest": True,
        }

        # Mark previous as not latest
        if latest:
            await col.update_many(
                {"policy_id": policy_id},
                {"$set": {"is_latest": False}},
            )

        result = await col.insert_one(doc)
        logger.info(f"Policy upserted: {policy_id} v{new_version}")
        return str(result.inserted_id)

    @staticmethod
    async def get_latest_policy(policy_id: str) -> Optional[Dict]:
        col = PolicyRepository._col()
        doc = await col.find_one(
            {"policy_id": policy_id, "is_latest": True},
            sort=[("version", DESCENDING)],
        )
        return _serialize(doc)

    @staticmethod
    async def get_policies_by_code(billing_code: str) -> List[Dict]:
        col = PolicyRepository._col()
        cursor = col.find(
            {"billing_codes": billing_code, "is_latest": True},
            sort=[("effective_date", DESCENDING)],
        )
        return [_serialize(d) async for d in cursor]

    @staticmethod
    async def get_recent_changes(hours: int = 24, limit: int = 100) -> List[Dict]:
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        col = PolicyRepository._col()
        cursor = col.find(
            {"created_at": {"$gte": cutoff}},
            sort=[("created_at", DESCENDING)],
        ).limit(limit)
        return [_serialize(d) async for d in cursor]

    @staticmethod
    async def search_policies(query: str, limit: int = 20) -> List[Dict]:
        col = PolicyRepository._col()
        # Text search (requires text index)
        try:
            cursor = col.find(
                {"$text": {"$search": query}, "is_latest": True},
                sort=[("score", {"$meta": "textScore"})],
            ).limit(limit)
            return [_serialize(d) async for d in cursor]
        except Exception:
            # Fallback to regex
            cursor = col.find(
                {"title": {"$regex": query, "$options": "i"}, "is_latest": True}
            ).limit(limit)
            return [_serialize(d) async for d in cursor]



# Claims Repository


class ClaimsRepository:

    @staticmethod
    def _col():
        return MongoManager.db().claims

    @staticmethod
    async def insert_claim(claim: Dict[str, Any]) -> str:
        col = ClaimsRepository._col()
        doc = {
            **claim,
            "submitted_at": datetime.now(timezone.utc),
            "status": claim.get("status", "pending"),
            "policy_version_snapshot": [],
        }
        result = await col.insert_one(doc)
        return str(result.inserted_id)

    @staticmethod
    async def update_claim_status(
        claim_id: str,
        status: str,
        rejection_reasons: Optional[List[str]] = None,
        risk_score: Optional[float] = None,
        policy_flags: Optional[List[Dict]] = None,
    ) -> bool:
        col = ClaimsRepository._col()
        update: Dict[str, Any] = {
            "$set": {
                "status": status,
                "updated_at": datetime.now(timezone.utc),
            }
        }
        if rejection_reasons is not None:
            update["$set"]["rejection_reasons"] = rejection_reasons
        if risk_score is not None:
            update["$set"]["risk_score"] = risk_score
        if policy_flags is not None:
            update["$set"]["policy_flags"] = policy_flags

        result = await col.update_one({"claim_id": claim_id}, update)
        return result.modified_count > 0

    @staticmethod
    async def get_claim(claim_id: str) -> Optional[Dict]:
        col = ClaimsRepository._col()
        return _serialize(await col.find_one({"claim_id": claim_id}))

    @staticmethod
    async def get_high_risk_claims(threshold: float = 0.7, limit: int = 50) -> List[Dict]:
        col = ClaimsRepository._col()
        cursor = col.find(
            {"risk_score": {"$gte": threshold}, "status": "pending"},
            sort=[("risk_score", DESCENDING)],
        ).limit(limit)
        return [_serialize(d) async for d in cursor]

    @staticmethod
    async def get_claims_dashboard_stats() -> Dict[str, Any]:
        col = ClaimsRepository._col()
        pipeline = [
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1},
                    "avg_risk": {"$avg": "$risk_score"},
                }
            }
        ]
        stats: Dict[str, Any] = {}
        async for doc in col.aggregate(pipeline):
            stats[doc["_id"]] = {
                "count": doc["count"],
                "avg_risk": round(doc.get("avg_risk") or 0, 3),
            }
        return stats



# Alerts Repository


class AlertsRepository:

    @staticmethod
    def _col():
        return MongoManager.db().policy_alerts

    @staticmethod
    async def create_alert(alert: Dict[str, Any]) -> str:
        col = AlertsRepository._col()
        doc = {
            **alert,
            "detected_at": datetime.now(timezone.utc),
            "resolved": False,
        }
        result = await col.insert_one(doc)
        logger.warning(
            f"[ALERT] {alert.get('severity','UNKNOWN')} — {alert.get('title','')}"
        )
        return str(result.inserted_id)

    @staticmethod
    async def resolve_alert(alert_id: str, resolution_note: str) -> bool:
        col = AlertsRepository._col()
        result = await col.update_one(
            {"_id": ObjectId(alert_id)},
            {
                "$set": {
                    "resolved": True,
                    "resolved_at": datetime.now(timezone.utc),
                    "resolution_note": resolution_note,
                }
            },
        )
        return result.modified_count > 0

    @staticmethod
    async def get_active_alerts(severity: Optional[str] = None) -> List[Dict]:
        col = AlertsRepository._col()
        query: Dict[str, Any] = {"resolved": False}
        if severity:
            query["severity"] = severity
        cursor = col.find(query, sort=[("detected_at", DESCENDING)])
        return [_serialize(d) async for d in cursor]



# Audit Log


class AuditLog:

    @staticmethod
    async def log(
        action: str,
        entity_type: str,
        entity_id: str,
        payload: Optional[Dict] = None,
        actor: str = "system",
    ) -> None:
        col = MongoManager.db().audit_log
        await col.insert_one({
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "actor": actor,
            "payload": payload or {},
            "timestamp": datetime.now(timezone.utc),
        })



# Helpers


def _serialize(doc: Optional[Dict]) -> Optional[Dict]:
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc

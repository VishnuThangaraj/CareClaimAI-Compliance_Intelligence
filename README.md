# 🏥 CareClaimAI — Autonomous Compliance Intelligence Platform

> **Datathon Project** — Regulatory Entropy Management for US Healthcare Billing

CareClaimAI is an autonomous AI agent that continuously monitors CMS (Centers for Medicare & Medicaid Services) policy pages, interprets regulatory changes using medical NLP, and updates billing system rules in real time — preventing claim rejections before they happen.

---

## 🎯 Problem Statement

The US healthcare reimbursement landscape has reached a state of **regulatory entropy** — thousands of policy updates across disparate CMS portals make manual compliance tracking a systemic liability. Static monitoring tools fail because they cannot:

- Interpret the **clinical intent** of a policy change
- Map changes to **specific billing codes** and their financial impact
- Execute **workflow adjustments** in real time to prevent catastrophic claim rejections

CareClaimAI solves this with an autonomous agent pipeline: **scrape → understand → reason → update**.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Autonomous CMS Scraping** | Async scraper hits CMS.gov every 30 minutes — LCD, NCD, Fee Schedules, MLN Matters |
| 🧠 **Medical NER** | BioBERT extracts CPT, HCPCS, ICD-10 codes and clinical entities from policy text |
| ⚡ **Zero-Shot Classification** | BART-large-MNLI classifies change type and severity without labeled training data |
| 🤖 **Multi-Agent Reasoning** | AutoGen 3-agent system: PolicyAnalyst → BillingValidator → ClaimsAdjudicator |
| 🛡 **Real-Time Claim Gate** | Every claim validated against live policy database before reaching the payer |
| 📊 **Compliance Dashboard** | Streamlit dashboard with regulatory entropy score, change cards, and code risk heatmap |
| 📝 **Plain-English Summaries** | DistilBART generates human-readable overviews of every detected policy change |
| 🔔 **Alert System** | CRITICAL/HIGH severity changes trigger immediate billing workflow updates |

---

## 🏗 Architecture

```
CMS.gov (Live)
    │
    ▼
cms_scraper.py          ← Async BeautifulSoup + Playwright, SHA-256 dedup
    │
    ▼
policy_ner.py           ← d4data/biomedical-ner-all (BioBERT) + clinical NER
policy_classifier.py    ← facebook/bart-large-mnli (zero-shot, no labels needed)
policy_summarizer.py    ← sshleifer/distilbart-cnn-12-6 + template fallback
    │
    ▼
policy_reasoner.py      ← AutoGen: PolicyAnalyst → BillingValidator → ClaimsAdjudicator
    │
    ▼
policy_monitor.py       ← APScheduler worker: scan(30m) + Redis pub/sub alerts
    │
    ├── MongoDB          ← policies, claims, alerts, audit_log (versioned)
    └── Redis            ← billing_alerts, critical_alerts, claim_gate_updates
              │
              ▼
billing_api.py          ← FastAPI: /claims/validate, /policies/changes/overview, WS /ws/alerts
              │
              ▼
main_dashboard.py       ← Streamlit: 5-page compliance operations center
```

---

## 📁 Project Structure

```
careclaim_ai/
├── run_platform.py          # Platform orchestrator — starts all services
├── requirements.txt
├── .env.example
│
├── scraper/
│   └── cms_scraper.py       # Async CMS scraper with quality filtering
│
├── nlp/
│   ├── policy_ner.py        # Medical NER — extracts billing codes + entities
│   ├── policy_classifier.py # Zero-shot change classification + risk scoring
│   └── policy_summarizer.py # Plain-English headline + 3-sentence overview
│
├── agents/
│   └── policy_reasoner.py   # AutoGen multi-agent compliance reasoning
│
├── workers/
│   └── policy_monitor.py    # Background worker with APScheduler
│
├── api/
│   └── billing_api.py       # FastAPI REST + WebSocket
│
├── database/
│   └── mongo.py             # Async Motor: policies, claims, alerts, audit log
│
├── dashboard/
│   └── main_dashboard.py    # Streamlit 5-page dashboard
│
└── logs/                    # Auto-created on first run
    ├── api.log
    └── dashboard.log
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- MongoDB (local or Atlas)
- Redis *(optional — graceful fallback without it)*

### 1. Clone & Install

```bash
git clone https://github.com/your-org/careclaim-ai.git
cd careclaim-ai
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
MONGO_URI=mongodb://localhost:27017
MONGO_DB=careclaim_ai

# Optional — Redis for real-time WebSocket alerts
REDIS_URL=redis://localhost:6379/0

# Optional — enables AutoGen multi-agent reasoning (falls back to rule engine)
OPENAI_API_KEY=sk-...
```

### 3. Install Playwright (for JS-rendered CMS pages)

```bash
playwright install chromium
```

### 4. Run the Platform

```bash
# Start all services (API + Dashboard + Monitor Worker)
python run_platform.py run

# First run — scrape live CMS data immediately
python run_platform.py run --initial-scan

# API only (no dashboard)
python run_platform.py run --api-only

# Check environment
python run_platform.py check
```

### 5. Access the Dashboard

| Service | URL |
|---|---|
| 📊 Compliance Dashboard | http://localhost:8501 |
| 🔌 Billing API | http://localhost:8000 |
| 📖 API Documentation | http://localhost:8000/docs |

---

## 🖥 Dashboard Pages

### 📊 Dashboard
Live command centre with:
- **4 KPIs** — total claims, flagged, policy changes, active alerts
- **Regulatory Entropy Score** — proprietary metric measuring policy-change fragmentation velocity
- **Policy Change Intelligence Feed** — plain-English summaries of every detected CMS change
- **Claim pipeline donut** and **severity breakdown** charts

### 📋 Policy Changes
- Full list of detected changes sorted by severity
- **Billing code rejection risk chart** — horizontal bars per CPT code, coloured by severity
- CRITICAL-only filter toggle

### 💊 Claim Validator
Pre-submission compliance check:
- Enter CPT, ICD-10, units, modifier, prior auth, place of service
- Returns: overall status (APPROVED / REVIEW REQUIRED / HIGH RISK), rejection risk %, and the specific policy change driving each flag

### 🔎 Code Lookup
- **10 popular codes** as quick-access buttons (99213, 99214, G0439, 74178…)
- Free-text entry for any CPT, HCPCS, or ICD-10 code
- Returns all active policy changes, rejection risk, and required action

### ⚠️ Alerts
- All unresolved HIGH/CRITICAL alerts with one-click resolution
- Shows: CMS reason codes, AutoGen agent decision, workflow updates dispatched

---

## 🤖 AI Models Used

| Model | Source | Purpose |
|---|---|---|
| `d4data/biomedical-ner-all` | HuggingFace | Medical NER — extracts clinical entities |
| `samrawal/bert-base-uncased_clinical-ner` | HuggingFace | Clinical NER — diagnoses, treatments |
| `facebook/bart-large-mnli` | HuggingFace | Zero-shot change type classification |
| `sshleifer/distilbart-cnn-12-6` | HuggingFace | Policy change summarisation |
| AutoGen `gpt-4o-mini` | OpenAI (optional) | Multi-agent compliance reasoning |

> **No OpenAI key required** — the multi-agent system falls back to a deterministic rule engine that produces the same structured outputs using severity, risk score, and NER results.

---

## 🔌 API Reference

### Key Endpoints

```
GET  /policies/changes/overview   Plain-English summaries of recent changes
POST /claims/validate             Pre-submission policy compliance check
POST /claims/submit               Submit claim with inline validation
GET  /billing-codes/{code}/check  Single-code policy status lookup
GET  /alerts/active               All unresolved policy alerts
GET  /dashboard/stats             Aggregate KPI metrics
GET  /health                      Service health check
WS   /ws/alerts                   Real-time alert stream (requires Redis)
```

### Example — Validate a Claim

```bash
curl -X POST http://localhost:8000/claims/validate \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PT-001",
    "provider_npi": "1234567890",
    "service_date": "2025-01-15",
    "service_lines": [{
      "cpt_code": "99213",
      "icd10_codes": ["E11.9"],
      "units": 1
    }],
    "diagnosis_codes": ["E11.9"]
  }'
```

### Example — Get Policy Change Overview

```bash
curl "http://localhost:8000/policies/changes/overview?hours=48&severity=HIGH"
```

---

## 📊 The Regulatory Entropy Score

CareClaimAI introduces a novel metric — the **Regulatory Entropy Score (0–100)** — that quantifies how chaotic the current policy environment is:

```
Entropy = (CRITICAL changes × 15) + (HIGH changes × 7) + 
          (total changes × 2, capped at 40) + (unresolved CRITICAL alerts × 10)
```

| Score | Tier | Meaning |
|---|---|---|
| 75–100 | 🔴 Critical Entropy | Immediate billing system action required |
| 50–74 | 🟠 High Entropy | Multiple HIGH severity changes active |
| 25–49 | 🟡 Moderate | Monitor closely |
| 0–24 | 🟢 Stable | Policy environment is calm |

---

## 🏆 Technical Highlights

- **Zero-shot NLP** — classifies policy changes without any labeled training data using BART-large-MNLI
- **Content deduplication** — SHA-256 hashing ensures no policy is processed twice
- **Quality filtering** — CMS navigation pages and index URLs are blocked at three independent layers (scraper, summarizer, dashboard)
- **Graceful degradation** — works without Redis (falls back to polling), without OpenAI (uses rule engine), and on CPU (no GPU required)
- **Auto-restart** — subprocess monitor detects crashes and restarts services automatically
- **Immutable audit log** — every policy change and claim validation is logged to MongoDB with timestamps

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Scraping | `aiohttp`, `BeautifulSoup4`, `Playwright` |
| NLP | `transformers`, `torch`, `spaCy`, `scispaCy` |
| Multi-Agent | `pyautogen` |
| API | `FastAPI`, `Uvicorn`, `WebSocket` |
| Database | `MongoDB` (Motor async driver) |
| Cache/Pub-Sub | `Redis` |
| Scheduling | `APScheduler` |
| Dashboard | `Streamlit`, `Plotly` |
| Orchestration | `Typer`, `Rich`, `Loguru` |

---

## 📝 License

MIT License — built for the CareClaimAI Datathon.

---

*Built to solve the $262 billion annual US healthcare claim denial problem.*

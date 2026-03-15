from __future__ import annotations
import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=_ROOT / ".env")

API_BASE    = os.getenv("API_URL", "http://localhost:8000")
CACHE_TTL   = 30

# Page config
st.set_page_config(
    page_title="CareClaimAI — Compliance Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens
BG      = "#07090f"
SURFACE = "#0d1117"
BORDER  = "#1a2234"
TEXT    = "#dce3ed"
MUTED   = "#5a7090"
DIMMED  = "#253548"

SEV_COLOR = {"CRITICAL":"#f87171","HIGH":"#fb923c","MEDIUM":"#fbbf24","LOW":"#4ade80"}
SEV_BG    = {"CRITICAL":"#2d0f0f","HIGH":"#2d1900","MEDIUM":"#2d2600","LOW":"#0d2b16"}
SEV_CLS   = {"CRITICAL":"sev-crit","HIGH":"sev-high","MEDIUM":"sev-med","LOW":"sev-low"}
SEV_EMOJI = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}

CHANGE_ICON = {
    "coverage restriction or exclusion": "⊘",
    "coverage expansion":                "✚",
    "reimbursement rate change":         "＄",
    "prior authorization requirement":   "🔒",
    "billing code addition or deletion": "⇄",
    "documentation requirement update":  "📄",
    "medical necessity criteria change": "✦",
    "frequency or quantity limitation":  "⏱",
    "place of service restriction":      "◈",
    "administrative policy update":      "📋",
}

POPULAR_CODES = [
    ("99213", "Office Visit E&M"),
    ("99214", "Office Visit E&M"),
    ("99215", "Complex E&M"),
    ("G0439", "Medicare Wellness"),
    ("74178", "CT Abdomen/Pelvis"),
    ("71250", "CT Thorax"),
    ("93000", "ECG"),
    ("99204", "New Patient E&M"),
    ("G2212", "Telehealth"),
    ("A4253", "Blood Glucose"),
]

JUNK_TITLES = {
    "submit feedback", "ask a question", "alphabetical index",
    "search criteria", "search results", "ncd alphabetical",
    "medicare coverage database", "cms.gov home", "addendum index",
    "fee schedule index", "national coverage", "coverage database",
}

JUNK_URL_PARTS = [
    "alphabetical-index", "indexes/", "search-criteria",
    "search/search", "#ncd", "feedback", "ask-a-question",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

html,body,[class*="css"]{{font-family:'Inter',sans-serif!important;background:{BG};color:{TEXT}}}
.block-container{{padding:1rem 1.6rem;max-width:1600px;padding-top:3rem}}

/* KPI */
.kpi{{background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:16px;text-align:center;height:100%}}
.kpi-v{{font-size:2rem;font-weight:600;font-family:'IBM Plex Mono',monospace;line-height:1.1}}
.kpi-l{{font-size:.6rem;font-weight:600;letter-spacing:.18em;text-transform:uppercase;color:{DIMMED};margin-top:5px}}
.kpi-s{{font-size:.68rem;color:{MUTED};margin-top:3px}}

/* Policy change card */
.pc{{background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
     padding:16px 18px;margin-bottom:10px}}
.pc.sev-crit{{border-left:3px solid {SEV_COLOR["CRITICAL"]}}}
.pc.sev-high{{border-left:3px solid {SEV_COLOR["HIGH"]}}}
.pc.sev-med {{border-left:3px solid {SEV_COLOR["MEDIUM"]}}}
.pc.sev-low {{border-left:3px solid {SEV_COLOR["LOW"]}}}

.pc-type  {{font-size:.65rem;color:{MUTED};margin-bottom:5px;letter-spacing:.02em}}
.pc-head  {{font-size:.88rem;font-weight:600;color:{TEXT};margin-bottom:7px;line-height:1.45}}
.pc-body  {{font-size:.76rem;color:{MUTED};line-height:1.7;margin-bottom:10px}}
.pc-action{{background:#080e1c;border-left:2px solid #2563eb;border-radius:0 5px 5px 0;
            padding:7px 12px;font-size:.73rem;color:#60a5fa;line-height:1.5}}
.pc-action b{{color:#93c5fd}}
.pc-footer{{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}}

/* Severity badge */
.sev-badge{{display:inline-block;padding:2px 9px;border-radius:99px;
            font-size:.63rem;font-weight:600;letter-spacing:.03em}}
/* Urgency tag */
.urg-tag{{display:inline-block;padding:2px 8px;border-radius:4px;
          font-size:.62rem;font-weight:600}}
.urg-imm{{background:#2d0f0f;color:#fca5a5}}
.urg-urg{{background:#2d1900;color:#fdba74}}
.urg-rtn{{background:#0d2b16;color:#86efac}}

/* Code chip */
.chip{{font-family:'IBM Plex Mono',monospace;background:#111827;color:#60a5fa;
       font-size:.63rem;padding:2px 7px;border-radius:4px;display:inline-block;margin:1px}}

/* Quick code button */
.qbtn{{background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
       padding:8px 10px;text-align:center;cursor:pointer;transition:border-color .15s}}
.qbtn:hover{{border-color:#2563eb}}
.qbtn-code{{font-family:'IBM Plex Mono',monospace;font-size:.8rem;font-weight:500;color:#60a5fa}}
.qbtn-desc{{font-size:.6rem;color:{MUTED};margin-top:2px}}

/* Section label */
.sl{{font-size:.58rem;font-weight:700;letter-spacing:.22em;text-transform:uppercase;
     color:{DIMMED};padding-bottom:7px;border-bottom:1px solid {BORDER};margin-bottom:13px}}

/* Streamlit overrides */
.stButton>button{{background:linear-gradient(135deg,#3730a3,#6d28d9);color:#fff;
    border:none;border-radius:8px;font-weight:600;padding:9px 18px;font-family:'Inter',sans-serif}}
.stButton>button:hover{{opacity:.88}}
div[data-testid="metric-container"]{{background:{SURFACE};border:1px solid {BORDER};border-radius:9px;padding:10px}}
.stSidebar,[data-testid="stSidebarContent"]{{background:#060810!important}}
.stTextInput>div>input{{background:{SURFACE}!important;color:{TEXT}!important;border:1px solid {BORDER}!important}}
.stSelectbox>div>div{{background:{SURFACE}!important;color:{TEXT}!important;border:1px solid {BORDER}!important}}
div[data-testid="stExpander"]{{background:{SURFACE};border:1px solid {BORDER};border-radius:8px}}
</style>
""", unsafe_allow_html=True)

# API
def _get(path: str, params: dict = None) -> Optional[Any]:
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params or {}, timeout=6.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def _post(path: str, payload: dict) -> Optional[Any]:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=payload, timeout=10.0)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=CACHE_TTL)
def api_stats():    return _get("/dashboard/stats") or {}
@st.cache_data(ttl=CACHE_TTL)
def api_changes(h, s=""):
    p = {"hours": h, "limit": 50}
    if s: p["severity"] = s
    return (_get("/policies/changes/overview", p) or {}).get("changes", [])
@st.cache_data(ttl=CACHE_TTL)
def api_alerts(s=""):
    return (_get("/alerts/active", {"severity": s} if s else {}) or {}).get("alerts", [])

# Helpers
def is_junk(ch: dict) -> bool:
    headline = (ch.get("headline") or ch.get("title") or "").strip().lower()
    url      = (ch.get("source_url") or "").lower()
    overview = (ch.get("overview") or ch.get("summary") or "").strip()
    if not headline or len(headline) < 8:
        return True
    if any(j in headline for j in JUNK_TITLES):
        return True
    if any(p in url for p in JUNK_URL_PARTS):
        return True
    if len(overview) < 30:
        return True
    return False

def urg_cls(u: str) -> str:
    ul = u.lower()
    if "immediate" in ul: return "urg-imm"
    if "hour" in ul:      return "urg-urg"
    return "urg-rtn"

def sev_badge_html(sev: str) -> str:
    c = SEV_COLOR.get(sev, "#60a5fa")
    bg = SEV_BG.get(sev, "#0d1c32")
    return f'<span class="sev-badge" style="background:{bg};color:{c}">{sev}</span>'

def chips_html(codes: list) -> str:
    if not codes:
        return ""
    top = codes[:7]
    html = "".join(f'<span class="chip">{c}</span>' for c in top)
    if len(codes) > 7:
        html += f'<span class="chip">+{len(codes)-7}</span>'
    return f'<div style="margin-top:8px">{html}</div>'

def render_card(ch: dict):
    if is_junk(ch):
        return

    sev      = ch.get("severity", "LOW")
    codes    = ch.get("codes_affected") or ch.get("billing_codes") or []
    ug       = ch.get("urgency") or "Routine"
    ct       = ch.get("what_changed") or ch.get("change_type") or ""
    icon     = CHANGE_ICON.get(ct.lower(), "📋")
    eff      = ch.get("effective_date") or ""
    url      = ch.get("source_url") or ""
    headline = (ch.get("headline") or ch.get("title") or "Policy change detected").strip()
    overview = (ch.get("overview") or ch.get("summary") or "").strip()
    action   = (ch.get("action") or ch.get("action_required") or "Review policy").strip()

    # Build HTML piecemeal — avoids f-string conditional rendering bugs
    eff_badge  = f'<span class="sev-badge" style="background:#062624;color:#5eead4">Eff: {eff}</span>' if eff else ""
    src_link   = (f'<div style="margin-top:8px"><a href="{url}" target="_blank" '
                  f'style="font-size:.6rem;color:{DIMMED};text-decoration:none">'
                  f'{url[:65]}…</a></div>') if url else ""
    codes_part = chips_html(codes)

    html = f"""
<div class="pc {SEV_CLS.get(sev, 'sev-low')}">
  <div class="pc-type">{icon}&nbsp; {ct}</div>
  <div class="pc-head">{headline}</div>
  <div class="pc-body">{overview}</div>
  <div class="pc-action"><b>Action:</b>&nbsp;{action}</div>
  <div class="pc-footer">
    <span class="urg-tag {urg_cls(ug)}">{ug}</span>
    {sev_badge_html(sev)}
    {eff_badge}
  </div>
  {codes_part}
  {src_link}
</div>"""
    st.markdown(html, unsafe_allow_html=True)

def section(label: str):
    st.markdown(f'<div class="sl">{label}</div>', unsafe_allow_html=True)

def kpi(col, label: str, value, color: str = "#60a5fa", sub: str = ""):
    sub_html = f'<div class="kpi-s">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi"><div class="kpi-v" style="color:{color}">{value}</div>'
        f'<div class="kpi-l">{label}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )

# Charts
_L = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color=MUTED, size=10, family="Inter"),
          margin=dict(t=12, b=12, l=12, r=12))

def _empty(msg="No data yet"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=.5, y=.5, showarrow=False,
                       font=dict(color=DIMMED, size=11))
    fig.update_layout(**_L, height=160)
    return fig

def chart_donut(cs: dict) -> go.Figure:
    cmap = {"submitted":"#60a5fa","approved":"#4ade80",
            "flagged":"#f87171","pending_review":"#fb923c"}
    labels = [k.replace("_"," ").title() for k in cs]
    vals   = [v.get("count",0) for v in cs.values()]
    colors = [cmap.get(k, DIMMED) for k in cs]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=.60,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textfont=dict(color=TEXT, size=9),
    ))
    fig.update_layout(**_L, legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"), height=200)
    return fig

def chart_sev_bars(changes: list) -> go.Figure:
    if not changes: return _empty()
    order  = ["CRITICAL","HIGH","MEDIUM","LOW"]
    counts = Counter(c.get("severity","LOW") for c in changes)
    fig = go.Figure(go.Bar(
        x=order, y=[counts.get(s,0) for s in order],
        marker_color=[SEV_COLOR[s] for s in order],
        text=[counts.get(s,0) for s in order],
        textposition="outside", textfont=dict(color=MUTED, size=10),
        marker_line_width=0,
    ))
    fig.update_layout(**_L, height=170, bargap=.45,
        xaxis=dict(tickfont=dict(size=9, color=MUTED), showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig

def chart_code_risk(changes: list) -> go.Figure:
    code_data: dict = {}
    for c in changes:
        sev  = c.get("severity","LOW")
        risk = {"CRITICAL":.88,"HIGH":.70,"MEDIUM":.45,"LOW":.20}.get(sev,.20)
        for code in (c.get("codes_affected") or [])[:8]:
            if risk > code_data.get(code, {}).get("risk", 0):
                code_data[code] = {"risk": risk, "sev": sev}
    if not code_data:
        return _empty("No code risk data")
    top    = sorted(code_data.items(), key=lambda x: -x[1]["risk"])[:14]
    codes  = [t[0] for t in top]
    risks  = [t[1]["risk"] for t in top]
    colors = [SEV_COLOR[t[1]["sev"]] for t in top]
    fig = go.Figure(go.Bar(
        x=risks, y=codes, orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{r:.0%}" for r in risks], textposition="outside",
        textfont=dict(color=MUTED, size=9),
        hovertemplate="<b>%{y}</b>  risk: %{x:.0%}<extra></extra>",
    ))
    fig.update_layout(**_L, height=max(180, len(top)*26),
        xaxis=dict(tickformat=".0%", range=[0,1.15],
                   tickfont=dict(size=9,color=MUTED), showgrid=False),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono",size=10,color="#60a5fa"),
                   gridcolor=BORDER))
    return fig

def chart_top_codes(top_codes: list) -> go.Figure:
    if not top_codes: return _empty()
    df = pd.DataFrame(top_codes).head(10)
    fig = go.Figure(go.Bar(
        x=df["policy_count"], y=df["code"], orientation="h",
        marker=dict(color=df["policy_count"],
                    colorscale=[[0,"#1e3a5f"],[.5,"#2563eb"],[1,"#f87171"]]),
        text=df["policy_count"], textposition="outside",
        textfont=dict(color=MUTED, size=9),
    ))
    fig.update_layout(**_L, height=220,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(family="IBM Plex Mono",size=10,color="#60a5fa"),
                   gridcolor=BORDER))
    return fig

# Regulatory entropy
def entropy_score(changes: list, alerts: list) -> float:
    crit = sum(1 for c in changes if c.get("severity")=="CRITICAL")
    high = sum(1 for c in changes if c.get("severity")=="HIGH")
    uc   = sum(1 for a in alerts  if a.get("severity")=="CRITICAL")
    return min(100.0, round(crit*15 + high*7 + min(len(changes),20)*2 + uc*10, 1))

def entropy_tier(score: float):
    if score >= 75: return "Critical entropy",  "#f87171"
    if score >= 50: return "High entropy",       "#fb923c"
    if score >= 25: return "Moderate",           "#fbbf24"
    return               "Stable",              "#4ade80"

# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
<div style="text-align:center;padding:14px 0 16px">
  <div style="font-size:1.8rem">🏥</div>
  <div style="font-size:.92rem;font-weight:600;color:#818cf8;margin-top:4px">CareClaimAI</div>
  <div style="font-size:.56rem;color:{DIMMED};letter-spacing:.18em;margin-top:2px">COMPLIANCE INTELLIGENCE</div>
</div>""", unsafe_allow_html=True)

        page = st.radio("", [
            "📊  Dashboard",
            "📋  Policy Changes",
            "💊  Claim Validator",
            "🔎  Code Lookup",
            "⚠️   Alerts",
        ], label_visibility="collapsed")

        st.markdown("---")
        hours = st.selectbox("Time window", [24,48,72,168], index=1,
                             format_func=lambda h: f"Last {h} hours")
        sev_f = st.selectbox("Severity", ["All","CRITICAL","HIGH","MEDIUM","LOW"])

        st.markdown("---")
        h     = _get("/health") or {}
        ok    = h.get("status") == "healthy"
        db_ok = h.get("mongodb") == "ok"
        rd_ok = h.get("redis")   == "ok"

        def dot(up): return f'<span style="color:{"#4ade80" if up else "#f87171"}">●</span>'
        st.markdown(f"""
<div style="font-size:.67rem;line-height:2.3;color:{MUTED}">
  {dot(ok)} API &nbsp;·&nbsp;
  {dot(db_ok)} MongoDB &nbsp;·&nbsp;
  {dot(rd_ok)} Redis
</div>""", unsafe_allow_html=True)

        if not ok:
            st.warning("API offline\n`python run_platform.py run`")

    return page, hours, "" if sev_f == "All" else sev_f

# ── Pages ─────────────────────────────────────────────────────────────────────
def page_dashboard(hours: int, sev: str):
    changes = [c for c in api_changes(hours, sev) if not is_junk(c)]
    alerts  = api_alerts(sev)
    stats   = api_stats()
    cs      = stats.get("claim_stats", {})
    total   = sum(v.get("count",0) for v in cs.values())
    flagged = cs.get("flagged",{}).get("count",0)
    crit_a  = stats.get("critical_alerts",0)
    ent     = entropy_score(changes, alerts)
    e_label, e_color = entropy_tier(ent)

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    kpi(k1, "Claims today",     f"{total:,}",  "#60a5fa")
    kpi(k2, "Flagged",          f"{flagged:,}",
        "#f87171" if flagged>30 else "#fb923c" if flagged>5 else "#4ade80",
        sub=f"{flagged/max(total,1):.1%} flag rate")
    kpi(k3, "Policy changes",   len(changes),  "#fb923c",
        sub=f"last {hours}h")
    kpi(k4, "Active alerts",    len(alerts),
        "#f87171" if crit_a>0 else "#fb923c" if alerts else "#4ade80",
        sub=f"{crit_a} critical" if crit_a else "none critical")

    st.markdown("<br>", unsafe_allow_html=True)

    col_feed, col_right = st.columns([1.65, 1])

    with col_feed:
        section("Policy change intelligence feed")
        if not changes:
            st.info("No policy changes detected. Run:\n`python run_platform.py run --initial-scan`")
        else:
            for ch in changes[:5]:
                render_card(ch)
            if len(changes) > 5:
                st.caption(f"+ {len(changes)-5} more — open **Policy Changes** page")

    with col_right:
        # Entropy widget
        bar_pct = int(ent)
        st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
            padding:18px;text-align:center;margin-bottom:16px">
  <div style="font-size:.57rem;font-weight:700;letter-spacing:.18em;
              text-transform:uppercase;color:{DIMMED};margin-bottom:8px">
    Regulatory Entropy Score
  </div>
  <div style="font-size:2.6rem;font-weight:600;font-family:'IBM Plex Mono',monospace;
              color:{e_color};line-height:1">{ent}</div>
  <div style="font-size:.68rem;font-weight:600;color:{e_color};margin:5px 0 12px">{e_label}</div>
  <div style="background:{BORDER};border-radius:4px;height:5px;overflow:hidden">
    <div style="width:{bar_pct}%;height:5px;background:{e_color};border-radius:4px"></div>
  </div>
  <div style="font-size:.6rem;color:{DIMMED};margin-top:8px;line-height:1.6">
    Measures change velocity × severity weight.<br>Higher = more fragmented policy landscape.
  </div>
</div>""", unsafe_allow_html=True)

        section("Claim pipeline")
        if cs:
            st.plotly_chart(chart_donut(cs), use_container_width=True)

        section("Severity breakdown")
        st.plotly_chart(chart_sev_bars(changes), use_container_width=True)


def page_policy_changes(hours: int, sev: str):
    changes = [c for c in api_changes(hours, sev) if not is_junk(c)]

    if not changes:
        st.info("No policy changes found. Run `python run_platform.py run --initial-scan`")
        return

    counts = Counter(c.get("severity","LOW") for c in changes)
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1,"Critical", counts.get("CRITICAL",0), "#f87171")
    kpi(c2,"High",     counts.get("HIGH",0),     "#fb923c")
    kpi(c3,"Medium",   counts.get("MEDIUM",0),   "#fbbf24")
    kpi(c4,"Low",      counts.get("LOW",0),       "#4ade80")

    st.markdown("<br>", unsafe_allow_html=True)

    col_feed, col_risk = st.columns([1.6, 1])

    with col_feed:
        section(f"{len(changes)} change(s) detected — last {hours}h")
        crit_only = st.checkbox("Show CRITICAL only", False)
        show = [c for c in changes if c.get("severity")=="CRITICAL"] if crit_only else changes
        for ch in show:
            render_card(ch)

    with col_risk:
        section("Billing code rejection risk")
        st.plotly_chart(chart_code_risk(changes), use_container_width=True)
        st.caption("Bar = estimated rejection risk by policy severity.")


def page_claim_validator():
    st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
            padding:13px 18px;margin-bottom:16px;font-size:.76rem;color:{MUTED};line-height:1.7">
  Enter claim details below. Every code is validated against the live CMS policy database.
  Returns a rejection risk score and the specific policy change driving each risk
  — before the claim reaches the payer.
</div>""", unsafe_allow_html=True)

    with st.form("clm_form"):
        r1, r2, r3 = st.columns(3)
        pid = r1.text_input("Patient ID",   "PT-100234")
        npi = r2.text_input("Provider NPI", "1234567890")
        svc = r3.text_input("Service date", datetime.now().strftime("%Y-%m-%d"))

        r4, r5, r6, r7 = st.columns([1.2,1.8,.6,.8])
        cpt = r4.text_input("CPT code",           "99213")
        icd = r5.text_input("ICD-10 (comma sep)", "E11.9,Z00.00")
        unt = r6.number_input("Units", 1, 99, 1)
        mod = r7.text_input("Modifier", "")

        r8, r9 = st.columns(2)
        pa  = r8.text_input("Prior auth number", "")
        pos = r9.selectbox("Place of service",
                           ["11 - Office","21 - Inpatient","22 - Outpatient",
                            "02 - Telehealth","23 - ER"])
        go_btn = st.form_submit_button("Validate against live policy database",
                                       use_container_width=True)

    if not go_btn:
        return

    payload = {
        "patient_id": pid, "provider_npi": npi, "service_date": svc,
        "service_lines": [{"cpt_code": cpt.strip(),
            "icd10_codes": [x.strip() for x in icd.split(",") if x.strip()],
            "units": int(unt), "modifier": mod or None, "charge_amount": 0.0,
            "place_of_service": pos.split("-")[0].strip()}],
        "diagnosis_codes": [x.strip() for x in icd.split(",") if x.strip()],
        "prior_auth_number": pa or None,
    }
    with st.spinner("Checking…"):
        result = _post("/claims/validate", payload)

    if not result or "error" in result:
        st.error(result.get("error","API offline") if result else "API offline")
        return

    status = result.get("overall_status","UNKNOWN")
    risk   = result.get("overall_risk_score", 0)
    s_col  = {"APPROVED":"#4ade80","REVIEW_REQUIRED":"#fbbf24","HIGH_RISK":"#f87171"}.get(status,"#60a5fa")
    icon   = {"APPROVED":"✅","REVIEW_REQUIRED":"⚠️","HIGH_RISK":"🚫"}.get(status,"—")

    st.markdown("<br>", unsafe_allow_html=True)
    rc1,rc2,rc3 = st.columns(3)
    kpi(rc1,"Claim status",    f"{icon} {status}", s_col)
    kpi(rc2,"Rejection risk",  f"{risk:.0%}",
        "#f87171" if risk>.6 else "#fb923c" if risk>.3 else "#4ade80")
    kpi(rc3,"Action needed",
        "Yes — review" if status!="APPROVED" else "None",
        "#f87171" if status!="APPROVED" else "#4ade80")

    for a in result.get("recommended_actions",[]): st.warning(a)

    section("Service line detail")
    for sl in result.get("service_line_results",[]):
        ok_sl = sl.get("valid", False)
        with st.expander(f"{'✅' if ok_sl else '❌'}  Code {sl.get('code')}  —  "
                         f"Rejection risk: {sl.get('rejection_risk',0):.0%}"):
            for i in sl.get("issues",[]): st.error(i)
            for w in sl.get("warnings",[]): st.warning(w)
            for r in sl.get("recommendations",[]): st.info(r)

    cr = _get(f"/billing-codes/{cpt.strip().upper()}/check")
    if cr and cr.get("recent_policies"):
        section("Why this code is flagged — active policy context")
        from nlp.policy_summarizer import summarise_policy
        for pol in cr["recent_policies"][:2]:
            s = summarise_policy(pol)
            render_card({"severity":s.severity,"urgency":s.urgency,
                         "headline":s.headline,"overview":s.overview,
                         "action":s.action,"what_changed":s.what_changed,
                         "codes_affected":s.codes_affected,
                         "effective_date":s.effective_date,"source_url":s.source_url})


def page_code_lookup():
    # Intro
    st.markdown(f"""
<div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
            padding:13px 18px;margin-bottom:16px;font-size:.76rem;color:{MUTED};line-height:1.7">
  Look up any CPT, HCPCS, or ICD-10 code to see every active CMS policy change
  affecting it, its current rejection risk, and the required action.
</div>""", unsafe_allow_html=True)

    # Popular codes grid
    section("Popular codes — click to check instantly")
    rows = [POPULAR_CODES[:5], POPULAR_CODES[5:]]
    chosen = None
    for row in rows:
        cols = st.columns(5)
        for col, (qc, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
<div class="qbtn" id="qb_{qc}">
  <div class="qbtn-code">{qc}</div>
  <div class="qbtn-desc">{desc}</div>
</div>""", unsafe_allow_html=True)
                if st.button(qc, key=f"qk_{qc}", use_container_width=True):
                    chosen = qc

    st.markdown("<br>", unsafe_allow_html=True)
    section("Or enter any code")
    code = st.text_input("", value=chosen or "",
                          placeholder="e.g. 99213, G0439, E11.9, J0702",
                          label_visibility="collapsed")

    if not code:
        return

    code = code.strip().upper()
    with st.spinner(f"Checking {code}…"):
        r = _get(f"/billing-codes/{code}/check")

    if not r:
        st.error("API offline or code not found")
        return

    status = r.get("status","UNKNOWN")
    risk   = r.get("max_rejection_risk", 0)
    s_col  = {"CLEAR":"#4ade80","REVIEW":"#fbbf24","HIGH_RISK":"#f87171"}.get(status,"#60a5fa")

    st.markdown("<br>", unsafe_allow_html=True)
    r1,r2,r3,r4 = st.columns(4)
    kpi(r1,"Policy status",    status,                         s_col)
    kpi(r2,"Policies found",   r.get("policies_found",0),     "#60a5fa")
    kpi(r3,"Active alerts",    r.get("active_alerts",0),      "#fb923c")
    kpi(r4,"Rejection risk",   f"{risk:.0%}",                  s_col)

    for iss in r.get("issues",[]): st.error(iss)

    if r.get("recent_policies"):
        section("Active policy changes for this code")
        from nlp.policy_summarizer import summarise_policy
        for pol in r["recent_policies"][:4]:
            s = summarise_policy(pol)
            render_card({"severity":s.severity,"urgency":s.urgency,
                         "headline":s.headline,"overview":s.overview,
                         "action":s.action,"what_changed":s.what_changed,
                         "codes_affected":s.codes_affected,
                         "effective_date":s.effective_date,"source_url":s.source_url})
    else:
        st.success(f"**{code}** — no active policy alerts. Clear to bill.")


def page_alerts(sev: str):
    alerts = api_alerts(sev)
    counts = Counter(a.get("severity","LOW") for a in alerts)

    c1,c2,c3 = st.columns(3)
    kpi(c1,"Critical unresolved", counts.get("CRITICAL",0), "#f87171")
    kpi(c2,"High unresolved",     counts.get("HIGH",0),     "#fb923c")
    kpi(c3,"Total active",        len(alerts),               "#60a5fa")

    st.markdown("<br>", unsafe_allow_html=True)

    if not alerts:
        st.success("No active alerts — compliance is current.")
        return

    section(f"{len(alerts)} unresolved alert(s)")

    for alert in alerts:
        sev_a  = alert.get("severity","LOW")
        risk   = alert.get("rejection_risk",0)
        codes  = alert.get("affected_codes",[])
        agent  = alert.get("agent_action","PENDING")

        with st.expander(
            f"{SEV_EMOJI.get(sev_a,'')}"
            f"  {alert.get('title','')[:72]}"
            f"  ·  Risk {risk:.0%}"
            f"  ·  {agent}"
        ):
            m1,m2,m3 = st.columns(3)
            m1.metric("Severity",       sev_a)
            m2.metric("Rejection risk", f"{risk:.0%}")
            m3.metric("Agent decision", agent)

            if codes:
                st.markdown("**Affected billing codes**")
                st.code(", ".join(codes[:15]))

            st.markdown("**Required action**")
            st.info(alert.get("action_required","Review policy change"))

            wf = alert.get("workflow_updates",[])
            if wf:
                st.markdown("**Billing system workflow updates**")
                for w in wf:
                    st.markdown(
                        f"`{w.get('update_type')}` → code `{w.get('code')}` "
                        f"via rule `{w.get('rule','')}`"
                    )

            rej = alert.get("rejection_codes",[])
            if rej:
                st.markdown(f"**CMS reason codes:** `{'`, `'.join(rej)}`")

            if st.button("Mark resolved", key=f"res_{alert.get('_id','')}"):
                _post(f"/alerts/{alert.get('_id','')}/resolve",
                      {"resolution_note": "Resolved via dashboard"})
                st.success("Alert resolved")
                st.cache_data.clear()
                st.rerun()

# Main
def main():
    # Sidebar must render first — otherwise the header overlaps the nav rail
    page, hours, sev = sidebar()

    st.markdown(f"""
<div style="background:linear-gradient(90deg,{SURFACE},#131040,{SURFACE});
            border:1px solid #2a2460;border-radius:10px;
            padding:14px 22px;margin-bottom:18px;
            display:flex;align-items:center;gap:14px">
  <div style="font-size:1.9rem;flex-shrink:0;line-height:1">🏥</div>
  <div>
    <div style="font-size:1.22rem;font-weight:600;
                background:linear-gradient(90deg,#818cf8,#60a5fa,#2dd4bf);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                letter-spacing:-.01em">
      CareClaimAI &mdash; Compliance Intelligence
    </div>
    <div style="font-size:.57rem;color:{DIMMED};letter-spacing:.13em;margin-top:3px">
      🔍 AUTONOMOUS CMS MONITORING &nbsp;·&nbsp;
      ⚡ REAL-TIME POLICY INTELLIGENCE &nbsp;·&nbsp;
      🛡 CLAIM REJECTION PREVENTION
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    if   "Dashboard"  in page: page_dashboard(hours, sev)
    elif "Changes"    in page: page_policy_changes(hours, sev)
    elif "Validator"  in page: page_claim_validator()
    elif "Lookup"     in page: page_code_lookup()
    elif "Alerts"     in page: page_alerts(sev)


if __name__ == "__main__":
    main()

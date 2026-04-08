"""
app.py
------
Main Streamlit application for the AI Legal Document Analyzer.
Run with: streamlit run app.py
"""

import streamlit as st

from extractor import extract_text_from_pdf, get_page_count
from analyzer import analyze_document, analyze_pdf, calculate_risk_score
from utils import clean_text, word_count, truncate_text, clause_icon


# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# CUSTOM CSS — Professional Dark Legal Theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: #0d0f14;
        color: #e8e6e1;
    }

    /* ── Main content padding ── */
    .main .block-container {
        padding: 2rem 3rem 4rem 3rem;
        max-width: 1100px;
    }

    /* ── Hero Header ── */
    .hero-header {
        background: linear-gradient(135deg, #1a1d25 0%, #12141c 60%, #1e1425 100%);
        border: 1px solid #2a2d3a;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #c9a84c, #e8c97a, #c9a84c);
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: #e8c97a;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #8b8fa8;
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .hero-badge {
        display: inline-block;
        background: #1e2530;
        border: 1px solid #2a3040;
        color: #c9a84c;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 1rem;
    }

    /* ── Section Cards ── */
    .section-card {
        background: #13151e;
        border: 1px solid #1e2230;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        color: #e8c97a;
        margin: 0 0 1rem 0;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #1e2230;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Summary Box ── */
    .summary-text {
        font-size: 0.97rem;
        line-height: 1.8;
        color: #c8c5be;
        background: #0f1118;
        border-left: 3px solid #c9a84c;
        padding: 1.2rem 1.5rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
    }

    /* ── Entity Pills ── */
    .entity-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }
    .entity-pill {
        background: #1a1f2e;
        border: 1px solid #2a3048;
        color: #9eb3d8;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .entity-pill.org {
        background: #1a2a1e;
        border-color: #2a4830;
        color: #7ec89a;
    }

    /* ── Clause Card ── */
    .clause-card {
        background: #0f1118;
        border: 1px solid #1e2230;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .clause-card:hover {
        border-color: #c9a84c40;
    }
    .clause-card.found {
        border-left: 3px solid #c9a84c;
    }
    .clause-card.not-found {
        opacity: 0.45;
        border-left: 3px solid #2a2d3a;
    }
    .clause-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: #d4c9a8;
        margin-bottom: 0.5rem;
    }
    .clause-text {
        font-size: 0.83rem;
        color: #8b8fa8;
        line-height: 1.6;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ── Risk Cards ── */
    .risk-card {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        border: 1px solid transparent;
    }
    .risk-card.high {
        background: #1e0f0f;
        border-color: #4a1515;
    }
    .risk-card.medium {
        background: #1e1a0a;
        border-color: #4a3a10;
    }
    .risk-card.low {
        background: #0f1e12;
        border-color: #15401a;
    }
    .risk-label {
        font-size: 0.88rem;
        font-weight: 500;
        flex: 1;
    }
    .risk-label.high   { color: #f87171; }
    .risk-label.medium { color: #fbbf24; }
    .risk-label.low    { color: #4ade80; }
    .risk-keywords {
        font-size: 0.73rem;
        color: #6b7280;
        margin-top: 3px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .risk-badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .risk-badge.high   { background: #4a1515; color: #f87171; }
    .risk-badge.medium { background: #4a3a10; color: #fbbf24; }
    .risk-badge.low    { background: #15401a; color: #4ade80; }

    /* ── Score Meter ── */
    .score-container {
        text-align: center;
        padding: 1.5rem;
    }
    .score-number {
        font-family: 'Playfair Display', serif;
        font-size: 5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .score-number.low    { color: #4ade80; }
    .score-number.medium { color: #fbbf24; }
    .score-number.high   { color: #f87171; }
    .score-category {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .score-category.low    { color: #4ade80; }
    .score-category.medium { color: #fbbf24; }
    .score-category.high   { color: #f87171; }

    /* ── Progress bar override ── */
    .stProgress > div > div > div {
        border-radius: 4px;
    }

    /* ── Stats strip ── */
    .stat-strip {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        flex: 1;
        background: #13151e;
        border: 1px solid #1e2230;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #e8c97a;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    /* ── Upload area ── */
    .upload-hint {
        text-align: center;
        color: #4a4f60;
        font-size: 0.88rem;
        padding: 1rem;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #1e2230;
        margin: 1.5rem 0;
    }

    /* ── Streamlit overrides ── */
    .stFileUploader {
        background: #13151e !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] label {
        color: #8b8fa8 !important;
    }
    .stSpinner > div {
        border-top-color: #c9a84c !important;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚖️ AI-Powered · Gemini Analysis</div>
    <div class="hero-title">Legal Document Analyzer</div>
    <div class="hero-subtitle">
        Instantly extract clauses, detect risks, and understand any legal contract —
        powered by Google Gemini with structured legal analysis.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECTION 1 — FILE UPLOAD
# ---------------------------------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📂 Upload Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a PDF contract or legal document here",
    type=["pdf"],
    help="Supports any PDF: contracts, NDAs, agreements, terms of service, etc.",
    label_visibility="visible",
)

if not uploaded_file:
    st.markdown(
        '<div class="upload-hint">Supports PDF files · Requires a valid Gemini API key in .env</div>',
        unsafe_allow_html=True,
    )

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ANALYSIS (only runs when a file is uploaded)
# ---------------------------------------------------------------------------
if uploaded_file:

    with st.spinner("Analyzing document…"):

        # ── 1. Extract local stats if possible ───────────────────────────
        raw_text = extract_text_from_pdf(uploaded_file)
        text = clean_text(raw_text)
        uploaded_file.seek(0)
        pages = get_page_count(uploaded_file)

        # ── 2. Run Gemini on extracted text when available; fall back to PDF ──
        try:
            if text.strip():
                analysis = analyze_document(text)
            else:
                uploaded_file.seek(0)
                analysis = analyze_pdf(uploaded_file)
        except Exception as exc:
            st.error(f"⚠️ Gemini analysis failed: {exc}")
            st.stop()

        summary     = analysis["summary"]
        entities    = analysis["entities"]
        clauses     = analysis["clauses"]
        risks       = analysis["risks"]
        score_data  = calculate_risk_score(risks)

    # ── STATS STRIP ────────────────────────────────────────────────────
    found_clauses = len(clauses)
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-box">
            <div class="stat-value">{pages}</div>
            <div class="stat-label">Pages</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{word_count(text):,}</div>
            <div class="stat-label">Words</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{found_clauses}</div>
            <div class="stat-label">Clauses Found</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(risks)}</div>
            <div class="stat-label">Risks Flagged</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{score_data['score']}%</div>
            <div class="stat-label">Risk Score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2 — SUMMARY ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Document Summary</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-text">{summary}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 3 — KEY INFORMATION (Entities) ───────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Key Information</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**👤 Persons Mentioned**")
        if entities["persons"]:
            pills = "".join(
                f'<span class="entity-pill">{p}</span>'
                for p in entities["persons"][:15]
            )
            st.markdown(f'<div class="entity-pills">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#4a4f60;font-size:0.85rem;">No persons identified</span>',
                        unsafe_allow_html=True)

    with col_b:
        st.markdown("**🏢 Organizations Mentioned**")
        if entities["organizations"]:
            pills = "".join(
                f'<span class="entity-pill org">{o}</span>'
                for o in entities["organizations"][:15]
            )
            st.markdown(f'<div class="entity-pills">{pills}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#4a4f60;font-size:0.85rem;">No organizations identified</span>',
                        unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 4 — CLAUSES ──────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Clause Detection</div>', unsafe_allow_html=True)

    if not clauses:
        st.markdown(
            '<p style="color:#8b8fa8;font-size:0.95rem;">No major clauses were confidently extracted.</p>',
            unsafe_allow_html=True,
        )
    else:
        for clause in clauses:
            icon = clause_icon(clause["name"])
            plain = truncate_text(clause["plain_language"], 220)
            excerpt = truncate_text(clause["excerpt"], 280) if clause["excerpt"] else ""
            excerpt_html = (
                f'<div class="risk-keywords">Source text: {excerpt}</div>'
                if excerpt
                else ""
            )
            st.markdown(f"""
            <div class="clause-card found">
                <div class="clause-name">{icon} {clause["name"]}</div>
                <div class="clause-text">{plain}</div>
                {excerpt_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 5 — RISKS ────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Risk Detection</div>', unsafe_allow_html=True)

    if not risks:
        st.markdown(
            '<p style="color:#4ade80;font-size:0.95rem;">✅ No significant risks detected in this document.</p>',
            unsafe_allow_html=True,
        )
    else:
        # Group by severity
        for sev in ("high", "medium", "low"):
            group = [r for r in risks if r["severity"] == sev]
            if not group:
                continue
            label_map = {"high": "🔴 High Severity", "medium": "🟡 Medium Severity", "low": "🟢 Low Severity"}
            st.markdown(f"**{label_map[sev]}**")
            for risk in group:
                kws = ", ".join(risk["found_keywords"])
                st.markdown(f"""
                <div class="risk-card {sev}">
                    <div style="flex:1">
                        <div class="risk-label {sev}">{risk['label']}</div>
                        <div class="clause-text">{truncate_text(risk['plain_language'], 220)}</div>
                        <div class="risk-keywords">Source text: {kws}</div>
                    </div>
                    <span class="risk-badge {sev}">{sev}</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── SECTION 6 — RISK SCORE ───────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Overall Risk Score</div>', unsafe_allow_html=True)

    score    = score_data["score"]
    category = score_data["category"]
    color_cls = (
        "low"    if score <= 30 else
        "medium" if score <= 70 else
        "high"
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown(f"""
        <div class="score-container">
            <div class="score-number {color_cls}">{score}%</div>
            <div class="score-category {color_cls}">{category}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)

        # Progress bar colored by risk
        bar_color = "#4ade80" if color_cls == "low" else "#fbbf24" if color_cls == "medium" else "#f87171"
        st.markdown(f"""
        <style>
        .stProgress > div > div > div > div {{
            background: {bar_color} !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        st.progress(score / 100)

        # Breakdown
        high_count   = sum(1 for r in risks if r["severity"] == "high")
        medium_count = sum(1 for r in risks if r["severity"] == "medium")
        low_count    = sum(1 for r in risks if r["severity"] == "low")

        st.markdown(f"""
        <div style="margin-top:1rem;display:flex;gap:1.5rem;font-size:0.85rem;">
            <span>🔴 <b style="color:#f87171">{high_count}</b> High</span>
            <span>🟡 <b style="color:#fbbf24">{medium_count}</b> Medium</span>
            <span>🟢 <b style="color:#4ade80">{low_count}</b> Low</span>
        </div>
        <div style="margin-top:1rem;font-size:0.8rem;color:#6b7280;line-height:1.6;">
            Score ranges:
            0–30% = Low &nbsp;|&nbsp; 31–70% = Medium &nbsp;|&nbsp; 71–100% = High
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── RAW TEXT EXPANDER ────────────────────────────────────────────────
    with st.expander("🔎 View Extracted Raw Text", expanded=False):
        st.text_area(
            "Extracted Document Text",
            value=text[:8000] + ("\n\n[... truncated for display ...]" if len(text) > 8000 else ""),
            height=300,
            label_visibility="collapsed",
        )

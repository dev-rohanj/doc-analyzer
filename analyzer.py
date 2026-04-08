"""
legal document analysis using a 5-pass pipeline:

  Pass 1 — Summary     : Summarise each chunk in plain English
  Pass 2 — Entities    : Extract persons and organisations only
  Pass 3 — Clauses     : Detect and explain key terms and provisions
  Pass 4 — Risks       : Flag one-sided, hidden, or harmful terms
  Pass 5 — Synthesis   : Merge, deduplicate, and score risk

Splitting work across focused passes keeps each prompt small enough
for an 8B model (gemma4) to handle accurately without losing context.
"""

import json
import os
import time
import httpx
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")


def _log(tag: str, msg: str) -> None:
    """Pretty-print a labelled log line to stdout."""
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(msg)
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Config — edit these directly to tune behaviour
# ---------------------------------------------------------------------------

OLLAMA_TIMEOUT_SECONDS = None     # None = no timeout; set e.g. 120.0 to add one

MAX_ANALYSIS_CHARS  = 120_000     # truncate input beyond this
CHUNK_SIZE_CHARS    = 4_000       # characters per chunk fed to the model
CHUNK_OVERLAP_CHARS = 400         # overlap between consecutive chunks
MAX_CHUNKS          = 24          # hard cap on number of chunks

MAX_SUMMARIES_IN_SYNTH = 12       # how many chunk summaries to pass into synthesis
MAX_CLAUSES_IN_SYNTH   = 24       # max raw clauses fed into synthesis
MAX_RISKS_IN_SYNTH     = 24       # max raw risks fed into synthesis
MAX_CLAUSES_OUTPUT     = 16       # max clauses in the final result
MAX_RISKS_OUTPUT       = 16       # max risks in the final result

REFERENCE_CLAUSE_TOPICS = [
    "payment and pricing",
    "renewal and auto-renewal",
    "termination and exit rights",
    "notice periods",
    "exclusivity or broker lock-in",
    "minimum commitment",
    "commission and recurring fees",
    "hidden charges or pass-through costs",
    "liability limits",
    "indemnity",
    "confidentiality",
    "data use and privacy",
    "non-compete or non-solicit",
    "governing law and dispute resolution",
    "assignment or transfer rights",
    "change of terms",
    "service levels or performance obligations",
    "warranties and disclaimers",
    "intellectual property ownership",
    "penalties, liquidated damages, or cancellation fees",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_document(text: str) -> dict:
    """
    Run the 5-pass pipeline and return a structured analysis result.
    """
    prepared = text[:MAX_ANALYSIS_CHARS]
    if not prepared.strip():
        return _empty_result()

    chunks = _chunk_text(prepared)
    total  = len(chunks)
    _log("PIPELINE START", f"Document split into {total} chunk(s)  |  model: {OLLAMA_MODEL}")

    # --- Pass 1: per-chunk summaries ------------------------------------
    _log("PASS 1 — SUMMARY", f"Summarising {total} chunk(s)...")
    summaries = []
    for i, chunk in enumerate(chunks):
        s = _pass_summary(chunk, i + 1, total)
        preview = (s[:200] + "...") if len(s) > 200 else s
        _log(f"PASS 1 — CHUNK {i+1}/{total}", preview or "(empty)")
        summaries.append(s)

    # --- Pass 2: per-chunk entity extraction ----------------------------
    _log("PASS 2 — ENTITIES", f"Extracting entities from {total} chunk(s)...")
    raw_persons = []
    raw_orgs    = []
    for i, chunk in enumerate(chunks):
        ents = _pass_entities(chunk)
        _log(
            f"PASS 2 — CHUNK {i+1}/{total}",
            f"  persons: {ents.get('persons', [])}\n  orgs:    {ents.get('organizations', [])}",
        )
        raw_persons.extend(ents.get("persons", []))
        raw_orgs.extend(ents.get("organizations", []))

    persons = _unique_strings(raw_persons, 40)
    orgs    = _unique_strings(raw_orgs, 40)
    _log("PASS 2 — MERGED", f"  persons: {persons}\n  orgs:    {orgs}")

    # --- Pass 3: per-chunk clause detection ----------------------------
    _log("PASS 3 — CLAUSES", f"Detecting clauses in {total} chunk(s)...")
    raw_clauses: list[dict] = []
    for i, chunk in enumerate(chunks):
        found = _pass_clauses(chunk, i + 1, total)
        _log(
            f"PASS 3 — CHUNK {i+1}/{total}",
            "\n".join(f"  [{c['name']}] {c['plain_language']}" for c in found) or "  (none found)",
        )
        raw_clauses.extend(found)

    clauses = _unique_items_by_key(
        [c for c in raw_clauses if c.get("name") and c.get("plain_language")],
        "name", MAX_CLAUSES_IN_SYNTH,
    )
    _log("PASS 3 — MERGED", f"  {len(clauses)} unique clause(s) collected")

    # --- Pass 4: per-chunk risk detection ------------------------------
    _log("PASS 4 — RISKS", f"Detecting risks in {total} chunk(s)...")
    raw_risks: list[dict] = []
    for i, chunk in enumerate(chunks):
        found = _pass_risks(chunk, i + 1, total)
        _log(
            f"PASS 4 — CHUNK {i+1}/{total}",
            "\n".join(
                f"  [{r['severity'].upper()}] {r['label']}: {r['plain_language']}"
                for r in found
            ) or "  (none found)",
        )
        raw_risks.extend(found)

    risks = _unique_items_by_key(
        [r for r in raw_risks if r.get("label") and r.get("plain_language")],
        "label", MAX_RISKS_IN_SYNTH,
    )
    _log("PASS 4 — MERGED", f"  {len(risks)} unique risk(s) collected")

    # --- Pass 5: synthesis (summary only — clauses/risks assembled directly) ---
    # The model cannot reliably merge 24 clauses + 18 risks in one call on 8B hardware.
    # So synthesis only writes the final summary. Clauses and risks come straight
    # from the per-chunk results, already deduplicated above.
    _log("PASS 5 — SYNTHESIS", "Writing final summary from chunk summaries...")

    # Slim evidence: only summaries + entity lists (no clauses/risks — too large)
    slim_evidence = {
        "chunk_summaries": [
            {"chunk": i + 1, "summary": s}
            for i, s in enumerate(summaries[:MAX_SUMMARIES_IN_SYNTH])
            if s
        ],
        "persons":       persons,
        "organizations": orgs,
    }
    final_summary, final_persons, final_orgs = _pass_synthesis(slim_evidence)

    # Assemble final result — clauses and risks come directly from per-chunk passes
    final = {
        "summary": final_summary,
        "entities": {
            "persons":       final_persons,
            "organizations": final_orgs,
        },
        "clauses": _unique_items_by_key(clauses, "name", MAX_CLAUSES_OUTPUT),
        "risks":   _unique_items_by_key(risks,   "label", MAX_RISKS_OUTPUT),
    }

    _log(
        "PASS 5 — FINAL OUTPUT",
        f"  summary:  {final['summary'][:200]}...\n"
        f"  clauses:  {len(final['clauses'])}\n"
        f"  risks:    {len(final['risks'])}\n"
        f"  persons:  {final['entities']['persons']}\n"
        f"  orgs:     {final['entities']['organizations']}",
    )
    _log("PIPELINE DONE", "Analysis complete.")
    return final


def analyze_pdf(uploaded_file) -> dict:
    """
    Ollama does not accept raw PDFs directly.
    Extract text before calling analyze_document().
    """
    raise RuntimeError(
        "No text could be extracted from this PDF. "
        "The Ollama pipeline only processes extracted text — "
        "scanned or image-only PDFs require OCR first."
    )


def calculate_risk_score(risks: list) -> dict:
    """
    Compute a 0-100 risk score from the detected risks list.
    """
    if not risks:
        return {"score": 0, "category": "Low Risk", "color": "green"}

    weights = {"high": 15, "medium": 7, "low": 3}
    raw    = sum(weights.get(r.get("severity", ""), 0) for r in risks)
    score  = min(int((raw / 120) * 100), 100)

    if score <= 30:
        return {"score": score, "category": "Low Risk",    "color": "green"}
    elif score <= 70:
        return {"score": score, "category": "Medium Risk", "color": "orange"}
    else:
        return {"score": score, "category": "High Risk",   "color": "red"}


# ---------------------------------------------------------------------------
# Pass 1 — Summary
# ---------------------------------------------------------------------------

_SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}

def _pass_summary(chunk: str, index: int, total: int) -> str:
    prompt = f"""
You are a legal document analyst. Read the following contract excerpt and write a concise summary.

Rules:
- Write 3 to 5 sentences only.
- Use simple, plain English that a non-lawyer can understand.
- Describe what this part of the document covers.
- Do not give legal advice or recommendations.
- Return only JSON matching this schema: {json.dumps(_SUMMARY_SCHEMA)}
- Do not wrap JSON in markdown fences.

Chunk {index} of {total}:
\"\"\"
{chunk}
\"\"\"
""".strip()
    result = _run_ollama_json(prompt)
    return (result.get("summary") or "").strip()


# ---------------------------------------------------------------------------
# Pass 2 — Entities
# ---------------------------------------------------------------------------

_ENTITIES_SCHEMA = {
    "type": "object",
    "properties": {
        "persons":       {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["persons", "organizations"],
}

def _pass_entities(chunk: str) -> dict:
    prompt = f"""
You are a named entity extractor. Read the following contract excerpt.

Tasks:
- Extract the full names of individual persons explicitly mentioned.
- Extract the full names of companies, organisations, or legal entities explicitly mentioned.
- Do not invent names that are not present in the text.

Return only JSON matching this schema: {json.dumps(_ENTITIES_SCHEMA)}
Do not wrap JSON in markdown fences.

Contract excerpt:
\"\"\"
{chunk}
\"\"\"
""".strip()
    result = _run_ollama_json(prompt)
    return {
        "persons":       _unique_strings(result.get("persons", []), 25),
        "organizations": _unique_strings(result.get("organizations", []), 25),
    }


# ---------------------------------------------------------------------------
# Pass 3 — Clauses
# ---------------------------------------------------------------------------

_CLAUSE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name":           {"type": "string"},
        "excerpt":        {"type": "string"},
        "plain_language": {"type": "string"},
    },
    "required": ["name", "excerpt", "plain_language"],
}
_CLAUSES_SCHEMA = {
    "type": "object",
    "properties": {"clauses": {"type": "array", "items": _CLAUSE_ITEM_SCHEMA}},
    "required": ["clauses"],
}

def _pass_clauses(chunk: str, index: int, total: int) -> list[dict]:
    topics_json = json.dumps(REFERENCE_CLAUSE_TOPICS)
    prompt = f"""
You are a legal clause extractor. Read the following contract excerpt and identify important clauses or terms.

Reference clause areas (not a fixed list — include other important terms too):
{topics_json}

For each clause found, provide:
- name: a short clause name (under 60 characters)
- excerpt: a short excerpt or close paraphrase under 280 characters
- plain_language: a plain English explanation under 220 characters that a non-lawyer can understand

Rules:
- Only include clauses clearly supported by the text.
- Do not guess or speculate.
- If the excerpt contains no meaningful clauses, return an empty clauses array.
- Return only JSON matching this schema: {json.dumps(_CLAUSES_SCHEMA)}
- Do not wrap JSON in markdown fences.

Chunk {index} of {total}:
\"\"\"
{chunk}
\"\"\"
""".strip()
    result = _run_ollama_json(prompt)
    raw = result.get("clauses", [])
    out = []
    for item in raw:
        name  = (item.get("name") or "").strip()
        plain = (item.get("plain_language") or "").strip()
        if not name or not plain:
            continue
        out.append({
            "name":           name,
            "excerpt":        (item.get("excerpt") or "").strip(),
            "plain_language": plain,
        })
    return out


# ---------------------------------------------------------------------------
# Pass 4 — Risks
# ---------------------------------------------------------------------------

_RISK_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "label":          {"type": "string"},
        "severity":       {"type": "string", "enum": ["high", "medium", "low"]},
        "found_keywords": {"type": "array", "items": {"type": "string"}},
        "plain_language": {"type": "string"},
    },
    "required": ["label", "severity", "found_keywords", "plain_language"],
}
_RISKS_SCHEMA = {
    "type": "object",
    "properties": {"risks": {"type": "array", "items": _RISK_ITEM_SCHEMA}},
    "required": ["risks"],
}

def _pass_risks(chunk: str, index: int, total: int) -> list[dict]:
    prompt = f"""
You are a legal risk analyst. Read the following contract excerpt and identify notable legal or commercial risks.

Look for:
- Broad or unlimited indemnity obligations
- Unilateral rights to change terms or pricing
- Auto-renewal or rollover traps
- Hidden fees or pass-through costs
- Harsh termination rights or exit penalties
- Exclusivity or lock-in clauses
- Non-compete overreach
- One-sided liability limits
- Missing obligations from one party
- Vague or ambiguous language that could be exploited
- Any term that is strongly unfair, unusual, or harmful

For each risk found:
- label: a short professional label under 80 characters
- severity: one of high, medium, or low
- found_keywords: up to 3 short phrases from the text that triggered this risk
- plain_language: a plain English explanation under 220 characters

Severity guide:
- high: materially adverse, punitive, or clearly harmful to one party
- medium: potentially problematic, worth careful review
- low: minor imbalance or noteworthy term

Rules:
- Only flag risks clearly supported by the text.
- Do not speculate or invent risks not present.
- If no meaningful risks exist, return an empty risks array.
- Return only JSON matching this schema: {json.dumps(_RISKS_SCHEMA)}
- Do not wrap JSON in markdown fences.

Chunk {index} of {total}:
\"\"\"
{chunk}
\"\"\"
""".strip()
    result = _run_ollama_json(prompt)
    raw = result.get("risks", [])
    out = []
    for risk in raw:
        severity = (risk.get("severity") or "").lower()
        if severity not in {"high", "medium", "low"}:
            continue
        label = (risk.get("label") or "").strip()
        plain = (risk.get("plain_language") or "").strip()
        if not label or not plain:
            continue
        out.append({
            "label":          label,
            "severity":       severity,
            "found_keywords": _unique_strings(risk.get("found_keywords", []), 3),
            "plain_language": plain,
        })
    return out


# ---------------------------------------------------------------------------
# Pass 5 — Synthesis
# ---------------------------------------------------------------------------

# Synthesis schema is now minimal — summary + entities only.
# Clauses and risks are assembled directly from per-chunk results.
_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "persons":       {"type": "array", "items": {"type": "string"}},
        "organizations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "persons", "organizations"],
}

def _pass_synthesis(evidence: dict) -> tuple[str, list, list]:
    """
    Only task: write a final summary and clean up entity lists.
    Returns (summary, persons, organizations).
    Clauses and risks are handled directly in analyze_document.
    """
    schema_json   = json.dumps(_SYNTHESIS_SCHEMA, indent=2)
    evidence_json = json.dumps(evidence, indent=2)
    prompt = f"""
You are a legal document analyst. You have been given summaries of each section
of a legal document, along with lists of persons and organisations mentioned.

Evidence:
{evidence_json}

Tasks:
1. Write a final summary in 5 to 8 sentences covering the whole document.
   - Use simple, plain English for a non-lawyer audience.
   - Describe what the document is, who the parties are, and what the key terms cover.
   - Do not give legal advice.
2. Return a clean, deduplicated list of persons mentioned.
   - Remove garbled text, OCR artifacts, and generic labels like "The owner".
   - Keep only real, legible names.
3. Return a clean, deduplicated list of organisations mentioned.
   - Remove garbled text, OCR artifacts, and generic labels like "Police".
   - Keep only real, legible organisation names.

Return only JSON matching this schema exactly:
{schema_json}
Do not wrap JSON in markdown fences.
Do not add any text before or after the JSON object.
""".strip()

    payload = _run_ollama_json(prompt)

    # Validate — if summary is missing or too short, fall back gracefully
    summary = (payload.get("summary") or "").strip()
    if len(summary) < 30:
        _log("PASS 5 — WARNING", f"Synthesis returned a weak summary: '{summary}'. Using chunk summaries as fallback.")
        summary = " ".join(v for v in evidence.get("chunk_summaries", [{}]) if isinstance(v, str))
        if not summary:
            summary = " ".join(
                item.get("summary", "") for item in evidence.get("chunk_summaries", [])
            )

    persons = _unique_strings(payload.get("persons", []), 25)
    orgs    = _unique_strings(payload.get("organizations", []), 25)

    # If model returned nothing for entities, fall back to raw input
    if not persons:
        persons = _unique_strings(evidence.get("persons", []), 25)
    if not orgs:
        orgs = _unique_strings(evidence.get("organizations", []), 25)

    return summary, persons, orgs


# _normalize_final removed — synthesis now returns (summary, persons, orgs) tuple directly.


def _empty_result() -> dict:
    return {
        "summary": "",
        "entities": {"persons": [], "organizations": []},
        "clauses":  [],
        "risks":    [],
    }


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def _run_ollama_json(prompt: str) -> dict:
    client = Client(
        host=OLLAMA_HOST.rstrip("/"),
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    try:
        t0 = time.time()
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            format="json",
            options={"temperature": 0.1},
        )
        elapsed = time.time() - t0
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "Ollama request timed out. Increase OLLAMA_TIMEOUT_SECONDS "
            "or set it to 0 to disable the client-side timeout."
        ) from exc

    message  = response.get("message") or {}
    raw_text = (message.get("content") or "").strip()
    _log(f"LLM RAW RESPONSE  ({elapsed:.1f}s)", raw_text or "(empty)")
    if not raw_text:
        raise RuntimeError("Ollama returned an empty response.")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return json.loads(_extract_json_object(raw_text))


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Ollama response was not valid JSON.")
    return text[start: end + 1]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return [""]

    chunks = []
    start  = 0
    length = len(normalized)

    while start < length and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE_CHARS, length)
        if end < length:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start + (CHUNK_SIZE_CHARS // 2):
                end = split_at

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        next_start = max(0, end - CHUNK_OVERLAP_CHARS)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks or [normalized[:CHUNK_SIZE_CHARS]]


def _unique_strings(values: list, limit: int) -> list:
    seen    = set()
    cleaned = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def _unique_items_by_key(items: list, key_name: str, limit: int) -> list:
    seen    = set()
    cleaned = []
    for item in items:
        key = item[key_name].casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned
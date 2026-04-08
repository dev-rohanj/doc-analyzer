"""
Legal document analysis — 5-pass pipeline
==========================================
Pass 1 — Summary     : Summarise each chunk in plain English
Pass 2 — Entities    : Extract persons and organisations only
Pass 3 — Clauses     : Detect and explain key terms and provisions
Pass 4 — Risks       : Flag one-sided, hidden, or harmful terms
Pass 5 — Synthesis   : Merge, deduplicate, and score risk
"""

import asyncio
import json
import math
import os
import re
import time
import httpx
from ollama import AsyncClient
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")


def _log(tag: str, msg: str) -> None:
    print(f"\n{'='*60}\n  {tag}\n{'='*60}\n{msg}\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_TIMEOUT_SECONDS  = None
MAX_ANALYSIS_CHARS      = 120_000
CHUNK_SIZE_CHARS        = 4_000
CHUNK_OVERLAP_CHARS     = 400
MAX_CHUNKS              = 24
MAX_SUMMARIES_IN_SYNTH  = 12
MAX_CLAUSES_IN_SYNTH    = 24
MAX_RISKS_IN_SYNTH      = 24
MAX_CLAUSES_OUTPUT      = 16
MAX_RISKS_OUTPUT        = 16
MIN_CHUNK_WORDS         = 40

# Minimum char length for a valid person/org name — rejects "Mr.", "A", etc.
MIN_ENTITY_LEN     = 4
# Reject strings with slashes, digits, or non-Latin garbage (OCR noise)
_GARBAGE_ENTITY_RE = re.compile(r"[/\\0-9@#$%^&*<>{}|~`]|æ|ø|ð", re.IGNORECASE)

VALID_SEVERITIES = {"high", "medium", "low"}

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

_TOPICS_BULLET = "\n".join(f"- {t}" for t in REFERENCE_CLAUSE_TOPICS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_document(text: str) -> dict:
    return asyncio.run(_analyze_document_async(text))


async def _analyze_document_async(text: str) -> dict:
    prepared = text[:MAX_ANALYSIS_CHARS]
    if not prepared.strip():
        return _empty_result()

    chunks = _chunk_text(prepared)
    chunks = [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]
    if not chunks:
        return _empty_result()

    total = len(chunks)
    _log("PIPELINE START", f"Document split into {total} chunk(s)  |  model: {OLLAMA_MODEL}")

    # --- Pass 1 ---
    _log("PASS 1 — SUMMARY", f"Summarising {total} chunk(s) in parallel...")
    summaries = await _parallel([_pass_summary(c, i + 1, total) for i, c in enumerate(chunks)])
    for i, s in enumerate(summaries):
        _log(f"PASS 1 — CHUNK {i+1}/{total}", (s[:200] + "...") if len(s) > 200 else s or "(empty)")

    # --- Pass 2 ---
    _log("PASS 2 — ENTITIES", f"Extracting entities from {total} chunk(s) in parallel...")
    entity_results = await _parallel([_pass_entities(c) for c in chunks])
    raw_persons, raw_orgs = [], []
    for i, ents in enumerate(entity_results):
        _log(
            f"PASS 2 — CHUNK {i+1}/{total}",
            f"  persons: {ents.get('persons', [])}\n  orgs:    {ents.get('organizations', [])}",
        )
        raw_persons.extend(ents.get("persons", []))
        raw_orgs.extend(ents.get("organizations", []))

    persons = _fuzzy_unique(_clean_entities(raw_persons), 40)
    orgs    = _fuzzy_unique(_clean_entities(raw_orgs),    40)
    _log("PASS 2 — MERGED", f"  persons: {persons}\n  orgs:    {orgs}")

    # --- Pass 3 ---
    _log("PASS 3 — CLAUSES", f"Detecting clauses in {total} chunk(s) in parallel...")
    clause_results = await _parallel([_pass_clauses(c, i + 1, total) for i, c in enumerate(chunks)])
    raw_clauses: list[dict] = []
    for i, found in enumerate(clause_results):
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

    # --- Pass 4 ---
    _log("PASS 4 — RISKS", f"Detecting risks in {total} chunk(s) in parallel...")
    risk_results = await _parallel([_pass_risks(c, i + 1, total) for i, c in enumerate(chunks)])
    raw_risks: list[dict] = []
    for i, found in enumerate(risk_results):
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

    # --- Pass 5 ---
    _log("PASS 5 — SYNTHESIS", "Writing final summary from chunk summaries...")
    slim_evidence = {
        "chunk_summaries": [
            {"chunk": i + 1, "summary": s}
            for i, s in enumerate(summaries[:MAX_SUMMARIES_IN_SYNTH])
            if s
        ],
        "persons":       persons,
        "organizations": orgs,
    }
    final_summary, final_persons, final_orgs = await _pass_synthesis(slim_evidence)

    final = {
        "summary": final_summary,
        "entities": {
            "persons":       final_persons,
            "organizations": final_orgs,
        },
        "clauses": _unique_items_by_key(clauses, "name",  MAX_CLAUSES_OUTPUT),
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
    raise RuntimeError(
        "No text could be extracted from this PDF. "
        "The Ollama pipeline only processes extracted text — "
        "scanned or image-only PDFs require OCR first."
    )


def calculate_risk_score(risks: list) -> dict:
    if not risks:
        return {"score": 0, "category": "Low Risk", "color": "green"}

    weights = {"high": 15, "medium": 7, "low": 3}
    raw   = sum(weights.get(r.get("severity", ""), 0) for r in risks)
    score = int(100 / (1 + math.exp(-0.04 * (raw - 60))))

    if score <= 30:
        return {"score": score, "category": "Low Risk",    "color": "green"}
    elif score <= 70:
        return {"score": score, "category": "Medium Risk", "color": "orange"}
    else:
        return {"score": score, "category": "High Risk",   "color": "red"}


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------
async def _parallel(coros: list) -> list:
    return list(await asyncio.gather(*coros, return_exceptions=False))


# ---------------------------------------------------------------------------
# Pass 1 — Summary
# Giving phi4-mini a concrete output example anchors its JSON structure.
# ---------------------------------------------------------------------------
async def _pass_summary(chunk: str, index: int, total: int) -> str:
    prompt = f"""You are a legal analyst reading part {index} of {total} of a contract.

Summarise the key points of this section in 3 to 5 plain English sentences.
Focus on: who the parties are, what is being agreed, any important dates or amounts, and key obligations.

Output ONLY this JSON and nothing else:
{{"summary": "Your summary here."}}

CONTRACT SECTION:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt)
        return (result.get("summary") or "").strip()
    except Exception as e:
        _log(f"PASS 1 — CHUNK {index} ERROR", str(e))
        return ""


# ---------------------------------------------------------------------------
# Pass 2 — Entities
# Explicit rejection rules are critical for phi4-mini on OCR-noisy PDFs.
# ---------------------------------------------------------------------------
async def _pass_entities(chunk: str) -> dict:
    prompt = f"""You are extracting named persons and organisations from a legal contract.

RULES — READ CAREFULLY:
1. Only extract names that are clearly and fully written in the text.
2. A valid person name has at least a first name and last name (e.g. "Mahesh Patil").
3. REJECT any string that contains: slashes, digits, symbols, garbled characters, or looks like OCR noise.
4. REJECT role labels like "Owner", "Licensee", "Licensor", "Tenant", "Party".
5. REJECT partial strings like "Mr.", "Owner Mr.", "Sign/Eætt".
6. If you are not sure a name is real, leave it out.
7. Return empty arrays if no valid names are found — do NOT guess.

Output ONLY this JSON and nothing else:
{{
  "persons": ["Full Name 1", "Full Name 2"],
  "organizations": ["Org Name 1"]
}}

CONTRACT SECTION:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt)
        return {
            "persons":       _fuzzy_unique(_clean_entities(result.get("persons", [])),       25),
            "organizations": _fuzzy_unique(_clean_entities(result.get("organizations", [])), 25),
        }
    except Exception as e:
        _log("PASS 2 — ERROR", str(e))
        return {"persons": [], "organizations": []}


# ---------------------------------------------------------------------------
# Pass 3 — Clauses
# Require verbatim excerpts and short names. Concrete example in prompt.
# ---------------------------------------------------------------------------
async def _pass_clauses(chunk: str, index: int, total: int) -> list[dict]:
    prompt = f"""You are a legal analyst extracting contract clauses from section {index} of {total}.

For each important clause you find, return:
- "name": a short label (3-6 words) matching one of these topics if possible:
{_TOPICS_BULLET}
- "excerpt": copy a SHORT verbatim quote (max 30 words) directly from the text. DO NOT invent or paraphrase.
- "plain_language": one clear sentence explaining what this clause means for the tenant/licensee.

RULES:
1. Only extract clauses that genuinely appear in the text below.
2. The excerpt MUST be words copied directly from the text — never invented.
3. If no relevant clauses exist in this section, return an empty list.
4. Do NOT duplicate clauses with the same meaning.
5. Maximum 5 clauses per section.

Output ONLY this JSON and nothing else:
{{
  "clauses": [
    {{
      "name": "Termination and Exit Rights",
      "excerpt": "The Licensor shall have an option to terminate this Agreement by giving one month prior notice.",
      "plain_language": "Either party can end this agreement by giving one month written notice."
    }}
  ]
}}

CONTRACT SECTION:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt)
        raw = result.get("clauses", [])
        cleaned = []
        for c in raw:
            name  = (c.get("name")           or "").strip()
            excpt = (c.get("excerpt")         or "").strip()
            plain = (c.get("plain_language")  or "").strip()
            if not name or not plain:
                continue
            # Reject excerpts that look invented (OCR-noise pattern: all-caps run + slash)
            if excpt and re.search(r"\b[A-Z]{5,}\b", excpt) and "/" in excpt:
                excpt = ""
            cleaned.append({"name": name, "excerpt": excpt, "plain_language": plain})
        return cleaned
    except Exception as e:
        _log(f"PASS 3 — CHUNK {index} ERROR", str(e))
        return []


# ---------------------------------------------------------------------------
# Pass 4 — Risks
# Hard-enforce severity enum. phi4-mini invents "medium-high", "medium-low".
# ---------------------------------------------------------------------------
async def _pass_risks(chunk: str, index: int, total: int) -> list[dict]:
    prompt = f"""You are a legal risk analyst reviewing section {index} of {total} of a contract.

Identify terms that are unfair, one-sided, risky, hidden, or harmful to the tenant/licensee.

For each risk, return:
- "label": a short name for the risk (4-7 words)
- "severity": MUST be exactly one of: "high", "medium", or "low"
  - high   = could cause significant financial loss or legal liability
  - medium = creates ambiguity or moderate disadvantage
  - low    = minor inconvenience or common standard clause
- "found_keywords": 1-3 short phrases from the text that triggered this risk
- "plain_language": one sentence explaining the risk in simple English

RULES:
1. Only flag risks grounded in the actual text — do NOT guess or assume.
2. The severity field MUST be exactly "high", "medium", or "low". Never use "medium-high", "medium-low", or any other value.
3. Do NOT flag normal standard clauses as risks (e.g. "tenant must not damage property").
4. If there are no risks in this section, return an empty list.
5. Maximum 6 risks per section.

Output ONLY this JSON and nothing else:
{{
  "risks": [
    {{
      "label": "Unilateral rent increase clause",
      "severity": "high",
      "found_keywords": ["10% rent increase", "at time of renewal"],
      "plain_language": "The landlord can raise rent by 10% at each renewal with no negotiation rights for the tenant."
    }}
  ]
}}

CONTRACT SECTION:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt)
        raw = result.get("risks", [])
        cleaned = []
        for r in raw:
            label = (r.get("label")          or "").strip()
            plain = (r.get("plain_language")  or "").strip()
            if not label or not plain:
                continue
            sev = _normalise_severity((r.get("severity") or "").lower().strip())
            cleaned.append({
                "label":          label,
                "severity":       sev,
                "found_keywords": _fuzzy_unique(r.get("found_keywords", []), 3),
                "plain_language": plain,
            })
        return cleaned
    except Exception as e:
        _log(f"PASS 4 — CHUNK {index} ERROR", str(e))
        return []


# ---------------------------------------------------------------------------
# Pass 5 — Synthesis
# Keep prompt short — phi4-mini quality degrades with very long prompts.
# Use bullet list instead of raw JSON evidence.
# ---------------------------------------------------------------------------
async def _pass_synthesis(evidence: dict) -> tuple[str, list, list]:
    bullets = "\n".join(
        f"- {item['summary']}"
        for item in evidence.get("chunk_summaries", [])
        if item.get("summary")
    )
    persons_hint = ", ".join(evidence.get("persons", [])) or "none found"
    orgs_hint    = ", ".join(evidence.get("organizations", [])) or "none found"

    prompt = f"""You are a legal analyst. Write a final summary of a contract based on these section summaries:

{bullets}

Known parties — persons: {persons_hint} | organisations: {orgs_hint}

Write a clear 5-8 sentence summary in plain English covering:
what the contract is for, who the parties are, key financial terms, duration, and termination rights.
Also return cleaned lists of persons and organisations (real full names only, no role labels).

Output ONLY this JSON and nothing else:
{{
  "summary": "...",
  "persons": ["Full Name"],
  "organizations": ["Org Name"]
}}
"""
    try:
        payload = await _run_ollama_json_async(prompt)
    except Exception as e:
        _log("PASS 5 — ERROR", str(e))
        payload = {}

    summary = (payload.get("summary") or "").strip()
    if len(summary) < 30:
        _log("PASS 5 — WARNING", "Weak synthesis — falling back to concatenated chunk summaries.")
        summary = " ".join(
            item.get("summary", "") for item in evidence.get("chunk_summaries", [])
        ).strip()

    persons = _fuzzy_unique(_clean_entities(payload.get("persons", [])), 25)
    orgs    = _fuzzy_unique(_clean_entities(payload.get("organizations", [])), 25)

    if not persons:
        persons = _fuzzy_unique(_clean_entities(evidence.get("persons", [])), 25)
    if not orgs:
        orgs = _fuzzy_unique(_clean_entities(evidence.get("organizations", [])), 25)

    return summary, persons, orgs


def _empty_result() -> dict:
    return {
        "summary": "",
        "entities": {"persons": [], "organizations": []},
        "clauses":  [],
        "risks":    [],
    }


# ---------------------------------------------------------------------------
# Ollama async client
# num_ctx MUST be >= chunk size in tokens.
# 4000 chars ≈ 700-900 tokens. With prompt overhead, 4096 gives solid headroom.
# 2048 was silently truncating every chunk — the #1 cause of hallucinated output.
# ---------------------------------------------------------------------------
async def _run_ollama_json_async(prompt: str) -> dict:
    client = AsyncClient(
        host=OLLAMA_HOST.rstrip("/"),
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    try:
        t0 = time.time()
        response = await client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only output engine. "
                        "Your entire response must be valid JSON. "
                        "Do not write any text outside the JSON object. "
                        "Do not explain your reasoning. "
                        "Do not add markdown formatting or code fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
            format="json",
            options={
                "temperature":    0.05,  # Low — reduces hallucination on structured tasks
                "num_ctx":        4096,  # Was 2048 — must cover chunk + prompt overhead
                "repeat_penalty": 1.1,   # Relaxed from 1.2 — was causing truncated JSON
                "top_p":          0.9,   # Nucleus sampling — better for structured output
            },
        )
        elapsed = time.time() - t0
    except httpx.TimeoutException as exc:
        raise RuntimeError("Ollama request timed out.") from exc

    message  = response.get("message") or {}
    raw_text = (message.get("content") or "").strip()

    # Strip chain-of-thought blocks before parse
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    _log(f"LLM RAW RESPONSE ({elapsed:.1f}s)", raw_text or "(empty)")

    if not raw_text:
        raise RuntimeError("Empty response from Ollama.")

    return _parse_json_response(raw_text)


def _parse_json_response(raw_text: str) -> dict:
    fenced = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    for candidate in (fenced, raw_text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for open_ch, close_ch in (('{', '}'), ('[', ']')):
        start = raw_text.find(open_ch)
        end   = raw_text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start: end + 1])
            except json.JSONDecodeError:
                pass

    raise RuntimeError(f"Ollama response was not valid JSON: {raw_text[:200]}")


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def _normalise_severity(sev: str) -> str:
    """Map any severity variant to exactly high/medium/low."""
    if sev in VALID_SEVERITIES:
        return sev
    if "high" in sev:
        return "high"
    if "low" in sev:
        return "low"
    return "medium"


def _clean_entities(values: list) -> list:
    """
    Remove OCR noise, role labels, and garbage strings from entity lists.
    Keeps only strings that look like plausible human/org names.
    """
    role_labels = {
        "owner", "licensee", "licensor", "tenant", "landlord",
        "party", "parties", "buyer", "seller", "agent", "broker",
        "depositor", "mortgagor", "purchaser", "signatory",
    }
    cleaned = []
    for val in values:
        item = str(val).strip()
        if len(item) < MIN_ENTITY_LEN:
            continue
        if _GARBAGE_ENTITY_RE.search(item):
            continue
        if item.lower() in role_labels:
            continue
        # Partial honorifics alone ("Mr.", "Mrs.")
        if re.match(r"^(Mr\.|Mrs\.|Ms\.|Dr\.)\s*$", item, re.IGNORECASE):
            continue
        # Short all-caps are abbreviations, not names
        if item.isupper() and len(item) <= 5:
            continue
        cleaned.append(item)
    return cleaned


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _chunk_text(text: str) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text).strip()
    if not normalized:
        return [""]

    sentence_ends = [0] + [
        m.end() for m in re.finditer(r"[.!?]\s+(?=[A-Z\u00C0-\u017E])", normalized)
    ]

    chunks: list[str] = []
    start  = 0
    length = len(normalized)

    while start < length and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE_CHARS, length)
        if end < length:
            lookback_from = start + int(CHUNK_SIZE_CHARS * 0.75)
            sent_candidates = [p for p in sentence_ends if lookback_from <= p <= end]
            if sent_candidates:
                end = sent_candidates[-1]
            else:
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


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    if max(la, lb) / max(min(la, lb), 1) > 2.5:
        return 0.0
    common_prefix = sum(1 for x, y in zip(a, b) if x == y)
    return common_prefix / max(la, lb)


def _fuzzy_unique(values: list, limit: int, threshold: float = 0.85) -> list:
    seen:    list[str] = []
    cleaned: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        key = item.casefold()
        if key in [s.casefold() for s in seen]:
            continue
        if any(_levenshtein_ratio(key, s.casefold()) >= threshold for s in seen):
            continue
        seen.append(item)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def _unique_strings(values: list, limit: int) -> list:
    seen:    set[str]  = set()
    cleaned: list[str] = []
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
    seen:    set[str]   = set()
    cleaned: list[dict] = []
    for item in items:
        key = item[key_name].casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned
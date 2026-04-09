"""
Legal document analysis — 5-pass pipeline + Progressive Context Accumulator
============================================================================

Architecture
------------
Pass 0 — Fingerprint : Run on chunk 1 only. Extracts doc type, parties, dates,
                       and financial amounts. Seeds the shared PipelineState.

Pass 1 — Summary     : Parallel across all chunks. Each prompt receives a compact
                       state prefix so the model knows who the parties are.

Pass 2 — Entities    : Parallel across all chunks. State prefix anchors names
                       found in chunk 1 so later chunks can confirm, not re-invent.

Pass 3 — Clauses     : Sequential across chunks. State tracks seen clause names
                       so each chunk is told "don't re-report these". Prevents
                       the 2-clause-repeated-4-times problem.

Pass 4 — Risks       : Sequential across chunks. State tracks seen risk labels
                       so near-duplicate risks are never generated in the first
                       place — not just deduped after the fact.

Pass 5 — Synthesis   : Single call. Receives confirmed state facts alongside
                       chunk summaries, so it doesn't have to guess doc type or
                       party names from garbled OCR hints.

Token budget per prompt
-----------------------
State prefix:  ~200–350 tokens (well within phi4-mini budget)
Chunk text:    ~700–900 tokens
Prompt shell:  ~200–300 tokens
Output cap:    set per-pass via num_predict
Total input:   ~1100–1550 tokens — leaves healthy headroom inside 4096 ctx
"""

import asyncio
import json
import math
import os
import re
import time
import httpx
from dataclasses import dataclass, field
from ollama import AsyncClient
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")


def _log(tag: str, msg: str) -> None:
    print(f"\n{'='*60}\n  {tag}\n{'='*60}\n{msg}\n{'='*60}\n")


# ─── Config ───────────────────────────────────────────────────────────────────

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

MIN_ENTITY_LEN     = 4
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


# ─── Progressive Context Accumulator ─────────────────────────────────────────
#
# Shared state object that travels through the entire pipeline.
# Populated by Pass 0 (fingerprint) before any other pass runs.
# Passes 3 and 4 also write to it as they process each chunk sequentially,
# so each subsequent chunk knows what has already been found.
#
# Design constraints:
#   - All fields serialise to short strings or small lists — never raw chunks
#   - seen_clauses / seen_risks are sets of casefolded keys for O(1) lookup
#   - build_prefix() renders state into a compact block (~200–350 tokens)
#     that can be prepended to any pass prompt without overloading phi4-mini
#

@dataclass
class PipelineState:
    # Populated by Pass 0
    doc_type:    str        = ""
    parties:     list[str]  = field(default_factory=list)   # confirmed real names
    key_dates:   list[str]  = field(default_factory=list)
    financials:  list[str]  = field(default_factory=list)   # e.g. ["Rs. 10,000/month"]

    # Populated by Pass 2 merge
    persons:     list[str]  = field(default_factory=list)
    orgs:        list[str]  = field(default_factory=list)

    # Written by Pass 3 as each chunk is processed (sequential)
    seen_clauses: set[str]  = field(default_factory=set)    # casefolded clause names

    # Written by Pass 4 as each chunk is processed (sequential)
    seen_risks:   set[str]  = field(default_factory=set)    # casefolded risk labels

    def build_prefix(self, include_seen: bool = True) -> str:
        """
        Render a compact context block to prepend to any LLM prompt.
        include_seen=True adds the "already found" dedup hints used by
        passes 3 and 4. Passes 1, 2, and 5 set include_seen=False.
        Returns empty string if there is nothing useful to add.
        """
        lines = ["DOCUMENT CONTEXT (established from the opening section):"]

        if self.doc_type:
            lines.append(f"  Document type : {self.doc_type}")

        all_parties = _fuzzy_unique(self.parties + self.persons + self.orgs, 12)
        if all_parties:
            lines.append(f"  Known parties : {', '.join(all_parties)}")

        if self.key_dates:
            lines.append(f"  Key dates     : {', '.join(self.key_dates)}")

        if self.financials:
            lines.append(f"  Known amounts : {', '.join(self.financials)}")

        if include_seen:
            if self.seen_clauses:
                sc = ", ".join(sorted(self.seen_clauses))
                lines.append(
                    f"  Clauses already found in earlier sections "
                    f"(DO NOT repeat these): {sc}"
                )
            if self.seen_risks:
                sr = ", ".join(sorted(self.seen_risks))
                lines.append(
                    f"  Risks already found in earlier sections "
                    f"(DO NOT repeat these): {sr}"
                )

        if len(lines) == 1:
            return ""   # Nothing to add — omit prefix entirely

        return "\n".join(lines) + "\n\n"

    def update_from_fingerprint(self, fp: dict) -> None:
        self.doc_type   = (fp.get("doc_type")   or "").strip()
        self.parties    = _clean_entities(fp.get("parties",    []))
        self.key_dates  = [str(d).strip() for d in fp.get("key_dates",  []) if d][:6]
        self.financials = [str(f).strip() for f in fp.get("financials", []) if f][:6]
        _log(
            "PASS 0 — STATE SEEDED",
            f"  doc_type  : {self.doc_type}\n"
            f"  parties   : {self.parties}\n"
            f"  key_dates : {self.key_dates}\n"
            f"  financials: {self.financials}",
        )


# ─── Public API ───────────────────────────────────────────────────────────────

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
    state = PipelineState()

    _log("PIPELINE START", f"Document split into {total} chunk(s)  |  model: {OLLAMA_MODEL}")

    # ── Pass 0 — Fingerprint (chunk 1 only) ───────────────────────────────────
    _log("PASS 0 — FINGERPRINT", "Extracting doc type, parties, dates, amounts from chunk 1...")
    fp = await _pass_fingerprint(chunks[0])
    state.update_from_fingerprint(fp)

    # ── Pass 1 — Summary (parallel, state prefix for party names) ─────────────
    _log("PASS 1 — SUMMARY", f"Summarising {total} chunk(s) in parallel...")
    summaries = await _parallel([
        _pass_summary(c, i + 1, total, state) for i, c in enumerate(chunks)
    ])
    for i, s in enumerate(summaries):
        _log(
            f"PASS 1 — CHUNK {i+1}/{total}",
            (s[:200] + "...") if len(s) > 200 else s or "(empty)",
        )

    # ── Pass 2 — Entities (parallel, state prefix anchors known names) ─────────
    _log("PASS 2 — ENTITIES", f"Extracting entities from {total} chunk(s) in parallel...")
    entity_results = await _parallel([
        _pass_entities(c, state) for c in chunks
    ])
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
    state.persons = persons
    state.orgs    = orgs
    _log("PASS 2 — MERGED", f"  persons: {persons}\n  orgs:    {orgs}")

    # ── Pass 3 — Clauses (sequential, state tracks seen clause names) ──────────
    _log("PASS 3 — CLAUSES", f"Detecting clauses in {total} chunk(s) sequentially...")
    raw_clauses: list[dict] = []
    for i, chunk in enumerate(chunks):
        found = await _pass_clauses(chunk, i + 1, total, state)
        _log(
            f"PASS 3 — CHUNK {i+1}/{total}",
            "\n".join(
                f"  [{c['name']}] {c['plain_language']}" for c in found
            ) or "  (none found)",
        )
        for c in found:
            state.seen_clauses.add(c["name"].casefold())
        raw_clauses.extend(found)

    clauses = _unique_items_by_key(
        [c for c in raw_clauses if c.get("name") and c.get("plain_language")],
        "name", MAX_CLAUSES_IN_SYNTH,
    )
    _log("PASS 3 — MERGED", f"  {len(clauses)} unique clause(s) collected")

    # ── Pass 4 — Risks (sequential, state tracks seen risk labels) ─────────────
    _log("PASS 4 — RISKS", f"Detecting risks in {total} chunk(s) sequentially...")
    raw_risks: list[dict] = []
    for i, chunk in enumerate(chunks):
        found = await _pass_risks(chunk, i + 1, total, state)
        _log(
            f"PASS 4 — CHUNK {i+1}/{total}",
            "\n".join(
                f"  [{r['severity'].upper()}] {r['label']}: {r['plain_language']}"
                for r in found
            ) or "  (none found)",
        )
        for r in found:
            state.seen_risks.add(r["label"].casefold())
        raw_risks.extend(found)

    risks = _unique_items_by_key(
        [r for r in raw_risks if r.get("label") and r.get("plain_language")],
        "label", MAX_RISKS_IN_SYNTH,
    )
    _log("PASS 4 — MERGED", f"  {len(risks)} unique risk(s) collected")

    # ── Pass 5 — Synthesis ─────────────────────────────────────────────────────
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
    final_summary, final_persons, final_orgs = await _pass_synthesis(slim_evidence, state)

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


# ─── Async helpers ────────────────────────────────────────────────────────────

async def _parallel(coros: list) -> list:
    return list(await asyncio.gather(*coros, return_exceptions=False))


# ─── Pass 0 — Fingerprint ─────────────────────────────────────────────────────
#
# Runs on chunk 1 only (~1 LLM call overhead).
# Extracts four facts that anchor every subsequent prompt:
#   doc type, named parties, key dates, financial amounts.
# Keeping the output schema flat and small ensures reliable extraction
# without overrunning the generation budget.
#
async def _pass_fingerprint(chunk: str) -> dict:
    prompt = f"""You are reading the opening section of a legal document.

Extract the following four facts from the text below. Each field must be short.

- "doc_type"   : What kind of legal agreement is this?
                 Examples: "residential lease", "employment contract", "loan agreement",
                 "SaaS subscription agreement", "non-disclosure agreement", "service agreement"
                 Write 2–5 words only. If unclear, write "legal agreement".

- "parties"    : Full names of real people or registered organisations who are parties to this
                 agreement. Include ONLY names that are clearly readable — no role labels, no
                 OCR noise, no garbled strings.
                 If no clear names exist, return an empty list.

- "key_dates"  : Any dates mentioned (signing date, start date, end date, term length).
                 Write each as a short string, e.g. "1 Jan 2024", "12-month term".
                 Maximum 4 entries. If none, return an empty list.

- "financials" : Any monetary amounts mentioned (rent, fees, deposit, salary, penalty).
                 Write each as a short string, e.g. "Rs. 10,000/month", "deposit Rs. 50,000".
                 Maximum 4 entries. If none, return an empty list.

RULES:
- Only include information clearly present in the text below — no inference or invention.
- If a field has no clear answer, use an empty string or empty list.

Output ONLY this JSON — no other text:
{{
  "doc_type"  : "type of agreement",
  "parties"   : ["Full Name or Org Name"],
  "key_dates" : ["date or term string"],
  "financials": ["amount string"]
}}

SECTION TEXT:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt, num_predict=300)
        return result
    except Exception as e:
        _log("PASS 0 — ERROR", str(e))
        return {}


# ─── Pass 1 — Summary ─────────────────────────────────────────────────────────
#
# State prefix (include_seen=False) provides doc type and party names so the
# model uses consistent plain-English labels rather than copying raw role
# labels ("Licensor", "Executant") from each chunk.
# List-response guard prevents the crash seen when phi4-mini returns a JSON
# array instead of a string for the summary field.
#
async def _pass_summary(chunk: str, index: int, total: int, state: PipelineState) -> str:
    prefix = state.build_prefix(include_seen=False)
    prompt = f"""{prefix}You are a legal document analyst. Read the excerpt below — it is section {index} of {total} from a legal agreement.

Write a factual summary of ONLY what this section says. Your summary will be combined with summaries from other sections later.

STRICT RULES:
- Include ONLY information explicitly present in this section's text.
- Never invent, infer, or borrow facts from outside this excerpt.
- If a value (amount, date, name) is illegible or ambiguous, write "not clearly stated" — do not guess.
- Write 3–5 complete sentences in plain English for a non-lawyer.
- Use plain English role words: replace "Licensor" → "landlord/owner", "Licensee" → "tenant",
  "Consideration" → "payment", "Executant" → "signer", "Mortgagor" → "borrower", etc.
- Where the Document Context above lists known party names, use those names instead of role labels.

Output ONLY this JSON — no other text:
{{"summary": "Your 3–5 sentence plain-English summary here."}}

SECTION TEXT:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt, num_predict=512)
        raw = result.get("summary") or ""
        # Guard: model sometimes returns a list — flatten to string
        if isinstance(raw, list):
            raw = " ".join(str(s).strip() for s in raw if s)
        return raw.strip()
    except Exception as e:
        _log(f"PASS 1 — CHUNK {index} ERROR", str(e))
        return ""


# ─── Pass 2 — Entities ────────────────────────────────────────────────────────
#
# State prefix (include_seen=False) lists names confirmed from chunk 1.
# Later chunks can corroborate the spelling rather than independently
# re-extracting OCR variants of the same name.
# Three-condition test forces the model to reason through each candidate.
#
async def _pass_entities(chunk: str, state: PipelineState) -> dict:
    prefix = state.build_prefix(include_seen=False)
    prompt = f"""{prefix}You are extracting the names of real people and real organisations from a section of a legal document.

PERSONS — include ONLY if ALL three conditions are true:
  1. It is a full human name (at minimum: first name + last name, both clearly readable)
  2. It refers to a specific real individual, not a role ("the Owner", "Party A", "the Tenant")
  3. The name is free of OCR errors — no slashes, mixed scripts, or garbled characters

ORGANISATIONS — include ONLY if ALL three conditions are true:
  1. It is a registered or named entity (company, authority, society, trust, bank, firm)
  2. It is referred to by its actual name, not a generic label ("the Bank", "the Society")
  3. The name appears at least once clearly and completely in the text

REJECT any of the following — do not include them under any circumstances:
  - Role labels: Owner, Licensor, Licensee, Tenant, Landlord, Buyer, Seller, Party, Agent,
    Witness, Mortgagor, Employer, Employee, Borrower, Lender, Vendor, Contractor
  - Honorifics alone: "Mr.", "Mrs.", "Ms.", "Dr.", "Shri", "Smt.", "Adv."
  - Strings containing: /  \\  digits  @  mixed scripts  garbled characters
  - Partial names, single words, or strings that could be a common noun

If a name from the Document Context above appears in this section (even slightly differently
spelled), prefer the confirmed spelling from the context over the raw text.

When uncertain about any name — OMIT IT. An empty list is the correct answer when no clear names exist.

Output ONLY this JSON — no other text:
{{
  "persons": ["First Last"],
  "organizations": ["Organisation Name"]
}}

SECTION TEXT:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt, num_predict=256)
        return {
            "persons":       _fuzzy_unique(_clean_entities(result.get("persons", [])),       25),
            "organizations": _fuzzy_unique(_clean_entities(result.get("organizations", [])), 25),
        }
    except Exception as e:
        _log("PASS 2 — ERROR", str(e))
        return {"persons": [], "organizations": []}


# ─── Pass 3 — Clauses ─────────────────────────────────────────────────────────
#
# Sequential execution means state.seen_clauses grows chunk by chunk.
# The prefix tells the model exactly which clause types have already been
# reported — it skips them entirely rather than generating duplicates that
# need post-hoc dedup. This is the primary fix for the 2-clause-repeated-4x
# problem observed in the original pipeline logs.
# Rule 7 reinforces the "already found" prefix instruction at the rules level.
#
async def _pass_clauses(chunk: str, index: int, total: int, state: PipelineState) -> list[dict]:
    topics_numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(REFERENCE_CLAUSE_TOPICS))
    prefix = state.build_prefix(include_seen=True)

    prompt = f"""{prefix}You are a contract analyst. Read the legal text below (section {index} of {total}) and identify key clauses.

STEP 1 — IDENTIFY: Which of these clause types are actually present in the text below?
{topics_numbered}

STEP 2 — EXTRACT: For each clause type identified in Step 1, produce one entry:
  - "name"           : The exact clause type label from the numbered list above
  - "excerpt"        : 10–25 words copied VERBATIM from the text (character-for-character, no paraphrasing)
  - "plain_language" : One sentence explaining what this means practically for the weaker or paying party

ABSOLUTE RULES — violating any of these makes your output invalid:
  1. Only extract clauses whose text is VISIBLY PRESENT in the section below. Do not infer or assume.
  2. The "excerpt" must be a direct copy from the text. If you cannot find exact words, omit that clause.
  3. Each clause type appears AT MOST ONCE in your output.
  4. Maximum 5 clauses total.
  5. If no clause types from the list are present, return an empty array — that is correct.
  6. Do not include a clause just because it is common in such documents — it must be in THIS text.
  7. If a clause type appears under "Clauses already found" in the Document Context above,
     DO NOT include it — skip it entirely even if related text exists in this section.

Output ONLY this JSON — no other text:
{{
  "clauses": [
    {{
      "name": "exact topic name from numbered list",
      "excerpt": "verbatim words copied from the section text",
      "plain_language": "one plain sentence on what this means in practice"
    }}
  ]
}}

SECTION TEXT:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt, num_predict=800)
        raw = result.get("clauses", [])
        # Initialise local seen set from global state to block duplicates at generation time
        seen_names: set[str] = set(state.seen_clauses)
        cleaned = []
        for c in raw:
            name  = (c.get("name")          or "").strip()
            excpt = (c.get("excerpt")        or "").strip()
            plain = (c.get("plain_language") or "").strip()
            if not name or not plain:
                continue
            name_key = name.casefold()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            # Drop excerpts that look like OCR garbage
            if excpt and re.search(r"\b[A-Z]{5,}\b", excpt) and "/" in excpt:
                excpt = ""
            cleaned.append({"name": name, "excerpt": excpt, "plain_language": plain})
            if len(cleaned) >= 5:
                break
        return cleaned
    except Exception as e:
        _log(f"PASS 3 — CHUNK {index} ERROR", str(e))
        return []


# ─── Pass 4 — Risks ───────────────────────────────────────────────────────────
#
# Sequential execution means state.seen_risks grows chunk by chunk.
# "Weaker party" framing generalises beyond rental docs to employment,
# SaaS, loans, NDAs, and service contracts.
# found_keywords verbatim requirement eliminates the bracket-notation
# hallucination pattern seen in original logs: "(vague)", "(unlimited)".
# Rule 7 reinforces the "already found" prefix instruction at the rules level.
#
async def _pass_risks(chunk: str, index: int, total: int, state: PipelineState) -> list[dict]:
    prefix = state.build_prefix(include_seen=True)

    prompt = f"""{prefix}You are a legal risk analyst reviewing section {index} of {total} of a contract on behalf of the weaker party (typically: tenant, buyer, employee, borrower, or service recipient).

Identify terms in THIS SECTION that could harm, obligate, or surprise the weaker party.

For each risk, provide:
  - "label"           : A specific 4–8 word name describing THIS exact risk (not a generic category)
  - "severity"        : Exactly one of — "high" | "medium" | "low"
      high   → direct financial loss, forced exit, eviction, or legal liability for the weaker party
      medium → vague, one-sided, or potentially costly term the weaker party might overlook
      low    → minor restriction or unusual obligation with limited financial impact
  - "found_keywords"  : 1–3 short phrases (3–7 words each) copied VERBATIM from the text
  - "plain_language"  : One sentence — the specific real-world consequence for the weaker party

RISK CATEGORIES TO LOOK FOR (flag only if direct text evidence exists):
  - Price, fee, or rate increases without the weaker party's consent
  - Deposit or prepayment conditions that disproportionately favour the stronger party
  - Termination or exit rights available only to the stronger party
  - Charges that are vague, uncapped, or determined solely by the stronger party
  - Obligations to maintain, insure, or repair that fall entirely on the weaker party
  - Restrictions on the weaker party's rights (to sublet, assign, transfer, or exit)
  - Automatic renewal or lock-in terms the weaker party may not notice
  - Waivers of the weaker party's rights to dispute, appeal, or seek legal remedy
  - Unilateral right of the stronger party to change terms without consent

STRICT RULES:
  1. Every risk must be grounded in text you can quote verbatim — no inferences or assumptions.
  2. "found_keywords" must be phrases copied verbatim from the section — never paraphrased or invented.
  3. Each "label" must be unique and specific — not generic titles like "Risk 1" or "Unfair Clause".
  4. "severity" must be exactly "high", "medium", or "low" — no other values accepted.
  5. Maximum 5 risks per section.
  6. If no genuine risks are present in this section, return an empty array — that is correct.
  7. If a risk label appears under "Risks already found" in the Document Context above,
     DO NOT report it again — skip it even if you see related text in this section.

Output ONLY this JSON — no other text:
{{
  "risks": [
    {{
      "label": "Specific risk name in 4–8 words",
      "severity": "high",
      "found_keywords": ["exact phrase from text", "another verbatim phrase"],
      "plain_language": "One sentence: what could actually happen to the weaker party."
    }}
  ]
}}

SECTION TEXT:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt, num_predict=900)
        raw = result.get("risks", [])
        # Initialise local seen set from global state to block duplicates at generation time
        seen_labels: set[str] = set(state.seen_risks)
        cleaned = []
        for r in raw:
            label = (r.get("label")          or "").strip()
            plain = (r.get("plain_language")  or "").strip()
            if not label or not plain:
                continue
            label_key = label.casefold()
            if label_key in seen_labels:
                continue
            seen_labels.add(label_key)
            sev = _normalise_severity((r.get("severity") or "").lower().strip())
            cleaned.append({
                "label":          label,
                "severity":       sev,
                "found_keywords": _fuzzy_unique(r.get("found_keywords", []), 3),
                "plain_language": plain,
            })
            if len(cleaned) >= 5:
                break
        return cleaned
    except Exception as e:
        _log(f"PASS 4 — CHUNK {index} ERROR", str(e))
        return []


# ─── Pass 5 — Synthesis ───────────────────────────────────────────────────────
#
# State provides confirmed doc_type, parties, key_dates, and financials from
# the fingerprint pass — synthesis uses these directly rather than re-deriving
# from potentially garbled chunk summaries.
# "Candidate party names" framing signals the merge hints are unverified.
# Discrepancy rule prevents silent figure selection when sections conflict.
# List-response guard mirrors the Pass 1 fix.
#
async def _pass_synthesis(evidence: dict, state: PipelineState) -> tuple[str, list, list]:
    numbered = "\n".join(
        f"[Section {item['chunk']}]: {item['summary']}"
        for item in evidence.get("chunk_summaries", [])
        if item.get("summary")
    )
    persons_hint = ", ".join(evidence.get("persons", [])) or "none identified"
    orgs_hint    = ", ".join(evidence.get("organizations", [])) or "none identified"

    # Build a confirmed-facts block from state — more reliable than post-merge hints
    confirmed_lines = []
    if state.doc_type:
        confirmed_lines.append(f"  Document type     : {state.doc_type}")
    all_confirmed = _fuzzy_unique(state.parties + state.persons + state.orgs, 12)
    if all_confirmed:
        confirmed_lines.append(f"  Confirmed parties : {', '.join(all_confirmed)}")
    if state.key_dates:
        confirmed_lines.append(f"  Confirmed dates   : {', '.join(state.key_dates)}")
    if state.financials:
        confirmed_lines.append(f"  Confirmed amounts : {', '.join(state.financials)}")

    confirmed_block = (
        "CONFIRMED FACTS (extracted directly from document opening — use these as ground truth):\n"
        + "\n".join(confirmed_lines) + "\n\n"
        if confirmed_lines else ""
    )

    prompt = f"""{confirmed_block}You are writing the final plain-English summary of a legal document from multiple section summaries.

SECTION SUMMARIES (each is one part of the same document):
{numbered}

ADDITIONAL CANDIDATE NAMES (unverified — use only if consistent across multiple sections):
  Persons: {persons_hint}
  Organisations: {orgs_hint}

YOUR TASK:
Write a single coherent summary (6–8 sentences) that integrates ALL sections above.

Cover each of the following points — skip only if genuinely absent across ALL sections:
  1. Document type and subject (what property, service, relationship, or transaction it governs)
  2. Named parties — who is the stronger party and who is the weaker party; use real names from
     Confirmed Facts where available, otherwise use clear role labels (landlord, tenant, etc.)
  3. Duration — start date, end date, and length of the initial term
  4. Financial terms — all amounts (prefer Confirmed Facts; add any others from summaries),
     payment schedule, and currency
  5. Renewal — whether and how the agreement renews, and on whose initiative
  6. Termination — notice period required, and whether both parties share this right equally
  7. Key obligations — anything unusual or significant required of either party beyond routine terms

RULES:
  - Prefer Confirmed Facts over section summaries for names, amounts, and doc type.
  - Draw from ALL section summaries for clauses, obligations, and narrative detail.
  - If two sections mention conflicting figures for the same item, include both and note the discrepancy.
  - Use specific numbers and dates — avoid vague phrases like "a certain amount" or "some period".
  - Replace all legal jargon with plain English equivalents throughout.
  - Do not repeat the same fact twice.
  - Write 6–8 sentences total.

For "persons" and "organizations": return ONLY the real names of parties clearly identified.
Omit role labels, honorifics, and any name that appears garbled, partial, or uncertain.

Output ONLY this JSON — no other text:
{{
  "summary": "6–8 sentence plain-English summary here.",
  "persons": ["Full Name"],
  "organizations": ["Organisation Name"]
}}
"""
    try:
        payload = await _run_ollama_json_async(prompt, num_predict=700)
    except Exception as e:
        _log("PASS 5 — ERROR", str(e))
        payload = {}

    summary = (payload.get("summary") or "").strip()
    # Guard: model sometimes returns a list — flatten to string
    if isinstance(summary, list):
        summary = " ".join(str(s).strip() for s in summary if s)
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

    # Final fallback: use parties confirmed by the fingerprint pass
    if not persons and not orgs and state.parties:
        persons = _fuzzy_unique(_clean_entities(state.parties), 25)

    return summary, persons, orgs


def _empty_result() -> dict:
    return {
        "summary": "",
        "entities": {"persons": [], "organizations": []},
        "clauses":  [],
        "risks":    [],
    }


# ─── Ollama async client ──────────────────────────────────────────────────────
#
# num_ctx MUST be >= chunk size in tokens.
# 4000 chars ≈ 700–900 tokens. State prefix adds ~200–350 tokens.
# 4096 ctx gives solid headroom for chunk + prefix + prompt shell + output.
# 2048 was silently truncating — the #1 original cause of hallucinated output.
#
async def _run_ollama_json_async(prompt: str, num_predict: int = 1024) -> dict:
    """
    num_predict caps the output token count per call.
    Each pass gets a different budget:
      - Fingerprint                    : 300  (4 small flat fields)
      - Summary / entities / synthesis : 512  (short outputs)
      - Clauses                        : 800  (5 clauses × ~160 tokens each)
      - Risks                          : 900  (5 risks   × ~180 tokens each)
    Without this cap phi4-mini enters a generation loop inside JSON arrays,
    producing 15+ duplicate entries and wasting 60–70 seconds per chunk.
    """
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
                        "Do not add markdown formatting or code fences. "
                        "Stop generating immediately after the closing brace of the JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
            format="json",
            options={
                "temperature":    0.05,
                "num_ctx":        4096,
                "num_predict":    num_predict,
                "repeat_penalty": 1.15,
                "top_p":          0.9,
                "stop":           ["}]}"],
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


# ─── Post-processing helpers ──────────────────────────────────────────────────

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
        "employer", "employee", "borrower", "lender", "vendor",
        "client", "customer", "contractor", "subcontractor",
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
        if re.match(r"^(Mr\.|Mrs\.|Ms\.|Dr\.|Shri|Smt\.)\s*$", item, re.IGNORECASE):
            continue
        if item.isupper() and len(item) <= 5:
            continue
        cleaned.append(item)
    return cleaned


# ─── Text helpers ─────────────────────────────────────────────────────────────

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
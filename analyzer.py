"""
Legal document analysis — 5-pass pipeline + Progressive Context Accumulator
============================================================================

Architecture
------------
Pass 0 — Fingerprint : Run on chunk 1 only. Extracts doc type, ROLE-AWARE parties
                       (stronger/weaker), dates, and financial amounts.
                       Seeds the shared PipelineState with structured facts.

Pass 1 — Summary     : Parallel across all chunks. State prefix injects confirmed
                       party names AND their roles so the model never guesses who
                       is landlord vs tenant.

Pass 2 — Entities    : Parallel across all chunks. Post-processing includes a
                       rescue step that moves misclassified persons out of the
                       organizations bucket (llama3.2 pattern).

Pass 3 — Clauses     : Sequential across chunks. All model-generated clause names
                       are normalised back to the canonical REFERENCE_CLAUSE_TOPICS
                       list before being accepted. Invented names are dropped.

Pass 4 — Risks       : Sequential across chunks. Fuzzy dedup runs in the main
                       pipeline loop against actual state (not a snapshot), so
                       near-duplicate labels are blocked even when casing differs.

Pass 5 — Synthesis   : Single call. Receives a CONFIRMED FACTS block that includes
                       role-labelled party names as ground truth, so it cannot
                       invert stronger/weaker party assignments.

Fixes applied vs previous version
----------------------------------
1. Role inversion (Mahesh labelled landlord when he is tenant) — Pass 0 now
   extracts {"name": ..., "role": "licensor|licensee"} and PipelineState exposes
   stronger_party() / weaker_party() helpers used by all downstream passes.

2. Persons placed in organizations bucket by llama3.2 — _rescue_misclassified_entities()
   runs after Pass 2 merge and moves human-name-shaped strings back to persons.

3. Invented clause names breaking dedup — _normalise_clause_name() maps free-text
   names back to the canonical topic list; unmatched names are dropped entirely.

4. Near-duplicate risks slipping through state filter — fuzzy dedup now runs in
   the main loop against live state, not inside the pass against a snapshot.

5. Single-word noise ("Indian", "Police") surviving _clean_entities — added
   _SINGLE_WORD_NOISE blocklist.
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nemotron-3-nano:4b")


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

# Canonical clause topic list — model output is normalised back to these strings.
# Any clause name the model invents that cannot be mapped here is dropped.
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

# Pre-built lookup for fast normalisation: lowercase → canonical form
_TOPICS_LOWER: dict[str, str] = {t.lower(): t for t in REFERENCE_CLAUSE_TOPICS}

# Role sets used to classify parties extracted from the fingerprint
_STRONGER_ROLES = {
    "licensor", "landlord", "lender", "employer", "owner", "lessor",
    "seller", "franchisor", "service provider", "vendor",
}
_WEAKER_ROLES = {
    "licensee", "tenant", "borrower", "employee", "lessee",
    "buyer", "franchisee", "client", "customer",
}

# Single-word strings that look like proper nouns but are not entity names
_SINGLE_WORD_NOISE = {
    "indian", "police", "government", "court", "authority", "ministry",
    "board", "committee", "society", "association", "federation", "union",
    "department", "office", "bureau", "agency", "institute", "council",
    "national", "municipal", "state", "central", "local", "public",
}

# Organisation suffixes — strings with these are genuine orgs, not misclassified persons
_ORG_SUFFIX_RE = re.compile(
    r"\b(Ltd|Pvt|Inc|Corp|LLC|LLP|Chs|Ngo|Trust|Society|Authority|Bank|"
    r"Board|Commission|Institute|Foundation|Services|Solutions|Enterprises|"
    r"Industries|Properties|Realty|Housing|Cooperative)\b",
    re.IGNORECASE,
)

# Words that flag a string as a law/government reference, not a party name
_LAW_OR_GOVT_RE = re.compile(
    r"\b(Act|Indian|Police|Government|Ministry|Court|Section|Rule|Regulation|"
    r"Gazette|Schedule|Clause|Article|Statute)\b",
    re.IGNORECASE,
)


# ─── Progressive Context Accumulator ─────────────────────────────────────────
#
# Key change from previous version: parties is now list[dict] with role info.
# This prevents synthesis from guessing who is landlord vs tenant.
#

@dataclass
class PipelineState:
    # Populated by Pass 0 — role-aware
    doc_type:    str         = ""
    parties:     list[dict]  = field(default_factory=list)
    # Each party: {"name": "Mahesh Patil", "role": "licensor"}
    # role is one of the strings in _STRONGER_ROLES or _WEAKER_ROLES

    key_dates:   list[str]   = field(default_factory=list)
    financials:  list[str]   = field(default_factory=list)

    # Populated by Pass 2 merge (after rescue step)
    persons:     list[str]   = field(default_factory=list)
    orgs:        list[str]   = field(default_factory=list)

    # Written by Pass 3 (sequential) — casefolded canonical clause names
    seen_clauses: set[str]   = field(default_factory=set)

    # Written by Pass 4 (sequential) — casefolded risk labels
    seen_risks:   set[str]   = field(default_factory=set)

    # ── Role helpers ──────────────────────────────────────────────────────────

    def stronger_party(self) -> str:
        """Name of the party with the stronger contractual position, if known."""
        for p in self.parties:
            if p.get("role", "").lower() in _STRONGER_ROLES:
                return p.get("name", "")
        return ""

    def weaker_party(self) -> str:
        """Name of the party with the weaker contractual position, if known."""
        for p in self.parties:
            if p.get("role", "").lower() in _WEAKER_ROLES:
                return p.get("name", "")
        return ""

    def all_party_names(self) -> list[str]:
        return [p["name"] for p in self.parties if p.get("name")]

    # ── Prefix builder ────────────────────────────────────────────────────────

    def build_prefix(self, include_seen: bool = True) -> str:
        """
        Render a compact, role-labelled context block (~200–400 tokens) to
        prepend to any LLM prompt. include_seen=False for passes 1, 2, 5.
        Returns empty string if there is nothing useful to add.
        """
        lines = ["DOCUMENT CONTEXT (confirmed facts — treat as ground truth):"]

        if self.doc_type:
            lines.append(f"  Document type  : {self.doc_type}")

        # Role-labelled parties — the critical addition vs previous version
        sp = self.stronger_party()
        wp = self.weaker_party()
        if sp:
            lines.append(f"  Stronger party : {sp}  (landlord / owner / lender side)")
        if wp:
            lines.append(f"  Weaker party   : {wp}  (tenant / borrower / employee side)")
        # Parties whose role is ambiguous
        ambiguous = [
            p["name"] for p in self.parties
            if p.get("name")
            and p.get("role", "").lower() not in _STRONGER_ROLES | _WEAKER_ROLES
        ]
        if ambiguous:
            lines.append(f"  Other parties  : {', '.join(ambiguous)}")

        # Confirmed persons and orgs from Pass 2
        extra_persons = [n for n in self.persons if n not in self.all_party_names()]
        if extra_persons:
            lines.append(f"  Other persons  : {', '.join(extra_persons)}")
        if self.orgs:
            lines.append(f"  Organisations  : {', '.join(self.orgs)}")

        if self.key_dates:
            lines.append(f"  Key dates      : {', '.join(self.key_dates)}")

        if self.financials:
            lines.append(f"  Known amounts  : {', '.join(self.financials)}")

        if include_seen:
            if self.seen_clauses:
                sc = ", ".join(sorted(self.seen_clauses))
                lines.append(
                    f"  Clauses already found (DO NOT repeat): {sc}"
                )
            if self.seen_risks:
                sr = ", ".join(sorted(self.seen_risks))
                lines.append(
                    f"  Risks already found (DO NOT repeat): {sr}"
                )

        if len(lines) == 1:
            return ""

        return "\n".join(lines) + "\n\n"

    # ── State update ──────────────────────────────────────────────────────────

    def update_from_fingerprint(self, fp: dict) -> None:
        self.doc_type  = (fp.get("doc_type") or "").strip()
        self.key_dates = [str(d).strip() for d in fp.get("key_dates",  []) if d][:6]
        self.financials= [str(f).strip() for f in fp.get("financials", []) if f][:6]

        raw_parties = fp.get("parties", [])
        self.parties = []
        for p in raw_parties:
            if not isinstance(p, dict):
                continue
            name = _clean_entities([p.get("name", "")])[:]
            role = (p.get("role") or "").lower().strip()
            if name:
                self.parties.append({"name": name[0], "role": role})

        _log(
            "PASS 0 — STATE SEEDED",
            f"  doc_type       : {self.doc_type}\n"
            f"  stronger party : {self.stronger_party() or '(unclear)'}\n"
            f"  weaker party   : {self.weaker_party()   or '(unclear)'}\n"
            f"  all parties    : {self.parties}\n"
            f"  key_dates      : {self.key_dates}\n"
            f"  financials     : {self.financials}",
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
    _log("PASS 0 — FINGERPRINT", "Extracting doc type, role-aware parties, dates, amounts...")
    fp = await _pass_fingerprint(chunks[0])
    state.update_from_fingerprint(fp)

    # ── Pass 1 — Summary (parallel) ───────────────────────────────────────────
    _log("PASS 1 — SUMMARY", f"Summarising {total} chunk(s) in parallel...")
    summaries = await _parallel([
        _pass_summary(c, i + 1, total, state) for i, c in enumerate(chunks)
    ])
    for i, s in enumerate(summaries):
        _log(
            f"PASS 1 — CHUNK {i+1}/{total}",
            (s[:200] + "...") if len(s) > 200 else s or "(empty)",
        )

    # ── Pass 2 — Entities (parallel) ──────────────────────────────────────────
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

    # Rescue persons misclassified as orgs (llama3.2 pattern)
    persons, orgs = _rescue_misclassified_entities(
        _clean_entities(raw_persons),
        _clean_entities(raw_orgs),
    )
    persons = _fuzzy_unique(persons, 40)
    orgs    = _fuzzy_unique(orgs,    40)
    state.persons = persons
    state.orgs    = orgs
    _log("PASS 2 — MERGED (after rescue)", f"  persons: {persons}\n  orgs:    {orgs}")

    # ── Pass 3 — Clauses (sequential) ─────────────────────────────────────────
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

    # ── Pass 4 — Risks (sequential, fuzzy dedup in loop) ──────────────────────
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
        # Fuzzy dedup runs here against live state — not inside the pass function
        for r in found:
            label_key = r["label"].casefold()
            if label_key in state.seen_risks:
                continue
            # Block near-duplicates: "Unfair Security Deposit" vs "Unfair Security Deposit Clause"
            if any(
                _levenshtein_ratio(label_key, seen) >= 0.80
                for seen in state.seen_risks
            ):
                _log(
                    f"PASS 4 — DEDUP",
                    f"  Fuzzy-blocked: '{r['label']}' (too similar to existing risk)",
                )
                continue
            state.seen_risks.add(label_key)
            raw_risks.append(r)

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
# KEY CHANGE: parties schema is now role-aware.
# Each party object has "name" AND "role" (licensor/licensee/employer/employee/etc).
# This is the single most important fix — it gives the pipeline ground truth about
# who holds the stronger position, preventing synthesis role inversions.
#
async def _pass_fingerprint(chunk: str) -> dict:
    prompt = f"""You are reading the opening section of a legal document.

Extract the following four facts from the text below.

─────────────────────────────────────────────────────────
FIELD 1: "doc_type"
What kind of legal agreement is this?
Examples: "residential lease", "employment contract", "loan agreement",
          "SaaS subscription agreement", "non-disclosure agreement",
          "leave and license agreement", "service agreement"
Write 2–5 words only. If unclear, write "legal agreement".

─────────────────────────────────────────────────────────
FIELD 2: "parties"
A list of party objects. Each party MUST have BOTH fields:
  - "name" : Full name of the real person or registered organisation
             Include ONLY names that are clearly readable — no role labels,
             no OCR noise, no garbled strings. If no clean name is readable
             for a party, use an empty string for that party's name.
  - "role" : Their contractual role — use EXACTLY one of these words:
             licensor  | licensee  | landlord   | tenant
             employer  | employee  | lender     | borrower
             seller    | buyer     | franchisor | franchisee
             lessor    | lessee    | vendor     | client
             If their role is unclear, write "unknown"

HOW TO IDENTIFY ROLES:
- The party GRANTING permission, property, or services is the STRONGER party
  (licensor, landlord, employer, lender, seller, lessor, franchisor, vendor)
- The party RECEIVING permission, property, or services is the WEAKER party
  (licensee, tenant, employee, borrower, buyer, lessee, franchisee, client)
- In a Leave & License Agreement: the property owner = licensor, the occupant = licensee
- In an Employment Contract: the company = employer, the worker = employee
- In a Loan Agreement: the bank = lender, the borrower = borrower

─────────────────────────────────────────────────────────
FIELD 3: "key_dates"
Any dates or term lengths mentioned (signing, start, end, duration).
E.g. "1 Jan 2024", "12-month term", "expires 31 Dec 2024"
Maximum 4 entries. Empty list if none.

─────────────────────────────────────────────────────────
FIELD 4: "financials"
Any monetary amounts mentioned (rent, fees, deposit, salary, penalty).
E.g. "Rs. 10,000/month", "security deposit Rs. 50,000", "annual fee USD 1200"
Maximum 4 entries. Empty list if none.

─────────────────────────────────────────────────────────
RULES:
- Only include information clearly present in the text — no inference or invention.
- For "parties": if a name is garbled or unreadable, it is better to have
  the role with an empty name than to invent or guess a name.

Output ONLY this JSON — no other text:
{{
  "doc_type"  : "type of agreement",
  "parties"   : [
    {{"name": "Full Name or Org Name", "role": "licensor"}},
    {{"name": "Full Name or Org Name", "role": "licensee"}}
  ],
  "key_dates" : ["date or term string"],
  "financials": ["amount string"]
}}

SECTION TEXT:
\"\"\"
{chunk}
\"\"\"
"""
    try:
        result = await _run_ollama_json_async(prompt, num_predict=400)
        return result
    except Exception as e:
        _log("PASS 0 — ERROR", str(e))
        return {}


# ─── Pass 1 — Summary ─────────────────────────────────────────────────────────
#
# Role-labelled prefix means the model sees "Stronger party: Mahesh Patil (landlord)"
# and "Weaker party: [name] (tenant)" — it can use correct role words without
# having to infer them from ambiguous OCR text.
#
async def _pass_summary(chunk: str, index: int, total: int, state: PipelineState) -> str:
    prefix = state.build_prefix(include_seen=False)
    prompt = f"""{prefix}You are a legal document analyst. Read the excerpt below — section {index} of {total} from a legal agreement.

Write a factual summary of ONLY what this section says. Your summary will be combined with other sections later.

STRICT RULES:
- Include ONLY information explicitly present in this section's text.
- Never invent, infer, or borrow facts from outside this excerpt.
- If a value (amount, date, name) is illegible or ambiguous, write "not clearly stated".
- Write 3–5 complete sentences in plain English for a non-lawyer.
- Use plain English: "Licensor" → use the Stronger Party name or "landlord/owner",
  "Licensee" → use the Weaker Party name or "tenant", "Consideration" → "payment", etc.
- Use the role-labelled party names from Document Context above — do NOT invert them.

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
        if isinstance(raw, list):
            raw = " ".join(str(s).strip() for s in raw if s)
        return raw.strip()
    except Exception as e:
        _log(f"PASS 1 — CHUNK {index} ERROR", str(e))
        return ""


# ─── Pass 2 — Entities ────────────────────────────────────────────────────────
#
# Three-condition test unchanged. The key improvement is the post-processing
# rescue step applied AFTER this function returns (in the main pipeline loop).
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
  - Strings with: /  \\  digits  @  mixed scripts  garbled characters
  - Single generic words: "Indian", "Police", "Government", "Court", "Authority"
  - Law names: "Indian Contract Act", "Maharashtra Rent Control Act" — these are laws, not parties
  - Partial names, single words, or strings that could be a common noun

If a name from Document Context appears in this section (even slightly differently spelled),
prefer the confirmed spelling from the context.

When uncertain — OMIT. An empty list is the correct answer when no clear names exist.

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
# Model output is now normalised to canonical topic names via _normalise_clause_name()
# before being accepted. Invented names ("security deposit", "delivery of possession")
# are silently dropped. This keeps state.seen_clauses clean and dedup accurate.
#
async def _pass_clauses(chunk: str, index: int, total: int, state: PipelineState) -> list[dict]:
    topics_numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(REFERENCE_CLAUSE_TOPICS))
    prefix = state.build_prefix(include_seen=True)

    prompt = f"""{prefix}You are a contract analyst. Read the legal text below (section {index} of {total}) and identify key clauses.

STEP 1 — IDENTIFY: Which of these clause types are actually present in the text below?
{topics_numbered}

STEP 2 — EXTRACT: For each clause type identified in Step 1, produce one entry:
  - "name"           : Use the EXACT label from the numbered list above — copy it character
                       for character. Do not invent, shorten, or rephrase the name.
  - "excerpt"        : 10–25 words copied VERBATIM from the text (no paraphrasing)
  - "plain_language" : One sentence explaining what this means for the weaker/paying party

ABSOLUTE RULES:
  1. Only extract clauses VISIBLY PRESENT in the section below — no inference.
  2. "excerpt" must be copied directly from the text. Omit the clause if you cannot.
  3. "name" must be exactly one of the numbered labels — no other names accepted.
  4. Each clause type appears AT MOST ONCE in your output.
  5. Maximum 5 clauses total.
  6. Return an empty array if no clause types from the list are present.
  7. Skip any clause type listed under "Clauses already found" in Document Context.

Output ONLY this JSON — no other text:
{{
  "clauses": [
    {{
      "name": "exact label from numbered list",
      "excerpt": "verbatim words from the section text",
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
        seen_names: set[str] = set(state.seen_clauses)
        cleaned = []
        for c in raw:
            raw_name = (c.get("name")          or "").strip()
            excpt    = (c.get("excerpt")        or "").strip()
            plain    = (c.get("plain_language") or "").strip()

            # Normalise to canonical topic — drops invented names
            name = _normalise_clause_name(raw_name)
            if not name or not plain:
                if raw_name and not name:
                    _log(
                        f"PASS 3 — CHUNK {index} DROPPED",
                        f"  Invented clause name rejected: '{raw_name}'",
                    )
                continue

            name_key = name.casefold()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)

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
# The pass function itself is unchanged. The key fix is that fuzzy dedup now
# runs in the main pipeline loop (after this returns) against live state,
# not inside this function against a snapshot.
#
async def _pass_risks(chunk: str, index: int, total: int, state: PipelineState) -> list[dict]:
    prefix = state.build_prefix(include_seen=True)

    prompt = f"""{prefix}You are a legal risk analyst reviewing section {index} of {total} on behalf of the weaker party (tenant, buyer, employee, borrower, or service recipient).

Use the role-labelled party names from Document Context above — the Stronger Party is the one
who holds power; the Weaker Party is the one at risk. Do not invert these roles.

Identify terms in THIS SECTION that could harm, obligate, or surprise the weaker party.

For each risk, provide:
  - "label"           : A specific 4–8 word name for THIS exact risk (not a generic category)
  - "severity"        : Exactly one of — "high" | "medium" | "low"
      high   → direct financial loss, forced exit, eviction, or legal liability
      medium → vague, one-sided, or potentially costly term easily overlooked
      low    → minor restriction or unusual obligation with limited financial impact
  - "found_keywords"  : 1–3 short phrases (3–7 words) copied VERBATIM from the text
  - "plain_language"  : One sentence — the real-world consequence for the weaker party

RISK CATEGORIES (flag only if direct text evidence exists):
  - Price, fee, or rate increases without the weaker party's consent
  - Deposit or prepayment conditions favouring the stronger party
  - Termination rights available only to the stronger party
  - Charges that are vague, uncapped, or set unilaterally by the stronger party
  - Obligations to maintain, insure, or repair falling entirely on the weaker party
  - Restrictions on the weaker party's rights (sublet, assign, transfer, exit)
  - Automatic renewal or lock-in terms the weaker party may not notice
  - Waivers of the weaker party's rights to dispute, appeal, or seek remedy
  - Unilateral right to change terms without the weaker party's consent

STRICT RULES:
  1. Every risk must be grounded in text you can quote verbatim.
  2. "found_keywords" must be verbatim phrases — never paraphrased or invented.
  3. Each "label" must be unique and specific.
  4. "severity" must be exactly "high", "medium", or "low".
  5. Maximum 5 risks per section.
  6. Return an empty array if no genuine risks are present.
  7. Skip any risk label listed under "Risks already found" in Document Context.

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
        # Local dedup against snapshot — catches exact duplicates within the same chunk
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
# CONFIRMED FACTS block now includes role-labelled party names as explicit
# ground truth. The synthesis model is told "Stronger party IS X, Weaker party
# IS Y" — it cannot invert or guess these from ambiguous summaries.
#
async def _pass_synthesis(evidence: dict, state: PipelineState) -> tuple[str, list, list]:
    numbered = "\n".join(
        f"[Section {item['chunk']}]: {item['summary']}"
        for item in evidence.get("chunk_summaries", [])
        if item.get("summary")
    )
    persons_hint = ", ".join(evidence.get("persons", [])) or "none identified"
    orgs_hint    = ", ".join(evidence.get("organizations", [])) or "none identified"

    # Build confirmed-facts block with role-labelled parties
    confirmed_lines = []
    if state.doc_type:
        confirmed_lines.append(f"  Document type  : {state.doc_type}")

    sp = state.stronger_party()
    wp = state.weaker_party()
    if sp:
        confirmed_lines.append(
            f"  Stronger party : {sp}  ← this party is the landlord/owner/lender side"
        )
    if wp:
        confirmed_lines.append(
            f"  Weaker party   : {wp}  ← this party is the tenant/borrower/employee side"
        )
    ambiguous = [
        p["name"] for p in state.parties
        if p.get("name")
        and p.get("role", "").lower() not in _STRONGER_ROLES | _WEAKER_ROLES
    ]
    if ambiguous:
        confirmed_lines.append(f"  Other parties  : {', '.join(ambiguous)}")

    if state.key_dates:
        confirmed_lines.append(f"  Confirmed dates  : {', '.join(state.key_dates)}")
    if state.financials:
        confirmed_lines.append(f"  Confirmed amounts: {', '.join(state.financials)}")

    confirmed_block = (
        "CONFIRMED FACTS — use these as absolute ground truth. "
        "Do NOT contradict or swap the stronger/weaker party roles:\n"
        + "\n".join(confirmed_lines) + "\n\n"
        if confirmed_lines else ""
    )

    prompt = f"""{confirmed_block}You are writing the final plain-English summary of a legal document from section summaries.

SECTION SUMMARIES:
{numbered}

ADDITIONAL CANDIDATE NAMES (unverified — use only if consistent across multiple sections):
  Persons: {persons_hint}
  Organisations: {orgs_hint}

YOUR TASK:
Write a single coherent summary of 6–8 sentences integrating ALL sections above.

Cover each of the following — skip only if genuinely absent across ALL sections:
  1. Document type and subject
  2. Named parties — use the Stronger/Weaker party names from Confirmed Facts above;
     clearly state who is the landlord/owner/lender and who is the tenant/borrower/employee.
     IMPORTANT: Do not swap or invert these roles — they are confirmed facts.
  3. Duration — start date, end date, and term length
  4. Financial terms — all amounts (prefer Confirmed Facts; add others from summaries),
     payment schedule, currency
  5. Renewal — whether and how the agreement renews
  6. Termination — notice period, whether both parties share this right equally
  7. Key obligations — anything unusual required of either party

RULES:
  - Confirmed Facts override section summaries for names, roles, amounts, and doc type.
  - Draw from ALL section summaries for obligations and narrative detail.
  - If sections mention conflicting figures, include both and note the discrepancy.
  - Use specific numbers and dates — no vague phrases like "a certain amount".
  - Plain English throughout — no legal jargon.
  - Do not repeat any fact twice.

For "persons" and "organizations": return ONLY real names of clearly identified parties.
Omit role labels, honorifics, garbled or uncertain names.

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

    # Final fallback: pull names from confirmed parties in state
    if not persons and state.parties:
        persons = _fuzzy_unique(
            _clean_entities([p["name"] for p in state.parties if p.get("name")]), 25
        )

    return summary, persons, orgs


def _empty_result() -> dict:
    return {
        "summary": "",
        "entities": {"persons": [], "organizations": []},
        "clauses":  [],
        "risks":    [],
    }


# ─── Ollama async client ──────────────────────────────────────────────────────

async def _run_ollama_json_async(prompt: str, num_predict: int = 1024) -> dict:
    """
    num_predict per-pass budgets:
      Pass 0 Fingerprint  : 400  (role-aware party objects need more tokens)
      Pass 1 Summary      : 512
      Pass 2 Entities     : 256
      Pass 3 Clauses      : 800  (5 clauses × ~160 tokens)
      Pass 4 Risks        : 900  (5 risks   × ~180 tokens)
      Pass 5 Synthesis    : 700
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


# ─── Entity post-processing helpers ──────────────────────────────────────────

def _rescue_misclassified_entities(
    persons: list[str], orgs: list[str]
) -> tuple[list[str], list[str]]:
    """
    llama3.2 (and other small models) frequently place human names in the
    organizations bucket. This function moves them back.

    Rules:
    - Strings with legal suffixes (Ltd, Pvt, Chs, Trust, etc.) → stay as orgs
    - Strings matching law/government patterns → dropped entirely
    - Strings that look like two-or-more capitalised words with no legal suffix
      → moved to persons
    - Single generic words → dropped
    """
    clean_persons = list(persons)
    clean_orgs: list[str] = []

    for org in orgs:
        words = org.strip().split()

        # Drop law names and generic government words
        if _LAW_OR_GOVT_RE.search(org):
            _log("PASS 2 — RESCUE", f"  Dropped law/govt noise: '{org}'")
            continue

        # Drop single generic words
        if len(words) == 1 and org.lower() in _SINGLE_WORD_NOISE:
            _log("PASS 2 — RESCUE", f"  Dropped single-word noise: '{org}'")
            continue

        # Keep genuine orgs (have a legal suffix)
        if _ORG_SUFFIX_RE.search(org):
            clean_orgs.append(org)
            continue

        # Two+ capitalised words with no legal suffix → likely a person name
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            _log("PASS 2 — RESCUE", f"  Moved org→person: '{org}'")
            clean_persons.append(org)
            continue

        # Single unrecognised word → drop
        if len(words) == 1:
            _log("PASS 2 — RESCUE", f"  Dropped single unknown word: '{org}'")
            continue

        # Anything else with lowercase words is probably a real org name
        clean_orgs.append(org)

    return clean_persons, clean_orgs


def _normalise_clause_name(raw_name: str) -> str:
    """
    Map a model-generated clause name back to the canonical REFERENCE_CLAUSE_TOPICS list.
    Returns the canonical name string if matched, empty string if no match found.

    Match strategy (in order):
    1. Exact lowercase match
    2. The raw name is fully contained within a canonical name
    3. A canonical name's first keyword is contained in the raw name
    4. Word-overlap score >= 0.5 (more than half the words match)
    """
    key = raw_name.lower().strip()
    if not key:
        return ""

    # 1. Exact match
    if key in _TOPICS_LOWER:
        return _TOPICS_LOWER[key]

    # 2. Raw name is a substring of a canonical topic
    for canonical_lower, canonical in _TOPICS_LOWER.items():
        if key in canonical_lower:
            return canonical

    # 3. First significant keyword of canonical is in raw name
    for canonical_lower, canonical in _TOPICS_LOWER.items():
        first_word = canonical_lower.split()[0]
        if len(first_word) >= 5 and first_word in key:
            return canonical

    # 4. Word overlap >= 50%
    raw_words  = set(key.split())
    for canonical_lower, canonical in _TOPICS_LOWER.items():
        canon_words = set(canonical_lower.split())
        overlap = len(raw_words & canon_words)
        if overlap >= 1 and overlap / max(len(raw_words), len(canon_words)) >= 0.5:
            return canonical

    return ""   # No match — caller will drop this clause


# ─── General post-processing helpers ─────────────────────────────────────────

def _normalise_severity(sev: str) -> str:
    if sev in VALID_SEVERITIES:
        return sev
    if "high" in sev:
        return "high"
    if "low" in sev:
        return "low"
    return "medium"


def _clean_entities(values: list) -> list:
    """
    Remove OCR noise, role labels, single-word noise, and garbage strings.
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
        if item.lower() in _SINGLE_WORD_NOISE:
            continue
        if re.match(r"^(Mr\.|Mrs\.|Ms\.|Dr\.|Shri|Smt\.)\s*$", item, re.IGNORECASE):
            continue
        if _LAW_OR_GOVT_RE.search(item) and len(item.split()) <= 3:
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
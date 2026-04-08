"""
analyzer.py
-----------
Gemini-powered legal document analysis:
  - Document summary
  - Clause detection
  - Risk detection
  - Named entity extraction
  - Risk scoring
"""

import json
import io
import os

from dotenv import load_dotenv
from google import genai



load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
# GEMINI_MODEL = "gemini-2.5-flash-lite"

MAX_ANALYSIS_CHARS = 120000

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

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {
            "type": "object",
            "properties": {
                "persons": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "organizations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["persons", "organizations"],
        },
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "excerpt": {"type": "string"},
                    "plain_language": {"type": "string"},
                },
                "required": ["name", "excerpt", "plain_language"],
            },
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "found_keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "plain_language": {"type": "string"},
                },
                "required": ["label", "severity", "found_keywords", "plain_language"],
            },
        },
    },
    "required": ["summary", "entities", "clauses", "risks"],
}


def analyze_document(text: str) -> dict:
    """
    Analyze a legal document with Gemini and return a structured result.
    """
    client = _build_client()
    prepared_text = text[:MAX_ANALYSIS_CHARS]

    prompt = _build_analysis_prompt(
        source_label="document text",
        source_payload=f'Document:\n"""\n{prepared_text}\n"""',
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ANALYSIS_SCHEMA,
            "temperature": 0.2,
        },
    )

    payload = json.loads(response.text)
    return _normalize_analysis(payload)


def analyze_pdf(uploaded_file) -> dict:
    """
    Analyze an uploaded PDF directly with Gemini Files API.
    This allows Gemini to inspect scanned/image-based pages.
    """
    client = _build_client()

    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes:
        raise RuntimeError("Uploaded PDF is empty.")

    pdf_buffer = io.BytesIO(pdf_bytes)
    pdf_buffer.name = getattr(uploaded_file, "name", "document.pdf")

    prompt = _build_analysis_prompt(
        source_label="PDF",
        source_payload="The PDF is attached as a file input.",
    )

    gemini_file = client.files.upload(
        file=pdf_buffer,
        config={"mime_type": "application/pdf"},
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[gemini_file, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": ANALYSIS_SCHEMA,
                "temperature": 0.2,
            },
        )
    finally:
        try:
            client.files.delete(name=gemini_file.name)
        except Exception:
            pass

    payload = json.loads(response.text)
    return _normalize_analysis(payload)


def calculate_risk_score(risks: list) -> dict:
    """
    Compute a 0-100 risk score from detected risks.
    """
    if not risks:
        return {"score": 0, "category": "Low Risk", "color": "green"}

    severity_weights = {"high": 15, "medium": 7, "low": 3}
    raw_score = sum(severity_weights.get(r["severity"], 0) for r in risks)
    score = min(int((raw_score / 120) * 100), 100)

    if score <= 30:
        category = "Low Risk"
        color = "green"
    elif score <= 70:
        category = "Medium Risk"
        color = "orange"
    else:
        category = "High Risk"
        color = "red"

    return {"score": score, "category": category, "color": color}


def _normalize_analysis(payload: dict) -> dict:
    """
    Normalize Gemini output into the shapes expected by the Streamlit app.
    """
    clauses = []
    for item in payload.get("clauses", []):
        name = (item.get("name") or "").strip()
        excerpt = (item.get("excerpt") or "").strip()
        plain_language = (item.get("plain_language") or "").strip()
        if not name or not plain_language:
            continue
        clauses.append(
            {
                "name": name,
                "excerpt": excerpt,
                "plain_language": plain_language,
            }
        )

    risks = []
    for risk in payload.get("risks", []):
        severity = (risk.get("severity") or "").lower()
        if severity not in {"high", "medium", "low"}:
            continue
        label = (risk.get("label") or "").strip()
        plain_language = (risk.get("plain_language") or "").strip()
        if not label or not plain_language:
            continue
        risks.append(
            {
                "label": label,
                "severity": severity,
                "found_keywords": _unique_strings(risk.get("found_keywords", []), 3),
                "plain_language": plain_language,
            }
        )

    entities = payload.get("entities", {})
    return {
        "summary": (payload.get("summary") or "").strip(),
        "entities": {
            "persons": _unique_strings(entities.get("persons", []), 25),
            "organizations": _unique_strings(entities.get("organizations", []), 25),
        },
        "clauses": _unique_items_by_key(clauses, "name", 16),
        "risks": _unique_items_by_key(risks, "label", 16),
    }


def _unique_strings(values: list, limit: int) -> list:
    seen = set()
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
    seen = set()
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


def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Add GEMINI_API_KEY to your .env file."
        )
    return genai.Client(api_key=api_key)


def _build_analysis_prompt(source_label: str, source_payload: str) -> str:
    return f"""
You are a professional legal document analysis engine. Review the provided {source_label} and return only valid JSON that matches the schema.

Primary objective:
- Produce a precise, formal, business-appropriate analysis suitable for a professional document review workflow.

Tasks:
1. Write a concise summary in 5 to 8 sentences.
   - The summary must be easy for a non-lawyer to understand.
   - Use simple everyday English.
   - Avoid legal jargon where possible and explain the practical meaning of the document.
2. Extract person names and organization names explicitly mentioned.
3. Identify the important clauses and meaningful terms in the agreement.
   - Do not limit yourself to predefined clause names.
   - Include standard clauses and also unusual, one-sided, or shady terms.
   - For each clause, provide:
     - a short clause name
     - a short excerpt or tightly grounded paraphrase
     - a plain_language explanation in simple non-lawyer English that explains what it means in practice
4. Identify notable legal or commercial risks, including suspicious, one-sided, unfair, overbroad, hidden, unusual, or potentially harmful terms even if they do not match a standard clause category.
   - Look for "shady" patterns such as broad indemnity, unilateral rights, hidden fees, vague obligations, auto-renewal traps, harsh termination rights, non-compete overreach, waiver of rights, unlimited liability, missing obligations from the other side, or terms that strongly favor one party.
   - Also look for lock-in language such as forced renewal, same-broker dependency, exclusivity, trailing commissions, rollover terms, one-sided price changes, penalties for switching, and hidden payment obligations.
   - Each risk must include a short professional label, a severity of high, medium, or low, supporting keywords or phrases, and a plain_language explanation in simple non-lawyer English.
   - Treat a risky clause, an unusual omission, or a strongly imbalanced term as a valid risk if the document supports it.

Reference clause areas to consider, but do not treat this as a fixed list:
{json.dumps(REFERENCE_CLAUSE_TOPICS)}

Hard constraints:
- Use only the provided material.
- Do not guess, infer hidden facts, or fill in missing details.
- If a point is uncertain or weakly supported, omit it rather than speculate.
- Maintain a neutral, professional, legal-review tone.
- Do not use hype, conversational filler, marketing language, or dramatic phrasing.
- Do not give legal advice, recommendations, or action items.
- Do not mention being an AI model.
- Do not mention the prompt, schema, or JSON rules in the output.
- Keep the summary factual, restrained, and non-promotional.
- Keep entity lists unique and limited to items actually present.
- Keep clause names short and readable.
- Keep clause excerpts under 280 characters and closely tied to the source.
- Keep clause plain_language explanations under 220 characters and easy for a non-lawyer to understand.
- Keep risk labels short, specific, and professional.
- Keep found_keywords short, source-grounded, and specific.
- Keep risk plain_language explanations under 220 characters and easy for a non-lawyer to understand.
- If no meaningful risks are present, return an empty risks array.
- If the document quality is poor or a section is unreadable, rely only on readable content.

Quality bar:
- Prefer precision over coverage.
- Prefer omission over overstatement.
- Favor established legal terminology for clause names when supported by the source, but keep explanations simple.
- Only include clauses that are materially relevant to understanding the agreement.
- Treat severity conservatively: use high only for materially adverse, strongly one-sided, punitive, or clearly harmful terms.
- For risk detection, do not limit yourself to the named clauses list.
- Surface materially imbalanced terms even when they appear subtle, indirect, or scattered across multiple sections.

{source_payload}
""".strip()

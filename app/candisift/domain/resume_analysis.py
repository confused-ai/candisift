"""
Enhanced resume analysis: action verbs, quantification, sections, formatting, length.
All deterministic, zero LLM. Inspired by ats-screener's per-category scoring.
"""
from __future__ import annotations

import re

_STRONG: frozenset[str] = frozenset([
    "accelerated", "achieved", "architected", "automated", "awarded",
    "boosted", "built",
    "championed", "created", "cut",
    "delivered", "deployed", "designed", "developed", "directed", "drove",
    "earned", "eliminated", "engineered", "established", "exceeded",
    "founded", "generated", "grew", "guided",
    "implemented", "improved", "increased", "initiated", "integrated",
    "launched", "led", "leveraged",
    "mentored", "migrated", "modernized",
    "negotiated",
    "optimized", "orchestrated", "overhauled",
    "pioneered", "published",
    "reduced", "refactored", "reorganized",
    "saved", "scaled", "shipped", "simplified", "spearheaded", "streamlined",
    "transformed",
    # present-tense / base forms so "Lead a team" / "Drive adoption" score too
    # (the set is past-tense heavy; _verb_forms normalizes plural/-ing, not tense).
    "lead", "build", "drive", "deliver", "design", "develop", "improve", "increase",
    "launch", "optimize", "reduce", "scale", "ship", "create", "mentor", "migrate",
    "automate", "architect", "achieve", "streamline", "spearhead",
])

_WEAK: frozenset[str] = frozenset([
    "assisted", "did", "got", "had", "handled", "helped",
    "involved", "made", "managed", "participated", "performed",
    "responsible", "used", "utilized", "was", "worked",
])

_BULLET = re.compile(r'^[ \t]*[•\-\*\–\—▪▸►◆○●]', re.M)
_NUMBER = re.compile(
    r'\b\d+(?:\.\d+)?\s*%'
    r'|\b\d+\s*[xX]\b'
    r'|\$\d[\d,]*(?:\.\d+)?k?'
    r'|\b\d+\s*(?:million|billion|trillion|thousand|[KkMmBbGg])\b'
    r'|\b\d+\s*(?:users?|customers?|engineers?|members?|clients?|repos?|services?|teams?)\b',
    # NOTE: deliberately NO bare `\b\d{2,}\b` — that counted a plain year ("2019") or
    # date range as "quantified", scoring a date-only bullet 100% and inflating quality.
    # Quantification requires a unit/symbol/scale-word/entity, which the alts above cover.
    re.I,
)

_SECTIONS: dict[str, str] = {
    "summary":        r"(professional\s+)?summary|objective|profile|about\s+me",
    "experience":     r"(work\s+)?(experience|history|employment)",
    "education":      r"education|academic|degree|university|college|school|bachelor|master|phd",
    "skills":         r"(technical\s+)?(skills?|competencies|expertise|technologies|stack|proficiencies)",
    "projects":       r"projects?|portfolio|open[- ]?source|side\s+projects?",
    "certifications": r"certif(ications?|ied)|licen[sc]es?|credentials?",
    "awards":         r"awards?|achievements?|accomplishments?|honors?|publications?|recognition",
}


def _bullet_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if _BULLET.match(ln)]


def _first_word(line: str) -> str:
    clean = re.sub(r'^[•\-\*\–\—▪▸►◆○●\s]+', '', line)
    parts = clean.split()
    return parts[0].lower() if parts else ''


def _verb_forms(w: str) -> set[str]:
    """Cheap morphological variants so present/past/3rd-person of the same verb match
    the same set: 'leads'/'led'→'lead', 'optimizes'→'optimize'. Avoids the old blanket
    rstrip('s') which mangled non-verbs ('process'→'proces') and missed tense.
    ponytail: a hand-rolled stemmer, not a lemmatizer — swap in nltk WordNet if the
    verb lists ever grow past hand-maintenance."""
    forms = {w}
    if w.endswith("s") and len(w) > 3:
        forms.add(w[:-1])                    # leads -> lead, runs -> run
    if w.endswith("ed") and len(w) > 4:
        forms.add(w[:-2]); forms.add(w[:-1]) # delivered -> deliver/delivere
    if w.endswith("ing") and len(w) > 5:
        forms.add(w[:-3]); forms.add(w[:-3] + "e")  # building -> build, scaling -> scale
    return forms


def action_verb_analysis(text: str) -> dict:
    bullets = _bullet_lines(text)
    total = len(bullets)
    strong = weak = 0
    weak_examples: list[str] = []

    for ln in bullets:
        forms = _verb_forms(_first_word(ln))
        if forms & _STRONG:
            strong += 1
        elif forms & _WEAK:
            weak += 1
            if len(weak_examples) < 4:
                weak_examples.append(ln[:90])

    score = int(round(strong / total * 100)) if total else 0
    return {
        "total_bullets": total,
        "strong": strong,
        "weak": weak,
        "score": score,
        "weak_examples": weak_examples,
        "ok": score >= 60 or total == 0,
    }


def quantification_analysis(text: str) -> dict:
    bullets = _bullet_lines(text)
    total = len(bullets)
    quantified = sum(1 for b in bullets if _NUMBER.search(b))
    unquantified = [b[:90] for b in bullets if not _NUMBER.search(b)][:4]
    rate = int(round(quantified / total * 100)) if total else 0
    return {
        "total_bullets": total,
        "quantified": quantified,
        "rate": rate,
        "unquantified_examples": unquantified,
        "ok": rate >= 40 or total == 0,
    }


def section_analysis(text: str) -> dict:
    found: list[str] = []
    for section, pattern in _SECTIONS.items():
        if re.search(rf'(?m)^[ \t]*({pattern})[ \t]*[:\-]?[ \t]*$', text, re.I):
            found.append(section)
    core = ["experience", "education", "skills"]
    missing = [s for s in core if s not in found]
    return {
        "found": found,
        "missing_core": missing,
        "ok": len(missing) == 0,
    }


def length_analysis(text: str) -> dict:
    words = len(text.split())
    pages = round(words / 450, 1)
    if words < 250:
        status, ok = "too short", False
    elif words > 1000:
        status, ok = "too long", False
    else:
        status, ok = "good", True
    return {"words": words, "pages_est": pages, "ok": ok, "status": status}


def formatting_issues(text: str) -> list[str]:
    issues: list[str] = []
    if text.count('|') > 4:
        issues.append("Table-like pipe characters — ATS may misparse columns")
    if re.search(r'\t{2,}', text):
        issues.append("Multiple consecutive tabs — likely multi-column layout")
    if re.search(r'<[a-zA-Z]+\s*/?>', text):
        issues.append("HTML/XML tags found — strip before submitting")
    if re.search(r'(?m)^.{150,}$', text):
        issues.append("Very long lines — likely table or multi-column layout")
    return issues


def full_analysis(resume_text: str) -> dict:
    return {
        "action_verbs":   action_verb_analysis(resume_text),
        "quantification": quantification_analysis(resume_text),
        "sections":       section_analysis(resume_text),
        "length":         length_analysis(resume_text),
        "formatting":     formatting_issues(resume_text),
    }

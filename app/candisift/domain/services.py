"""Pure domain rules — deterministic, no I/O, no framework. Unit-testable in isolation.

These are business invariants, not infrastructure: who is hard-filtered out, what
counts as the same candidate, what identity data must never reach an evaluator.
The ranking *algorithm* is an adapter (swappable for embeddings); the ranking
*inputs/rules* that are business policy live here.
"""
from __future__ import annotations

import hashlib
import re

from .models import CandidateProfile, JDSpec


# ---- skills ontology (stub) ----------------------------------------------

# ponytail: hand-map, not a real taxonomy. The proper upgrade is skillNER (spaCy)
# for extraction + ESCO/O*NET as the canonical skill graph — drop them in behind
# this function (canon stays the call site). Canonicalization is what lets
# "ReactJS" match "react" so the ranker and keyword checks aren't fooled by spelling.
_CANON = {
    "reactjs": "react", "react.js": "react", "reactnative": "react native",
    "js": "javascript", "ts": "typescript", "typescript": "typescript",
    "node": "nodejs", "node.js": "nodejs", "nodejs": "nodejs",
    "k8s": "kubernetes", "kube": "kubernetes",
    "postgres": "postgresql", "psql": "postgresql",
    "py": "python", "golang": "go", "gcp": "google cloud", "aws": "aws",
    "tf": "terraform", "k8": "kubernetes", "ml": "machine learning",
    "nlp": "natural language processing", "ci/cd": "cicd", "ci": "cicd",
    "rest": "rest api", "restful": "rest api", "gql": "graphql",
}


def canon(skill: str) -> str:
    s = skill.strip().lower()
    return _CANON.get(s, s)


# ---- identity / dedup -----------------------------------------------------

def dedup_key(profile: CandidateProfile) -> str:
    """Same person re-applies with tweaked resumes. Key on normalized
    name+email+phone so we don't re-screen or double-count.

    Returns "" when the profile has NO identity at all (name+email+phone all blank
    — e.g. extraction returned nothing). An empty identity must NOT produce a shared
    hash: that would collapse every identity-less applicant onto the first one and
    silently discard the rest. The caller treats "" as non-dedupable (exact re-uploads
    are still caught upstream by content_sha256)."""
    parts = [
        re.sub(r"\s+", "", v.lower())
        for v in (profile.name, profile.email, re.sub(r"\D", "", profile.phone))
    ]
    if not any(parts):
        return ""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


# ---- compliance: strip protected-class proxies before screening ----------

# free-text identity that survives the scalar blanking: an email, a personal URL
# (LinkedIn/GitHub/portfolio), or the candidate's own name quoted inside a summary
# or an evidence snippet. We do NOT strip bare digit runs/years here — those are the
# metrics ("reduced cost 1.5M", "12 years") the evaluators must see; blanking them
# would gut the signal. grad_year (the age proxy) is already zeroed as a scalar.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# URLs and personal profile links — including bare domains with a path
# ("linkedin.com/in/jane") that carry no http/www prefix. A domain followed by a
# "/path" is a link, not prose, so redacting it is safe; "node.js" (no slash) is left
# alone. Known social/profile hosts are caught even without a path.
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+"
    r"|\b(?:linkedin|github|gitlab|bitbucket|twitter|x|behance|dribbble|medium)\.com\S*"
    r"|\b[\w-]+\.(?:com|io|dev|me|net|org|co|ai)/\S+",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"

# A phone number written differently from the stored scalar ("(555) 123-4567" vs the
# extracted "+15551234567") survives identifier matching, so catch the shape too.
# Deliberately NOT a bare digit-run: it must carry a "+" prefix or separators between
# three groups, so the metrics the evaluators need ("reduced cost 1500000", "2020-2023")
# are never redacted — same reason the module leaves bare digits/years alone above.
_PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    r"\+\d[\d\s().-]{6,18}\d"                    # +1 555 123 4567 (international)
    r"|\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"        # (555) 123-4567 — US 10-digit
    r")(?!\w)"
    # The US form ends in a 4-digit group on purpose: a date ("2020-01-15") ends in 2
    # and a dotted metric ("12.500.000") in 3, so neither matches — the module's rule
    # that digit metrics are signal, not PII, holds.
)


def _loc_parts(location: str) -> list[str]:
    """City/region tokens of a location string ("Hyderabad, Telangana, India" ->
    the three parts) so each can be redacted out of free text individually."""
    return [p.strip() for p in re.split(r"[,/|]", location or "") if p.strip()]


def _redact_free_text(text: str, identifiers: list[str]) -> str:
    """Redact the candidate's own name/phone/location (passed in `identifiers`) and any
    email/phone/personal-URL pattern from one free-text field, case-insensitively.

    Identifiers match on word boundaries: a bare substring pass would turn "India" into
    "[redacted]n Institute of Technology" and mangle the very signal the evaluator needs."""
    if not text:
        return text
    out = text
    # longest first so "john smith" is redacted before "john" leaves a dangling "smith"
    for ident in sorted(identifiers, key=len, reverse=True):
        if len(ident) >= 3:                      # avoid redacting 1-2 char noise
            out = re.sub(rf"(?<!\w){re.escape(ident)}(?!\w)", _REDACTED, out,
                         flags=re.IGNORECASE)
    out = _EMAIL_RE.sub(_REDACTED, out)
    out = _URL_RE.sub(_REDACTED, out)
    out = _PHONE_RE.sub(_REDACTED, out)
    return out


def _scrub_linkedin(digest: dict, idents: list[str]) -> dict:
    """Redact the candidate's identity from the resume-derived LinkedIn digest's
    free-text fields before it reaches an evaluator — same boundary as work_entries.
    Skills are canonical tech names, not identity, so they pass through unchanged."""
    if not digest:
        return digest
    positions = [
        {
            **pos,
            "title": _redact_free_text(pos.get("title", ""), idents),
            "company": _redact_free_text(pos.get("company", ""), idents),
            "highlights": [_redact_free_text(h, idents) for h in pos.get("highlights", [])],
        }
        for pos in digest.get("positions", [])
    ]
    return {**digest,
            "headline": _redact_free_text(digest.get("headline", ""), idents),
            "positions": positions}


def strip_pii(profile: CandidateProfile) -> CandidateProfile:
    """Blank name/email/phone/location/grad-year (age proxy) AND redact the same
    identity from every free-text field an evaluator sees — summary, titles,
    employment-gap notes, and each skill's verbatim evidence quote — so the screen
    is on skills and experience, not identity. Without the free-text pass a quoted
    "Jane Doe led the team" snippet leaks the name straight to the persona LLMs.
    Original is retained for the recruiter UI shown only after a human decision."""
    # the candidate's own identifiers to scrub out of free text (name parts too, so
    # "Jane" and "Doe" are caught individually inside prose). Location goes in — it is
    # blanked as a scalar because it proxies national origin, so "based in Lagos" in a
    # summary would hand the proxy back — but ONLY when it names a real place: a
    # work-mode value ("Remote") isn't identity and would erase "remote" everywhere.
    loc_idents = [profile.location, *_loc_parts(profile.location)] if is_place(profile.location) else []
    # A first name that is also a listed skill ("Ruby", "Jordan") must not be redacted,
    # or the strongest evidence quote ("10 years of Ruby on Rails") is gutted.
    skill_tokens = {s.name.strip().lower() for s in profile.skills if s.name.strip()}
    name_parts = [p for p in profile.name.split() if p.lower() not in skill_tokens]
    idents = [profile.name, profile.email, profile.phone, *name_parts, *loc_idents]
    idents = [i.strip() for i in idents if i and i.strip()]

    scrubbed_skills = [
        s.model_copy(update={"evidence": _redact_free_text(s.evidence, idents)})
        for s in profile.skills
    ]
    scrubbed_work = [
        w.model_copy(update={
            "highlights": [_redact_free_text(h, idents) for h in w.highlights],
        })
        for w in profile.work_entries
    ]
    # education dates are the age proxy that zeroing grad_year exists to remove —
    # leaving "2004 - 2008" on a degree hands the evaluator the graduation year the
    # scalar blanking just took away. The recruiter UI reads the unstripped profile,
    # so the dates are only hidden from the models.
    scrubbed_edu = [
        e.model_copy(update={
            "institution": _redact_free_text(e.institution, idents),
            "start_date": "", "end_date": "",
        })
        for e in profile.education
    ]
    # github.com/<user> is a full identity handle; blank the URL and drop the per-project
    # links (keep name/tech/counts — those are the signal) so the username never reaches
    # a persona LLM in the serialized profile.
    scrubbed_gh = [{k: v for k, v in proj.items() if k not in ("github_url", "live_url")}
                   for proj in profile.github_projects]
    return profile.model_copy(update={
        "name": "", "email": "", "phone": "", "location": "", "grad_year": 0,
        "linkedin_url": "", "portfolio_url": "", "github_url": "",
        "github_projects": scrubbed_gh,
        "linkedin_profile": _scrub_linkedin(profile.linkedin_profile, idents),
        "summary": _redact_free_text(profile.summary, idents),
        "titles": [_redact_free_text(t, idents) for t in profile.titles],
        "employment_gaps": [_redact_free_text(g, idents) for g in profile.employment_gaps],
        "skills": scrubbed_skills,
        "work_entries": scrubbed_work,
        "education": scrubbed_edu,
    })


# ---- deterministic experience validator ----------------------------------

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_date(date_str: str) -> tuple[int, int] | None:
    """Parse a date string into (year, month). Returns None if unparseable.

    Resumes write dates every way imaginable; every format missed here silently drops
    a work span, which undercounts total_years and can hard-filter a good candidate out
    on a years gate. So this stays deliberately permissive."""
    import datetime
    if not date_str:
        return None
    s = date_str.strip().lower().rstrip(".")
    # normalize separators/punctuation: "Jan. 2020", "Jan, 2020", "Jan-2020" -> "jan 2020"
    s = re.sub(r"[.,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None                      # whitespace/punctuation-only is unparseable, NOT "present"
    if s in ("present", "current", "now", "till date", "to date", "ongoing"):
        today = datetime.date.today()
        return (today.year, today.month)

    # try YYYY-MM or YYYY/MM
    m = re.match(r"(\d{4})[-/](\d{1,2})(?!\d)", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return (int(m.group(1)), int(m.group(2)))
    # try MM/YYYY or MM-YYYY ("03/2021") — the year is 4 digits, so no ambiguity
    m = re.match(r"(\d{1,2})[-/](\d{4})(?!\d)", s)
    if m and 1 <= int(m.group(1)) <= 12:
        return (int(m.group(2)), int(m.group(1)))
    # try "Jan 2020" / "January 2020" / "Jan-2020"
    m = re.match(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*[\s-]+(\d{4})", s)
    if m:
        return (int(m.group(2)), _MONTH_MAP[m.group(1)[:3]])
    # try "2020 Jan"
    m = re.match(r"(\d{4})[\s-]+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s)
    if m:
        return (int(m.group(1)), _MONTH_MAP[m.group(2)[:3]])
    # last resort: any plausible 4-digit year in the string ("Summer 2020", "FY2019").
    # Month is unknown -> January, which can only *under*state a span, never inflate it.
    m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", s)
    if m:
        return (int(m.group(1)), 1)
    return None


def validate_experience(profile: CandidateProfile) -> CandidateProfile:
    """Deterministic post-extraction validator. Cross-checks total_years against
    parseable work_entry dates and detects concurrent full-time roles. When EVERY entry's
    dates parse, the computed total replaces the LLM-provided total_years (which may
    hallucinate); when only some parse the computed figure is a floor, so the larger of
    the two wins. Also auto-detects concurrent_fulltime and gaps."""
    entries = profile.work_entries
    if not entries:
        return profile

    # A blank end_date is ambiguous — it means "current role" OR "extraction dropped it".
    # Assuming "present" for every entry inflates an old job to today and can even
    # manufacture a false concurrent-employment fraud signal, so treat blank end as
    # present ONLY for the single latest-starting role; for any earlier entry with no
    # end, drop the span (it counts as unparsed and lets `partial` keep the stated total).
    starts = [_parse_date(e.start_date) for e in entries]
    latest_idx = max((i for i, st in enumerate(starts) if st), key=lambda i: starts[i], default=None)

    spans: list[tuple[int, int]] = []  # (start_month_idx, end_month_idx)
    for i, e in enumerate(entries):
        start = starts[i]
        if not start:
            continue
        if e.end_date.strip():
            end = _parse_date(e.end_date)
        elif i == latest_idx:
            end = _parse_date("present")     # only the current role runs to today
        else:
            end = None                       # blank end on an older role -> unknown, skip
        if start and end:
            s = start[0] * 12 + start[1]
            n = end[0] * 12 + end[1]
            if n >= s:
                spans.append((s, n))

    if not spans:
        return profile
    # Some entry's dates didn't parse, so the computed total covers only part of the
    # career and is a FLOOR, not the truth. Replacing total_years with it would
    # undercount a mixed-format resume straight into a years-gate rejection, so take
    # the more generous of the two instead of trusting a partial computation.
    partial = len(spans) < len(entries)

    # sort and merge overlapping spans (union)
    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for s, n in spans[1:]:
        if s <= merged[-1][1]:
            # overlap → extend
            merged[-1] = (merged[-1][0], max(merged[-1][1], n))
        else:
            merged.append((s, n))

    # total non-overlapping months → years (inclusive of both start and end months)
    total_months = sum(n - s + 1 for s, n in merged)
    computed_years = round(total_months / 12, 1)

    # detect concurrent full-time: any two original spans where one starts before the other ends
    concurrent = False
    sorted_spans = sorted(spans)
    for i in range(len(sorted_spans) - 1):
        if sorted_spans[i][1] > sorted_spans[i + 1][0] + 1:  # +1 for 1-month tolerance
            concurrent = True
            break

    # detect employment gaps (>6 months between merged spans)
    gaps = profile.employment_gaps[:]
    for i in range(len(merged) - 1):
        gap_months = merged[i + 1][0] - merged[i][1]
        if gap_months > 6:
            gap_years = round(gap_months / 12, 1)
            gaps.append(f"~{gap_years}y gap detected between employment entries")

    updates: dict = {
        "total_years": max(profile.total_years, computed_years) if partial else computed_years
    }
    if concurrent and not profile.concurrent_fulltime:
        updates["concurrent_fulltime"] = True
    if gaps and not profile.employment_gaps:
        updates["employment_gaps"] = gaps

    return profile.model_copy(update=updates)


# ---- hard filters (deterministic, ~free) ---------------------------------

# City equivalences so "Bengaluru" on a resume matches "Bangalore" in a JD.
# Longest alias first so "delhi ncr" canonicalizes before bare "ncr".
# ponytail: hand-map like _CANON above; upgrade path is a geocoding lib if
# international coverage ever matters.
_LOC_ALIASES = [
    ("bengaluru", "bangalore"), ("blr", "bangalore"),
    ("gurugram", "gurgaon"),
    ("bombay", "mumbai"),
    ("madras", "chennai"),
    ("calcutta", "kolkata"),
    ("delhi ncr", "delhi"), ("new delhi", "delhi"), ("ncr", "delhi"),
    ("secunderabad", "hyderabad"), ("hyd", "hyderabad"),
    ("trivandrum", "thiruvananthapuram"),
    ("vizag", "visakhapatnam"),
    ("nyc", "new york"), ("sf", "san francisco"),
]

_RELOC_RE = re.compile(
    r"\b(?:willing|open|ready)\s+to\s+relocat|\bcan\s+relocate\b|\bopen\s+for\s+relocation\b",
    re.IGNORECASE,
)

# A phrase like "US citizen" or "willing to relocate" is negated when a negator sits in
# the SAME clause before it. Scoping to the clause (not a fixed char window) is what
# stops "Not an H1B holder. US citizen" from vetoing the citizenship, and "Cannot travel
# but willing to relocate" from vetoing the relocation, while still catching "not a US
# citizen". Clause ends at . ; ! ? or a contrastive conjunction (but/however/though).
_NEGATORS = r"(?:\b(?:no|not|non|never|without|neither|nor|un(?:willing|able|authorized)|cannot|can ?not)\b|n't)"
# A period splits only after 2+ letters, so "U.S." / "Ph.D" / "3.5" stay intact while
# "holder. US citizen" still splits; ; ! ? and contrastive conjunctions always split.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[a-z]{2})\.|[;!?]|\b(?:but|however|though|although|yet)\b", re.I)


def _negated_before(text: str, pos: int) -> bool:
    """True if a negator governs the phrase ending at `pos` — i.e. appears in the same
    clause, before it. Only the text back to the nearest clause boundary is considered,
    so a negation in an unrelated clause can't reach across."""
    boundaries = [m.end() for m in _CLAUSE_SPLIT_RE.finditer(text, 0, pos)]
    clause = text[boundaries[-1]:pos] if boundaries else text[:pos]
    return re.search(_NEGATORS, clause, re.I) is not None

# a work-mode preference in the location field means no city was stated -> unknown
_NON_PLACE_LOCATIONS = {"remote", "anywhere", "wfh", "work from home", "n/a", "unknown",
                        "hybrid", "onsite", "on-site", "flexible", "any", "none"}


def _canon_loc(text: str) -> str:
    t = text.lower()
    for alias, canonical in _LOC_ALIASES:
        t = re.sub(rf"\b{re.escape(alias)}\b", canonical, t)
    return t


def states_relocation_intent(text: str) -> bool:
    """True only for an AFFIRMED willingness to relocate."""
    for m in _RELOC_RE.finditer(text or ""):
        if not _negated_before(text, m.start()):
            return True
    return False


# work-mode / filler tokens, plus bare country-or-region qualifiers that ride along with
# "Remote" without naming a city. A value made only of these states no place.
_NON_PLACE_WORDS = {"remote", "anywhere", "wfh", "work", "from", "home", "n", "a", "na",
                    "unknown", "hybrid", "onsite", "on", "site", "flexible", "any", "none",
                    "us", "usa", "india", "uk", "eu", "emea", "apac", "global"}


def is_place(location: str) -> bool:
    """False when the location field states a work-mode preference rather than a city.
    Parentheticals are dropped first ("Remote (US)" -> "Remote"), since they qualify the
    mode rather than naming a place; what's left is a place only if some token is not a
    work-mode/qualifier word — so "Bangalore (remote ok)" is a place, "Remote (US)" isn't."""
    stripped = re.sub(r"\([^)]*\)", " ", location or "")
    tokens = [t for t in re.split(r"[^a-z]+", stripped.lower()) if t]
    return any(t not in _NON_PLACE_WORDS for t in tokens)


def location_matches(profile_location: str, jd_locations: list[str]) -> bool:
    """Word-boundary match after alias canonicalization: a bare substring made JD "Bath"
    match "Bathinda". Blank JD entries are skipped — "" is a substring of everything, so
    one would silently disable the gate.

    Still deliberately containment, not equality, so JD "Bangalore" matches a resume's
    "Greater Bangalore Area". The cost is that JD "York" would match "New York"; that
    over-accepts (a recruiter catches it at outreach) where equality would under-accept,
    and this gate leans toward the recoverable error."""
    cand = _canon_loc(profile_location)
    for loc in jd_locations:
        needle = _canon_loc(loc).strip()
        if needle and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", cand):
            return True
    return False


# ---- work authorization ---------------------------------------------------

# Canonical work-auth statuses and the phrasings that map onto them. A resume says
# "Authorized to work in the United States"; a JD says "US Citizen". Comparing those
# as raw substrings false-rejects on a LEGAL gate, so both sides canonicalize first.
# ponytail: hand-map. Anything it can't canonicalize routes to a human (see
# hard_filter), which is what keeps an incomplete table from silently rejecting.
_AUTH_ALIASES: list[tuple[str, str]] = [
    ("us citizen", "us_citizen"), ("united states citizen", "us_citizen"),
    ("american citizen", "us_citizen"), ("citizen of the united states", "us_citizen"),
    ("usc", "us_citizen"), ("naturalized citizen", "us_citizen"),
    ("lawful permanent resident", "green_card"), ("us permanent resident", "green_card"),
    ("permanent resident", "green_card"), ("green card holder", "green_card"),
    ("green card", "green_card"),
    ("h1b visa", "h1b"), ("h1b", "h1b"),
    ("l1 visa", "l1"), ("l1", "l1"),
    ("stem opt", "opt"), ("f1 opt", "opt"), ("opt", "opt"),
    ("cpt", "cpt"), ("ead", "ead"),
    ("tn visa", "tn"), ("tn permit", "tn"),
    ("requires sponsorship", "needs_sponsorship"),
    ("require sponsorship", "needs_sponsorship"),
    ("needs sponsorship", "needs_sponsorship"),
    ("need sponsorship", "needs_sponsorship"),
    ("will require sponsorship", "needs_sponsorship"),
    ("no sponsorship required", "us_authorized"),
    ("does not require sponsorship", "us_authorized"),
    ("authorized to work in the united states", "us_authorized"),
    ("authorized to work in the us", "us_authorized"),
    ("eligible to work in the united states", "us_authorized"),
    ("eligible to work in the us", "us_authorized"),
    ("us work authorization", "us_authorized"),
    ("us work authorized", "us_authorized"),
    ("authorized to work", "us_authorized"),
    ("indian citizen", "in_citizen"), ("oci", "oci"),
]

# A status on the left legally satisfies a requirement on the right. Citizens and green
# card holders have unrestricted authorization; a visa (H1B/L1/TN/OPT/CPT) and an EAD
# ARE authorization to work in the US, so they satisfy an "authorized to work" ask — but
# none of them imply citizenship or a green card. One-directional by design.
_AUTH_IMPLIES: dict[str, set[str]] = {
    "us_citizen": {"us_citizen", "us_authorized"},
    "green_card": {"green_card", "us_authorized"},
    "ead": {"ead", "us_authorized"},
    "h1b": {"h1b", "us_authorized"}, "l1": {"l1", "us_authorized"},
    "tn": {"tn", "us_authorized"}, "opt": {"opt", "us_authorized"},
    "cpt": {"cpt", "us_authorized"},
}


def _norm_auth(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\bu\.?\s?s\.?a?\.?\b", "us", t)     # "U.S." / "U. S." / "USA" -> "us"
    t = re.sub(r"[^\w\s]", " ", t)                   # "H-1B" -> "h 1b"
    t = re.sub(r"\b([a-z]) (\d)", r"\1\2", t)        # rejoin visa codes: "h 1b" -> "h1b"
    return re.sub(r"\s+", " ", t).strip()


def _auth_statuses(text: str) -> set[str]:
    """Canonical work-auth statuses AFFIRMED in `text`. A negated status ("not a US
    citizen") is not affirmed, scoped to its own clause so "Not an H1B holder. US
    citizen" still affirms citizenship. Empty set == nothing recognized, which the
    caller treats as "unknown", never as "does not qualify"."""
    found: set[str] = set()
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        t = _norm_auth(clause)
        # longest alias first so "us permanent resident" wins over "permanent resident"
        for alias, status in sorted(_AUTH_ALIASES, key=lambda p: len(p[0]), reverse=True):
            for m in re.finditer(rf"(?<!\w){re.escape(_norm_auth(alias))}(?!\w)", t):
                if not _negated_before(t, m.start()):
                    found.add(status)
                    break
    return found


def work_auth_satisfies(stated: str, required: list[str]) -> bool | None:
    """True = qualifies, False = a recognized but conflicting status, None = unknown
    (either side unparseable) -> the caller routes it to a human rather than rejecting."""
    req = set()
    for r in required:
        req |= _auth_statuses(r)
    if not req:
        return None                                  # can't canonicalize the requirement
    have = _auth_statuses(stated)
    if not have:
        return None                                  # can't canonicalize the candidate
    implied: set[str] = set()
    for h in have:
        implied |= _AUTH_IMPLIES.get(h, {h})
    return bool(req & implied)


# ---- certifications -------------------------------------------------------

# "CKA" and "Certified Kubernetes Administrator" are the same credential; exact string
# equality rejects the candidate who wrote the other one.
_CERT_ALIASES = {
    "cka": "certified kubernetes administrator",
    "ckad": "certified kubernetes application developer",
    "cks": "certified kubernetes security specialist",
    "aws saa": "aws certified solutions architect associate",
    "saa": "aws certified solutions architect associate",
    "aws sap": "aws certified solutions architect professional",
    "pmp": "project management professional",
    "cissp": "certified information systems security professional",
    "ccna": "cisco certified network associate",
    "rhce": "red hat certified engineer",
    "rhcsa": "red hat certified system administrator",
    "csm": "certified scrum master",
    "scrum master": "certified scrum master",
    "gcp ace": "google cloud associate cloud engineer",
    "az 900": "microsoft azure fundamentals",
    "ocjp": "oracle certified professional java programmer",
}

# words that carry no identity of their own. Tier words (professional/associate) are
# NOT here: they distinguish real credentials — dropping them made an AWS SAP match an
# AWS SAA. "certified" etc. are safe because both sides carry them equally.
_CERT_STOPWORDS = {"certified", "certification", "certificate", "certs", "cert",
                   "exam", "the", "of", "in", "and", "v2", "v3"}


def _cert_tokens(cert: str) -> frozenset[str]:
    # normalise: all punctuation to spaces (so "AZ-900" -> "az 900" and a tier in parens
    # "(Associate)" survives as a token), then drop only year-like tokens ("(2021)") —
    # NOT tier words, which distinguish real credentials.
    s = re.sub(r"[^a-z0-9]+", " ", cert.lower())
    s = re.sub(r"\b(?:19|20)\d{2}\b", " ", s)        # drop a calendar year, keep "900" etc.
    s = re.sub(r"\s+", " ", s).strip()
    toks = set(_CERT_ALIASES.get(s, s).split())      # whole-string acronym (e.g. "aws saa")
    for t in list(toks):                             # + any acronym as a lone token ("pmp 2021")
        if t in _CERT_ALIASES:
            toks |= set(_CERT_ALIASES[t].split())
    content = {t for t in toks if t} - _CERT_STOPWORDS
    return frozenset(content or toks)    # all-stopword cert -> fall back to raw tokens


def cert_satisfied(required: str, have: list[str]) -> bool:
    """A required cert is satisfied when every content word of it appears in one of the
    candidate's certs (after alias expansion) — so "AWS Certified Solutions Architect –
    Associate" is met by "AWS Solutions Architect (Associate)"."""
    req = _cert_tokens(required)
    if not req:
        return True
    return any(req <= _cert_tokens(h) for h in have)


# A years gate rejects only BELOW this fraction of the requirement. 4.9y against a
# "5+ years" ask is a conversation, not a knockout — and total_years is itself an
# estimate (see validate_experience), so a hard edge would reject on rounding noise.
_YEARS_TOLERANCE = 0.85


def hard_filter(profile: CandidateProfile, jd: JDSpec) -> tuple[bool, list[str], list[str]]:
    """Stage that removes ~50-70% at zero cost.
    Returns (passed, rejection reasons, advisory flags).

    The organizing rule: reject on CONFLICTING EVIDENCE, flag on MISSING EVIDENCE.
    Extraction is imperfect and a resume is not a form — "we could not parse it" is
    not the same fact as "they do not qualify", and only the second one may reject.
    So a stated-but-unrecognized work auth, an unknown location, or unestablished
    years pass with an advisory flag for the recruiter, while a recognized conflicting
    status (needs sponsorship vs citizens-only) still rejects for free.

    The one exception is work auth left entirely blank, which stays fail-closed: an
    unstated legal authorization cannot be assumed."""
    reasons: list[str] = []
    flags: list[str] = []

    # Blank entries state no requirement, and extraction produces them. Dropped up front
    # for every list: a lone "" in required_work_auth/required_certs rejected EVERY
    # candidate, and in locations it matched every candidate — a spec artefact must not
    # decide anyone's screen.
    jd_auth = _nonblank(jd.required_work_auth)
    jd_locations = _nonblank(jd.locations)
    jd_certs = _nonblank(jd.required_certs)
    jd_knockouts = _nonblank(jd.knockouts)
    have_certs = _nonblank(profile.certifications)

    if jd_auth:
        if not profile.work_authorization.strip():
            reasons.append("work authorization not stated; required " + str(jd_auth))
        else:
            ok = work_auth_satisfies(profile.work_authorization, jd_auth)
            if ok is None:
                flags.append(f"work auth '{profile.work_authorization}' could not be matched "
                             f"against required {jd_auth} — verify with candidate")
            elif not ok:
                reasons.append(f"work auth '{profile.work_authorization}' not in {jd_auth}")

    # Only the JD's remote_ok opens the location gate. profile.remote_ok is an
    # LLM-inferred boolean with no evidence requirement behind it, and letting a model's
    # guess ("remote_ok=False") arm a deterministic reject on a remote-friendly role is
    # exactly the kind of unverifiable auto-rejection this stage must not do.
    if jd_locations and not jd.remote_ok:
        if not is_place(profile.location):
            # blank, or a work-mode preference ("Remote") standing in for a city
            flags.append("location not stated; on-site role needs one of "
                         f"{jd_locations} — verify at outreach")
        elif not location_matches(profile.location, jd_locations):
            if states_relocation_intent(f"{profile.location} {profile.summary}"):
                flags.append(f"location '{profile.location}' not in {jd_locations} "
                             "but candidate states willingness to relocate")
            else:
                reasons.append(f"location '{profile.location}' not in {jd_locations}")

    if jd.min_years > 0:
        if profile.total_years <= 0:
            # Nothing measurable. Work history present with a 0 total means extraction
            # failed (validate_experience leaves total_years alone when no date parses),
            # and no history at all means nothing was extracted either — a resume that
            # genuinely has zero experience still lists something. Either way this is
            # missing evidence, not evidence of a shortfall.
            flags.append(f"years of experience not established from the resume; "
                         f"role asks {jd.min_years}y — verify at outreach")
        elif profile.total_years < jd.min_years * _YEARS_TOLERANCE:
            reasons.append(f"{profile.total_years}y experience < required {jd.min_years}y")
        elif profile.total_years < jd.min_years:
            flags.append(f"{profile.total_years}y experience just under the {jd.min_years}y "
                         "asked for — near miss, recruiter call")

    missing = [c for c in jd_certs if not cert_satisfied(c, have_certs)]
    if missing:
        if not have_certs:
            # no certs extracted at all -> we cannot tell "has none" from "we missed
            # the section", and a cert is usually verifiable in one question.
            flags.append(f"no certifications extracted; role requires {missing} — verify")
        else:
            reasons.append(f"missing required certs: {missing}")

    # JD knockouts: explicit disqualifying keywords. The deterministic reading is
    # "auto-reject if this term is present in the candidate's skills/titles/summary"
    # (e.g. a contract-to-hire role knocking out "consultant only"). Matched here for
    # free rather than spending an LLM call to rediscover a stated hard rule.
    if jd_knockouts:
        haystack = " ".join(
            [profile.summary, *profile.titles, *(s.name for s in profile.skills)]
        ).lower()
        hit = [k for k in jd_knockouts if _knockout_hit(k, haystack)]
        if hit:
            reasons.append(f"matched JD knockout(s): {hit}")

    return (not reasons), reasons, flags


def _nonblank(values: list[str]) -> list[str]:
    return [v for v in values if v and v.strip()]


def _knockout_hit(knockout: str, haystack: str) -> bool:
    """Word-boundary match, never a bare substring, and never a negated mention.
    `"java" in haystack` fires on "JavaScript", "intern" on "international"; "no
    experience with java" affirms the opposite of a "java" knockout — each a silent
    auto-reject on a coincidence of spelling or polarity."""
    k = knockout.strip().lower()
    for m in re.finditer(rf"(?<!\w){re.escape(k)}(?!\w)", haystack):
        if not _negated_before(haystack, m.start()):
            return True
    return False

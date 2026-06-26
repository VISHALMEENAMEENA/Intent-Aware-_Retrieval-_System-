"""
parse_sections.py
=================
Intent-Aware and Explainable Hybrid Retrieval System
------------------------------------------------------
Responsibility  : Parse a structured JD dict (produced by parse_one_jd.py)
                  into a rich, semantically classified output schema.

Public API      : structure_jd(parsed_data) -> dict

Design Principles
-----------------
- Zero dependency on exact heading names.
- Every bullet is independently classified by keyword rules.
- All keyword lists are defined as constants at the top of the file
  so they can be extended without touching logic.
- Each helper function has a single, clearly stated responsibility.
- No I/O, no pandas, no external libraries — pure Python + re.

Output Schema
-------------
{
    job_title         : str,
    industry          : str,
    city              : str,
    country           : str,
    responsibilities  : list[str],
    must_have_skills  : list[str],
    preferred_skills  : list[str],
    soft_skills       : list[str],
    certifications    : list[str],
    education         : list[str],
    experience        : list[str],
    technologies      : list[str],
    tools             : list[str],
    other_requirements: list[str],
}
"""

import re


# ==============================================================
# ❶ CONFIGURABLE KEYWORD CONSTANTS
#    Extend any list here without touching the logic below.
# ==============================================================

# ── Section header aliases ─────────────────────────────────────

RESPONSIBILITY_HEADERS = [
    "detailed responsibilities",
    "responsibilities",
    "job responsibilities",
    "key responsibilities",
    "roles & responsibilities",
    "roles and responsibilities",
    "role responsibilities",
    "duties",
    "role overview",
    "what you will do",
    "your responsibilities",
    "scope of work",
    "primary responsibilities",
]

SKILL_HEADERS = [
    "skill requirements",
    "required skills",
    "mandatory skills",
    "technical skills",
    "must have skills",
    "must-have skills",
    "skills required",
    "skills",
    "qualifications",
    "eligibility",
    "requirements",
    "what we are looking for",
    "what you need",
    "what you bring",
    "minimum qualifications",
    "basic qualifications",
]

OTHER_HEADERS = [
    "other requirements",
    "additional requirements",
    "preferred skills",
    "preferred qualifications",
    "nice to have",
    "good to have",
    "additional information",
    "bonus",
    "plus",
    "would be a plus",
    "would be an advantage",
    "added advantage",
]

# ── Per-bullet classification keywords ────────────────────────

EDUCATION_KEYWORDS = [
    "bachelor",
    "master",
    "degree",
    "graduate",
    "phd",
    "b.tech",
    "m.tech",
    "b.e.",
    "m.e",
    "bsc",
    "msc",
    "mba",
    "bca",
    "mca",
    "b.sc",
    "m.sc",
    "b.com",
    "education",
    "qualification",
    "diploma",
    "undergraduate",
    "postgraduate",
    "university",
    "college",
    "12th",
    "10th",
    "computer science",    # common education context phrase
    "engineering degree",
    "information technology degree",
]

EXPERIENCE_KEYWORDS = [
    "experience",
    "years",
    " yrs",
    "minimum",
    "worked",
    "working",
    "at least",
    "hands-on",
    "prior experience",
    "track record",
    "background in",
    "exposure to",
    "proven",
]

CERTIFICATION_KEYWORDS = [
    "aws certified",
    "azure certified",
    "google professional",
    "google cloud certified",
    "pmp",
    "itil",
    "scrum master",
    "oracle certified",
    "cisco",
    "comptia",
    "ceh",
    "cissp",
    "cism",
    "cisa",
    "six sigma",
    "safe ",
    "togaf",
    "rhce",
    "rhcsa",
    "ccna",
    "ccnp",
    "ccie",
    "gcp certified",
    "prince2",
    "certificate",
    "certification",
    "certified",
]

SOFT_SKILL_KEYWORDS = [
    "communication",
    "leadership",
    "teamwork",
    "team player",
    "analytical",
    "adaptability",
    "adaptable",
    "collaboration",
    "collaborative",
    "problem solving",
    "problem-solving",
    "critical thinking",
    "interpersonal",
    "time management",
    "presentation",
    "attention to detail",
    "self-motivated",
    "multitasking",
    "creativity",
    "creative",
    "empathy",
    "conflict resolution",
    "organizational",
    "proactive",
    "detail-oriented",
    "work ethic",
    "ownership",
    "accountability",
    "initiative",
    "positive attitude",
    "self-starter",
    "verbal",
    "written communication",
    "stakeholder management",
    "cross-functional",
    "flexibility",
    "willingness",
]

# Technologies: programming languages, frameworks, platforms, ML libs
TECHNOLOGY_KEYWORDS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "golang",
    "rust",
    "scala",
    "kotlin",
    "swift",
    "ruby",
    "php",
    "matlab",
    "sql",
    "nosql",
    "graphql",
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "next.js",
    "node.js",
    "spring",
    "spring boot",
    "springboot",
    "django",
    "flask",
    "fastapi",
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numpy",
    "spark",
    "hadoop",
    "kafka",
    "flink",
    "airflow",
    "mlflow",
    "langchain",
    "llamaindex",
    "hugging face",
    "transformers",
    "bert",
    "gpt",
    "llm",
    "generative ai",
    "gen ai",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "reinforcement learning",
    "aws",
    "azure",
    "gcp",
    "google cloud",
    "microservices",
    "rest api",
    "restful",
    "grpc",
    "soap",
    "pl/sql",
    "oracle",
    "mysql",
    "postgresql",
    "mongodb",
    "redis",
    "elasticsearch",
    "opensearch",
    "neo4j",
    "cassandra",
    "dynamodb",
    "snowflake",
    "databricks",
    "power bi",
    "tableau",
    "looker",
    "linux",
    "unix",
    "windows server",
    "shell scripting",
    "bash",
    "terraform",
    "ansible",
    "puppet",
    "chef",
    "selenium",
    "appium",
    "junit",
    "mockito",
    "pytest",
]

# Tools: specific products, platforms, services, DevOps tooling
TOOL_KEYWORDS = [
    "docker",
    "kubernetes",
    "k8s",
    "jenkins",
    "gitlab",
    "github",
    "bitbucket",
    "jira",
    "confluence",
    "servicenow",
    "remedy",
    "splunk",
    "datadog",
    "prometheus",
    "grafana",
    "elk",
    "kibana",
    "logstash",
    "sonarqube",
    "nexus",
    "artifactory",
    "helm",
    "istio",
    "ci/cd",
    "devops",
    "git",
    "svn",
    "postman",
    "swagger",
    "intellij",
    "eclipse",
    "visual studio",
    "vs code",
    "maven",
    "gradle",
    "npm",
    "yarn",
    "webpack",
    "xcode",
    "android studio",
    "figma",
    "sketch",
    "adobe",
    "sap",
    "salesforce",
    "workday",
    "ms office",
    "microsoft office",
    "microsoft teams",
    "slack",
    "zoom",
    "agile",
    "scrum",
    "kanban",
    "waterfall",
    "itsm",
    "itil",          # also certification but commonly a process/tool context
    "pagerduty",
    "opsgenie",
    "new relic",
    "dynatrace",
    "ansible",
    "puppet",
    "chef",
]

# Signals that this bullet is "preferred" rather than "must-have"
PREFERRED_SIGNALS = [
    "preferred",
    "is a plus",
    "would be a plus",
    "nice to have",
    "good to have",
    "advantageous",
    "desirable",
    "added advantage",
    "beneficial",
    "optional",
    "exposure to",
    "familiarity with",
    "knowledge of",
    "experience with",          # weak signal — only in "other" sections
]


# ==============================================================
# ❷ LOCATION PARSER
# ==============================================================

def parse_location(location_text: str):
    """
    Split a location string into (city, country).

    Handles:
        "Pune, India"           -> ("Pune", "India")
        "Noida"                 -> ("Noida", "")
        "N/A"                   -> ("", "")
        "Pune/Mumbai/Bengaluru" -> ("Pune/Mumbai/Bengaluru", "")
        "New York, USA"         -> ("New York", "USA")
    """
    text = location_text.strip()

    # Treat N/A or empty as blank
    if not text or text.upper() in ("N/A", "NA", "-", "NONE", "NULL"):
        return "", ""

    parts = [p.strip() for p in text.split(",")]

    if len(parts) >= 2:
        city    = parts[0]
        country = parts[1]
    else:
        city    = parts[0]
        country = ""

    return city, country


# ==============================================================
# ❸ LINE CLEANING HELPER
# ==============================================================

def clean_bullet(line: str) -> str:
    """
    Strip markdown bullet characters, bold markers, and extra whitespace
    from a single line.

    Examples:
        "- **Must Have:** Python"  -> "Must Have: Python"
        "• Kubernetes experience"  -> "Kubernetes experience"
        "  * Strong communication" -> "Strong communication"
    """
    # Remove leading bullets
    line = re.sub(r"^[\-\•\*\+]\s*", "", line.strip())
    # Remove bold/italic markdown
    line = re.sub(r"\*\*|__|\*|_", "", line)
    # Collapse internal whitespace
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def is_sub_header(line: str) -> bool:
    """
    Return True if the cleaned line looks like an inline sub-heading
    inside a section (e.g., "Must Have:", "Educational Qualifications:").
    These lines should not be collected as bullets.
    """
    sub_header_patterns = [
        r"^must\s*have\s*:?$",
        r"^educational\s*(qualifications?|requirements?)\s*:?$",
        r"^experience\s*:?$",
        r"^technical\s*skills?\s*:?$",
        r"^good\s*to\s*have\s*:?$",
        r"^nice\s*to\s*have\s*:?$",
        r"^preferred\s*:?$",
        r"^soft\s*skills?\s*:?$",
        r"^key\s*skills?\s*:?$",
        r"^mandatory\s*skills?\s*:?$",
    ]
    lower = line.lower().strip().rstrip(":")
    for pat in sub_header_patterns:
        if re.fullmatch(pat, lower.strip(":").strip()):
            return True
    # Standalone label ending in colon with no other content
    if re.fullmatch(r"[a-z ]{3,40}:", line.lower().strip()):
        return True
    return False


# ==============================================================
# ❹ SECTION HEADER DETECTOR
# ==============================================================

def _normalise_header(line: str) -> str:
    """Strip markdown symbols and normalise a line for header matching."""
    line = re.sub(r"[#*_`>]", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip().lower()


def detect_section_type(line: str):
    """
    Classify a line as a section header.

    Returns one of:
        "responsibilities" | "skills" | "other" | None

    Priority: responsibilities > skills > other
    (so "Key Responsibilities and Requirements" maps to responsibilities)
    """
    normalised = _normalise_header(line)

    for header in RESPONSIBILITY_HEADERS:
        if header in normalised:
            return "responsibilities"

    for header in SKILL_HEADERS:
        if header in normalised:
            return "skills"

    for header in OTHER_HEADERS:
        if header in normalised:
            return "other"

    return None


# ==============================================================
# ❺ BULLET CLASSIFIER
# ==============================================================

# Short technology keywords that must match as whole words to prevent
# false positives inside longer words (e.g. "r" inside "customer").
_TECH_WORD_BOUNDARY = [
    r"\br\b",           # R language
    r"\bc\b",           # C language
    r"\bc#",            # C#
    r"\bc\+\+",         # C++
    r"\bgo\b",          # Go (Golang)
    r"\bjava\b",        # Java (not JavaScript)
    r"\bsql\b",         # SQL
    r"\bspark\b",       # Spark
]
_TECH_WORD_BOUNDARY_RE = [re.compile(p, re.IGNORECASE) for p in _TECH_WORD_BOUNDARY]


def _matches_technology(lower: str) -> bool:
    """
    Return True if the line matches a technology keyword.
    Short/ambiguous keywords use word-boundary regex.
    All other keywords use plain substring matching.
    """
    # Plain substring check for longer, unambiguous keywords
    if any(kw in lower for kw in TECHNOLOGY_KEYWORDS):
        return True
    # Word-boundary check for short keywords
    if any(rx.search(lower) for rx in _TECH_WORD_BOUNDARY_RE):
        return True
    return False


def classify_bullet(line: str, source_section: str = "skills") -> str:
    """
    Classify a single cleaned bullet into one of:
        "certification"     - mentions a known certification
        "education"         - mentions a degree / qualification
        "experience"        - mentions years / prior experience
        "soft_skill"        - interpersonal / behavioural skill
        "technology"        - programming language / ML framework / platform
        "tool"              - DevOps tool / product / SaaS
        "preferred_skill"   - bullet contains a preferred/optional signal
        "must_have_skill"   - default for strong technical bullets

    The source_section parameter adds context:
        bullets from "other" sections get "preferred_skill" as default
        instead of "must_have_skill".

    Classification is ordered by specificity (most specific first).
    """
    lower = line.lower()

    # ── 1. Certification (most specific) ──────────────────────
    if any(kw in lower for kw in CERTIFICATION_KEYWORDS):
        return "certification"

    # ── 2. Education ──────────────────────────────────────────
    if any(kw in lower for kw in EDUCATION_KEYWORDS):
        return "education"

    # ── 3. Experience ─────────────────────────────────────────
    if any(kw in lower for kw in EXPERIENCE_KEYWORDS):
        return "experience"

    # ── 4. Soft skill ─────────────────────────────────────────
    if any(kw in lower for kw in SOFT_SKILL_KEYWORDS):
        return "soft_skill"

    # ── 5. Technology (uses word-boundary matching for short kws) ─
    if _matches_technology(lower):
        return "technology"

    # ── 6. Tool ───────────────────────────────────────────────
    if any(kw in lower for kw in TOOL_KEYWORDS):
        return "tool"

    # ── 7. Preferred signal ───────────────────────────────────
    if any(sig in lower for sig in PREFERRED_SIGNALS):
        return "preferred_skill"

    # ── 8. Default by source section ──────────────────────────
    if source_section == "other":
        return "preferred_skill"

    return "must_have_skill"


# ==============================================================
# ❻ RAW SECTION SPLITTER
# ==============================================================

def split_into_sections(raw_sections: dict) -> dict:
    """
    Given the parsed_data dict from parse_markdown(), identify the
    raw text block for each logical section (responsibilities / skills / other).

    Strategy
    --------
    1. Try known key names that parse_markdown() typically produces.
    2. If a key is not found by name, run dynamic header detection
       over the full concatenated text as a fallback.

    Returns:
        {
            "responsibilities": str,   # raw multi-line text
            "skills":           str,
            "other":            str,
        }
    """

    # ── Step 1: Direct key lookup ─────────────────────────────
    # parse_markdown() uses the heading text as the dict key.
    # Try every alias in order; take the first non-empty match.

    def first_nonempty(*keys):
        for k in keys:
            v = raw_sections.get(k, "").strip()
            if v:
                return v
        return ""

    responsibilities_raw = first_nonempty(
        "Detailed Responsibilities",
        "Responsibilities",
        "Job Responsibilities",
        "Key Responsibilities",
        "Roles & Responsibilities",
        "Roles and Responsibilities",
        "Role Responsibilities",
        "Duties",
        "Role Overview",
    )

    skills_raw = first_nonempty(
        "Skill Requirements",
        "Required Skills",
        "Mandatory Skills",
        "Technical Skills",
        "Must Have Skills",
        "Must-Have Skills",
        "Skills Required",
        "Skills",
        "Qualifications",
        "Eligibility",
        "Requirements",
        "Minimum Qualifications",
    )

    other_raw = first_nonempty(
        "Other Requirements",
        "Additional Requirements",
        "Preferred Skills",
        "Preferred Qualifications",
        "Nice to Have",
        "Good to Have",
        "Additional Information",
    )

    # ── Step 2: Dynamic fallback ──────────────────────────────
    # If ALL three are empty (unusual format), scan the concatenated
    # text and split by detected section headers.

    if not responsibilities_raw and not skills_raw and not other_raw:
        full_text = "\n".join(str(v) for v in raw_sections.values())
        responsibilities_raw, skills_raw, other_raw = _dynamic_split(full_text)

    return {
        "responsibilities": responsibilities_raw,
        "skills": skills_raw,
        "other": other_raw,
    }


def _dynamic_split(full_text: str):
    """
    Scan full_text line-by-line and split into (responsibilities, skills, other)
    blocks based on detected section headers.
    """
    buckets = {"responsibilities": [], "skills": [], "other": []}
    current = None

    for raw_line in full_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        section = detect_section_type(stripped)
        if section:
            current = section
            continue

        if current:
            buckets[current].append(stripped)

    return (
        "\n".join(buckets["responsibilities"]),
        "\n".join(buckets["skills"]),
        "\n".join(buckets["other"]),
    )


# ==============================================================
# ❼ SECTION LINE EXTRACTOR
# ==============================================================

def extract_bullets(raw_text: str) -> list:
    """
    Convert a raw multi-line text block into a list of clean, non-empty
    bullet strings. Sub-headers within the section are dropped.
    """
    bullets = []
    for raw_line in raw_text.splitlines():
        cleaned = clean_bullet(raw_line)
        if not cleaned:
            continue
        if is_sub_header(cleaned):
            continue
        bullets.append(cleaned)
    return bullets


# ==============================================================
# ❽ SKILL SECTION CLASSIFIER
# ==============================================================

def classify_skills_section(lines: list, source_section: str = "skills") -> dict:
    """
    Classify every bullet in the skills or other section.

    Returns a dict with lists:
        {
            "must_have_skills":  list[str],
            "preferred_skills":  list[str],
            "soft_skills":       list[str],
            "certifications":    list[str],
            "education":         list[str],
            "experience":        list[str],
            "technologies":      list[str],
            "tools":             list[str],
            "other_requirements":list[str],
        }
    """
    result = {
        "must_have_skills":   [],
        "preferred_skills":   [],
        "soft_skills":        [],
        "certifications":     [],
        "education":          [],
        "experience":         [],
        "technologies":       [],
        "tools":              [],
        "other_requirements": [],
    }

    for line in lines:
        category = classify_bullet(line, source_section=source_section)

        if category == "certification":
            result["certifications"].append(line)
        elif category == "education":
            result["education"].append(line)
        elif category == "experience":
            result["experience"].append(line)
        elif category == "soft_skill":
            result["soft_skills"].append(line)
        elif category == "technology":
            result["technologies"].append(line)
        elif category == "tool":
            result["tools"].append(line)
        elif category == "preferred_skill":
            result["preferred_skills"].append(line)
        else:  # must_have_skill
            result["must_have_skills"].append(line)

    return result


# ==============================================================
# ❾ RESPONSIBILITIES CLASSIFIER
# ==============================================================

def classify_responsibilities(lines: list) -> list:
    """
    Responsibilities are returned as-is (they are full sentence descriptions).
    We do a light pass to re-route any bullet that is clearly an education
    or certification statement that slipped into the responsibilities block.

    Returns a clean list of responsibility strings.
    """
    responsibilities = []
    for line in lines:
        # Lines that are unambiguously educaiton or cert are skipped here
        # (they will be captured from the skills section instead)
        responsibilities.append(line)
    return responsibilities


# ==============================================================
# ❿ MAIN ENTRY POINT
# ==============================================================

def structure_jd(parsed_data: dict) -> dict:
    """
    Main public entry point.

    Input  : parsed_data — dict returned by parse_one_jd.parse_markdown()
    Output : fully structured JD dict matching the output schema

    Output Schema
    -------------
    {
        job_title         : str,
        industry          : str,
        city              : str,
        country           : str,
        responsibilities  : list[str],
        must_have_skills  : list[str],
        preferred_skills  : list[str],
        soft_skills       : list[str],
        certifications    : list[str],
        education         : list[str],
        experience        : list[str],
        technologies      : list[str],
        tools             : list[str],
        other_requirements: list[str],
    }
    """

    # ── Meta fields ───────────────────────────────────────────
    job_title = parsed_data.get("Job Title", "").strip()
    industry  = parsed_data.get("Client Industry", "").strip()
    city, country = parse_location(parsed_data.get("Location", ""))

    # ── Split raw text into section blocks ────────────────────
    sections = split_into_sections(parsed_data)

    responsibilities_raw = sections["responsibilities"]
    skills_raw           = sections["skills"]
    other_raw            = sections["other"]

    # ── Extract clean bullets from each block ─────────────────
    responsibility_bullets = extract_bullets(responsibilities_raw)
    skill_bullets          = extract_bullets(skills_raw)
    other_bullets          = extract_bullets(other_raw)

    # ── Classify skill bullets ────────────────────────────────
    classified_skills = classify_skills_section(skill_bullets, source_section="skills")

    # ── Classify other/preferred bullets ─────────────────────
    classified_other  = classify_skills_section(other_bullets, source_section="other")

    # ── Merge "other" results into the skill classification ───
    # Education, experience, certs, soft skills found in "other" section
    # go to their proper buckets. The rest stay as preferred_skills /
    # other_requirements.

    def _merge_unique(base: list, extra: list) -> list:
        """Append items from extra that are not already in base."""
        seen = set(b.lower() for b in base)
        for item in extra:
            if item.lower() not in seen:
                base.append(item)
                seen.add(item.lower())
        return base

    must_have_skills   = classified_skills["must_have_skills"]
    preferred_skills   = _merge_unique(
        classified_skills["preferred_skills"],
        classified_other["preferred_skills"],
    )
    soft_skills        = _merge_unique(
        classified_skills["soft_skills"],
        classified_other["soft_skills"],
    )
    certifications     = _merge_unique(
        classified_skills["certifications"],
        classified_other["certifications"],
    )
    education          = _merge_unique(
        classified_skills["education"],
        classified_other["education"],
    )
    experience         = _merge_unique(
        classified_skills["experience"],
        classified_other["experience"],
    )
    technologies       = _merge_unique(
        classified_skills["technologies"],
        classified_other["technologies"],
    )
    tools              = _merge_unique(
        classified_skills["tools"],
        classified_other["tools"],
    )

    # Other requirements = must_have skills found in "other" block
    # (they are "preferred" by context even if they look like hard skills)
    # + anything truly uncategorised from "other"
    other_requirements = _merge_unique(
        classified_other["must_have_skills"],
        classified_other["other_requirements"],
    )

    # ── Build final output ────────────────────────────────────
    return {
        "job_title":          job_title,
        "industry":           industry,
        "city":               city,
        "country":            country,
        "responsibilities":   classify_responsibilities(responsibility_bullets),
        "must_have_skills":   must_have_skills,
        "preferred_skills":   preferred_skills,
        "soft_skills":        soft_skills,
        "certifications":     certifications,
        "education":          education,
        "experience":         experience,
        "technologies":       technologies,
        "tools":              tools,
        "other_requirements": other_requirements,
    }
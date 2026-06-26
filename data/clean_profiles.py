import os
import re
import json
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(CLEAN_DIR, exist_ok=True)

PROFILE_PATH = os.path.join(RAW_DIR, "profiles.csv")
OUTPUT_PATH = os.path.join(CLEAN_DIR, "profiles_cleaned.csv")


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def clean_text(text):
    """Remove extra spaces/newlines."""
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_skill(skill):
    """Normalize skill names."""
    return clean_text(skill).lower()


def parse_skill_column(skill_string, category):
    """
    Input:
        Python (Expert), SQL (Competent)

    Output:
        skills
        metadata
    """

    if pd.isna(skill_string):
        return [], {}

    skills = []
    metadata = {}

    for item in str(skill_string).split(","):

        item = item.strip()

        if item == "":
            continue

        match = re.match(r"(.+?)\s*\((.+?)\)", item)

        if match:

            skill = normalize_skill(match.group(1))
            level = normalize_skill(match.group(2))

        else:

            skill = normalize_skill(item)
            level = "unknown"

        if skill not in skills:
            skills.append(skill)

        metadata[skill] = {
            "level": level,
            "category": category
        }

    return skills, metadata


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(PROFILE_PATH)

# Remove name column
if "name" in df.columns:
    df.drop(columns=["name"], inplace=True)


all_skills_column = []
skill_metadata_column = []
profile_text_column = []


# --------------------------------------------------
# Process Every Profile
# --------------------------------------------------

for idx, row in df.iterrows():

    core_skills, core_meta = parse_skill_column(
        row["core_skills"],
        "core"
    )

    secondary_skills, secondary_meta = parse_skill_column(
        row["secondary_skills"],
        "secondary"
    )

    soft_skills, soft_meta = parse_skill_column(
        row["soft_skills"],
        "soft"
    )

    # -----------------------------------------
    # Preserve natural order:
    # Core -> Secondary -> Soft
    # -----------------------------------------

    all_skills = []

    for skill in core_skills + secondary_skills + soft_skills:

        if skill not in all_skills:
            all_skills.append(skill)

    # -----------------------------------------

    skill_metadata = {}

    skill_metadata.update(core_meta)
    skill_metadata.update(secondary_meta)
    skill_metadata.update(soft_meta)

    # -----------------------------------------
    # Clean Columns
    # -----------------------------------------

    df.at[idx, "core_skills"] = ", ".join(core_skills)

    df.at[idx, "secondary_skills"] = ", ".join(secondary_skills)

    df.at[idx, "soft_skills"] = ", ".join(soft_skills)

    df.at[idx, "potential_roles"] = clean_text(
        row["potential_roles"]
    ).lower()

    df.at[idx, "skill_summary"] = clean_text(
        row["skill_summary"]
    )

    # Experience
    try:
        experience = int(float(row["years_of_experience"]))
    except:
        experience = 0

    df.at[idx, "years_of_experience"] = experience

    # -----------------------------------------

    all_skills_column.append(", ".join(all_skills))

    skill_metadata_column.append(
        json.dumps(skill_metadata)
    )

    # -----------------------------------------
    # Build Profile Text
    # -----------------------------------------

    core_text = "\n".join([
        f"- {skill} ({core_meta[skill]['level']})"
        for skill in core_skills
    ])

    secondary_text = "\n".join([
        f"- {skill} ({secondary_meta[skill]['level']})"
        for skill in secondary_skills
    ])

    soft_text = "\n".join([
        f"- {skill} ({soft_meta[skill]['level']})"
        for skill in soft_skills
    ])

    profile_text = f"""
Candidate Profile

Experience:
{experience} years

Core Skills:
{core_text}

Secondary Skills:
{secondary_text}

Soft Skills:
{soft_text}

Potential Roles:
{df.at[idx,'potential_roles']}

Professional Summary:
{df.at[idx,'skill_summary']}
"""

    profile_text_column.append(
        clean_text(profile_text)
    )


# --------------------------------------------------
# New Columns
# --------------------------------------------------

df["all_skills"] = all_skills_column

df["skill_metadata"] = skill_metadata_column

df["profile_text"] = profile_text_column


# --------------------------------------------------
# Save
# --------------------------------------------------
# Convert experience column to integer
df["years_of_experience"] = (
    pd.to_numeric(df["years_of_experience"], errors="coerce")
      .fillna(0)
      .astype(int)
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 70)
print("Profiles cleaned successfully.")
print(f"Saved to : {OUTPUT_PATH}")
print("=" * 70)

print("\nColumns Created")
print("---------------------------")
print("✓ all_skills")
print("✓ skill_metadata")
print("✓ profile_text")
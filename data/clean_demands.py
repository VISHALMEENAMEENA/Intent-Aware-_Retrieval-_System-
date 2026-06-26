import os
import json
import re
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

os.makedirs(CLEAN_DIR, exist_ok=True)

INPUT_FILE = os.path.join(RAW_DIR, "demands_data.csv")
OUTPUT_FILE = os.path.join(CLEAN_DIR, "demands_cleaned.csv")


# ==========================================================
# Utility Functions
# ==========================================================

def clean_text(text):
    """Remove extra spaces and normalize text."""

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def split_skills(skill_string):
    """
    Convert

    SQL,Python,AWS

    into

    ['sql','python','aws']
    """

    if pd.isna(skill_string):
        return []

    skills = []

    for skill in str(skill_string).split(","):

        skill = clean_text(skill)

        if skill == "":
            continue

        if skill not in skills:
            skills.append(skill)

    return skills


# ==========================================================
# Load Dataset
# ==========================================================

def load_data():

    df = pd.read_csv(INPUT_FILE)

    return df


# ==========================================================
# Clean Job Title
# ==========================================================

def clean_job_title(df):

    # designation is the real title

    df["job_title"] = (
        df["designation"]
        .fillna("")
        .apply(clean_text)
    )

    # remove old columns

    df.drop(columns=["designation", "state"], inplace=True)

    return df


# ==========================================================
# Clean Skills
# ==========================================================

def clean_skills(df):

    all_skills = []
    metadata = []

    primary_column = []
    secondary_column = []

    for _, row in df.iterrows():

        primary = split_skills(row["primary_skills"])
        secondary = split_skills(row["secondary_skills"])

        primary_column.append(", ".join(primary))
        secondary_column.append(", ".join(secondary))

        # Preserve order
        merged = []

        for skill in primary + secondary:

            if skill not in merged:
                merged.append(skill)

        all_skills.append(", ".join(merged))

        skill_meta = {}

        for skill in primary:

            skill_meta[skill] = {
                "category": "primary"
            }

        for skill in secondary:

            if skill not in skill_meta:

                skill_meta[skill] = {
                    "category": "secondary"
                }

        metadata.append(json.dumps(skill_meta))

    df["primary_skills"] = primary_column
    df["secondary_skills"] = secondary_column

    df["all_skills"] = all_skills
    df["skill_metadata"] = metadata

    return df


# ==========================================================
# Clean Location
# ==========================================================

def clean_location(df):

    city = df["city"].fillna("").apply(clean_text)

    country = df["country"].fillna("").apply(clean_text)

    df["city"] = city
    df["country"] = country

    locations = []

    for c, co in zip(city, country):

        if c and co:

            locations.append(f"{c}, {co}")

        elif co:

            locations.append(co)

        else:

            locations.append(c)

    df["location"] = locations

    return df


# ==========================================================
# Experience
# ==========================================================

def clean_experience(df):

    df["experience_lower"] = (
        pd.to_numeric(df["experience_lower"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df["experience_upper"] = (
        pd.to_numeric(df["experience_upper"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return df


# ==========================================================
# Job Text
# ==========================================================

def create_job_text(df):

    job_text = []

    for _, row in df.iterrows():

        text = f"""
Job Posting

Job Title:
{row['job_title']}

Location:
{row['location']}

Primary Skills:
{row['primary_skills']}

Secondary Skills:
{row['secondary_skills']}

Experience:
{row['experience_lower']} - {row['experience_upper']} years
"""

        job_text.append(clean_text(text))

    df["job_text"] = job_text

    return df


# ==========================================================
# Save
# ==========================================================

def save_data(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Cleaning Demand Dataset")
    print("=" * 60)

    df = load_data()

    df = clean_job_title(df)

    df = clean_skills(df)

    df = clean_location(df)

    df = clean_experience(df)

    df = create_job_text(df)

    save_data(df)

    print("\nCleaning Completed Successfully")
    print(f"\nSaved To : {OUTPUT_FILE}")

    print("\nNew Columns Created")
    print("---------------------------")
    print("✓ all_skills")
    print("✓ skill_metadata")
    print("✓ location")
    print("✓ job_text")


if __name__ == "__main__":
    main()
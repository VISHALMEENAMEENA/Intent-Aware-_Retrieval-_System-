import os
import json
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROFILE_PATH = os.path.join(BASE_DIR, r"C:data\raw\profiles.csv" )
DEMAND_PATH = os.path.join(BASE_DIR, r"C:data\raw\demands_data.csv")
JD_PATH = os.path.join(BASE_DIR, r"C:data\raw\jd_dataset")


def line():
    print("=" * 80)


def inspect_csv(path, name):
    line()
    print(f"{name}")
    line()

    df = pd.read_csv(path)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nFirst 5 Rows")
    print(df.head())

    print("\nSample Row")
    print(df.iloc[0].to_dict())

    return df


def inspect_jd_folder(jd_root):

    line()
    print("JD DATASET")
    line()

    folders = sorted(
        [
            f
            for f in os.listdir(jd_root)
            if os.path.isdir(os.path.join(jd_root, f))
        ],
        key=lambda x: int(x)
    )

    print(f"Total JD folders : {len(folders)}")

    raw_count = 0
    enhanced_count = 0

    industries = {}

    for folder in folders:

        path = os.path.join(jd_root, folder)

        raw = os.path.join(path, "raw_jd.txt")
        enhanced = os.path.join(path, "enhanced_job_description.md")

        if os.path.exists(raw):
            raw_count += 1

        if os.path.exists(enhanced):
            enhanced_count += 1

        if os.path.exists(raw):

            try:

                with open(raw, "r", encoding="utf8") as f:
                    data = json.load(f)

                ind = data.get("industry", "Unknown")
                industries[ind] = industries.get(ind, 0) + 1

            except Exception:
                pass

    print("\nRaw JD Files :", raw_count)
    print("Enhanced JD Files :", enhanced_count)

    print("\nIndustry Distribution")

    for k, v in sorted(
        industries.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(f"{k:35} {v}")

    print("\nExample Folder")

    sample = folders[0]

    print(sample)

    raw = os.path.join(jd_root, sample, "raw_jd.txt")

    with open(raw, encoding="utf8") as f:
        obj = json.load(f)

    print("\nKeys")

    print(obj.keys())

    print("\nIndustry")

    print(obj["industry"])

    print("\nRaw JD Preview")

    print(obj["raw_jd"][:700])


def inspect_skill_distribution(df, columns):

    line()
    print("Skill Statistics")
    line()

    skills = {}

    for col in columns:

        if col not in df.columns:
            continue

        for item in df[col].fillna(""):

            for skill in str(item).split(","):

                skill = skill.strip()

                if skill == "":
                    continue

                skills[skill] = skills.get(skill, 0) + 1

    print("\nTop 30 Skills")

    top = sorted(
        skills.items(),
        key=lambda x: x[1],
        reverse=True
    )[:30]

    for s, c in top:
        print(f"{s:35} {c}")


def main():

    profile = inspect_csv(PROFILE_PATH, "PROFILES")

    demand = inspect_csv(DEMAND_PATH, "DEMANDS")

    inspect_jd_folder(JD_PATH)

    inspect_skill_distribution(
        profile,
        [
            "core_skills",
            "secondary_skills",
            "soft_skills"
        ]
    )

    inspect_skill_distribution(
        demand,
        [
            "primary_skills",
            "secondary_skills"
        ]
    )


if __name__ == "__main__":
    main()
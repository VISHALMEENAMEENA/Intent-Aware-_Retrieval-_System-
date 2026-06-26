import os
import json
import pandas as pd

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

FILES = {
    "Profiles": "profiles_cleaned.csv",
    "Demands": "demands_cleaned.csv",
    "JD": "jd_cleaned.csv"
}


# ==========================================================
# Utility
# ==========================================================

def is_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False


def avg_words(series):
    return round(series.fillna("").apply(lambda x: len(str(x).split())).mean(), 2)


# ==========================================================
# Check One Dataset
# ==========================================================

def inspect_dataset(name, filename):

    path = os.path.join(CLEAN_DIR, filename)

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    df = pd.read_csv(path)

    # ------------------------------------------------------

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(list(df.columns))

    # ------------------------------------------------------

    print("\nDuplicate Rows :", df.duplicated().sum())

    # ------------------------------------------------------

    id_col = None

    for col in df.columns:
        if col.endswith("id") or col == "id":
            id_col = col
            break

    if id_col:
        print(f"\nDuplicate {id_col} :", df[id_col].duplicated().sum())

    # ------------------------------------------------------

    print("\nMissing Values")

    print(df.isnull().sum())

    # ------------------------------------------------------

    print("\nEmpty String Count")

    for col in df.columns:

        if df[col].dtype == object:

            empty = (df[col].fillna("").str.strip() == "").sum()

            if empty:
                print(f"{col:25} {empty}")

    # ------------------------------------------------------

    print("\nJSON Validation")

    for col in df.columns:

        if "metadata" in col:

            valid = df[col].fillna("{}").apply(is_json).sum()

            print(f"{col:25} {valid}/{len(df)} valid")

    # ------------------------------------------------------

    print("\nAverage Text Length")

    for col in df.columns:

        if "text" in col or "summary" in col:

            print(f"{col:25} {avg_words(df[col])} words")

    # ------------------------------------------------------

    print("\nAverage Skill Count")

    for col in df.columns:

        if "skills" in col or "keywords" in col:

            avg = df[col].fillna("").apply(
                lambda x: len([i for i in str(x).split(",") if i.strip()])
            ).mean()

            print(f"{col:25} {round(avg,2)}")

    # ------------------------------------------------------

    print("\nSample Record")

    sample = df.iloc[0].to_dict()

    for k, v in sample.items():

        value = str(v)

        if len(value) > 120:
            value = value[:120] + "..."

        print(f"{k:25}: {value}")


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n")
    print("=" * 80)
    print("HYBRID RETRIEVAL DATA QUALITY REPORT")
    print("=" * 80)

    for name, file in FILES.items():

        inspect_dataset(name, file)

    print("\n")
    print("=" * 80)
    print("CHECK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
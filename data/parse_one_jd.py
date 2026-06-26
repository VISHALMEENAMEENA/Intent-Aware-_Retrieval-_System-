import os
import re

# ==========================================================
# Change this to test another folder later
# ==========================================================

JD_FOLDER = "1"

# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JD_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "jd_dataset",
    JD_FOLDER,
    "enhanced_job_description.md"
)


# ==========================================================
# Read Markdown
# ==========================================================

def read_markdown(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================================
# Parse Markdown Sections
# ==========================================================

def parse_markdown(markdown):

    data = {}

    current_section = None

    lines = markdown.splitlines()

    for line in lines:

        line = line.strip()

        if line == "":
            continue

        # Heading
        if line.startswith("## "):

            current_section = line.replace("## ", "").strip()

            data[current_section] = []

        else:

            if current_section is not None:

                data[current_section].append(line)

    # Convert list into text

    for key in data:

        data[key] = "\n".join(data[key]).strip()

    return data


# ==========================================================
# Pretty Print
# ==========================================================

def print_sections(data):

    print("=" * 70)

    print("Parsed Sections")

    print("=" * 70)

    for section, content in data.items():

        print()

        print(f"[{section}]")

        print("-" * 40)

        print(content[:500])

        print()


# ==========================================================
# Main
# ==========================================================

def main():

    markdown = read_markdown(JD_FILE)

    parsed = parse_markdown(markdown)

    print_sections(parsed)


if __name__ == "__main__":
    main()
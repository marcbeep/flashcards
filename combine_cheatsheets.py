import os
import glob


def combine_markdown_files():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cheatsheets_dir = os.path.join(script_dir, "ci", "cheatsheets")
    output_file = os.path.join(cheatsheets_dir, "combined.md")

    # Get all markdown files and sort them
    markdown_files = glob.glob(os.path.join(cheatsheets_dir, "*.md"))
    markdown_files.sort()  # This will sort them alphabetically

    # Combine the files
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in markdown_files:
            filename = os.path.basename(file_path)

            # Skip the output file if it exists
            if filename == "combined.md":
                continue

            # Write separator with filename
            outfile.write(f"# File: {filename}\n")

            # Read and write the content of each file
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())


if __name__ == "__main__":
    combine_markdown_files()
    print("Markdown files have been combined into 'combined.md'")

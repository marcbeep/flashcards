import os
from PyPDF2 import PdfReader, PdfWriter


def combine_pdfs(input_dir, output_file):
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    pdf_files.sort(key=lambda x: int("".join(filter(str.isdigit, x.split(".")[0]))))

    # Create a PDF writer object
    writer = PdfWriter()

    # Process each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(input_dir, pdf_file)
        reader = PdfReader(file_path)

        # Add each page to the writer
        for page in reader.pages:
            writer.add_page(page)

    # Write the combined PDF
    with open(output_file, "wb") as output:
        writer.write(output)


if __name__ == "__main__":
    input_directory = "dm/problemsets"
    output_file = "ps.pdf"
    combine_pdfs(input_directory, output_file)
    print(f"Combined PDF created successfully: {output_file}")

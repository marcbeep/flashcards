import os
from PyPDF2 import PdfReader, PdfWriter


def combine_pdfs(input_dir, output_file):
    # Get all PDF files in the directory
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    # Sort files by converting the number part to float
    def get_number(filename):
        # Remove .pdf and convert to float
        return float(filename[:-4])

    pdf_files.sort(key=get_number)

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
    input_directory = "dm/lectures"
    output_file = "combined.pdf"
    combine_pdfs(input_directory, output_file)
    print(f"Combined PDF created successfully: {output_file}")

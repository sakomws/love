import os
import json
from fpdf import FPDF

def create_pdf_from_json_file(json_file_path, output_pdf_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)

    # Add a title or header for the JSON content
    pdf.multi_cell(0, 10, f"Content of {os.path.basename(json_file_path)}:")

    # Convert the JSON object to a pretty-printed string
    json_str = json.dumps(data, indent=4)
    
    # Add each line of the JSON string to the PDF
    for line in json_str.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf_path)
    print(f"PDF created for {json_file_path} at {output_pdf_path}")

def main():
    directory = './json'  # Replace with your JSON files directory
    output_dir = './data'  # Directory to save PDF files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory, filename)
            output_pdf_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.pdf")
            create_pdf_from_json_file(json_file_path, output_pdf_path)

if __name__ == "__main__":
    main()
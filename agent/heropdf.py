from fpdf import FPDF
from datetime import datetime

def create_pdf(input_data):
    # Step 1: Check the type of input_data and parse if necessary
    if isinstance(input_data, str):
        try:
            data = json.loads(input_data)  # Parsing JSON string to a dictionary
        except json.JSONDecodeError as e:
            return f"Error parsing input: {str(e)}"
    elif isinstance(input_data, dict):
        data = input_data  # Input is already a dictionary, no parsing needed
    else:
        return "Input must be a JSON string or a dictionary"

    # Step 2: Extract the list of SOC-2 compliant services
    services = data.get('result', '')

    # Create a new FPDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)

    # Add a title
    pdf.cell(200, 10, 'People Matching', 0, 1, 'C')
    
    # Add each service to the PDF
    services_lines = services.split('\n')
    for line in services_lines:
        if line.strip():  # Ensure no empty lines are added
            pdf.cell(200, 10, line, 0, 1)

    # Save the PDF
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")  # Formats the date and time in a file-friendly manner
    file_name = f'/Users/sakom/github/hackathon05122024/safescale-run/ui/public/pdfs/evidence_doc_{started_at}.pdf'
    pdf.output(file_name)
    return f"Evidence created successfully, file saved as: {file_name}"

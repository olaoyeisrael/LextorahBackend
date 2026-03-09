from fpdf import FPDF
import textwrap

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Lextorah AI - Lesson Transcript', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def write_inline_markdown(pdf, text):
    parts = text.split('**')
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 1:
            # Odd index means it was inside ** tags
            pdf.set_font('Arial', 'B', 12)
        else:
            # Even index means normal text
            pdf.set_font('Arial', '', 12)
        
        pdf.write(7, part)
    pdf.ln(7)

def create_pdf_from_text(text: str, filename: str):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Sanitize text for FPDF encoding limitations
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2019', "'").replace('\u2018', "'").replace('\u2013', "-").replace('\u2014', "-")
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Handle Headers (### or ## or #)
        if line.startswith('#'):
            # Count hashes
            header_level = len(line) - len(line.lstrip('#'))
            clean_text = line.lstrip('#').strip()
            # Clean up inline bolding from headers if they exist e.g. ### **Header**
            clean_text = clean_text.replace('**', '')
            
            pdf.set_font('Arial', 'B', 14)
            pdf.multi_cell(0, 8, clean_text)
            pdf.ln(2)
            
        # Handle List Items
        elif line.startswith('- ') or line.startswith('* ') or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
            # Indent slightly and write inline markdown
            pdf.set_x(20)
            write_inline_markdown(pdf, line)
            
        # Handle Normal Text
        else:
            pdf.set_x(10)
            write_inline_markdown(pdf, line)
            
    return pdf.output(dest='S').encode('latin-1') # Return bytes

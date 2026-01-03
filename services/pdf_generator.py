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

def create_pdf_from_text(text: str, filename: str):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    # Sanitize text
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    for line in text.split('\n'):
        # pdf.write(5, line + '\n') # Multi_cell is better for wrapping
        pdf.multi_cell(0, 7, line)
        
    return pdf.output(dest='S').encode('latin-1') # Return bytes

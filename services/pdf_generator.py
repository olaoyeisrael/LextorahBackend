from fpdf import FPDF
import textwrap

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        title_val = getattr(self, 'title_text', 'Lextorah AI - Lesson Transcript')
        self.cell(0, 10, title_val, 0, 0, 'C')
        self.ln(15)

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
    if "STUDENT PROGRESS TRACKER" in text:
        pdf = PDF(orientation='L')
        pdf.title_text = "Lextorah - Student Progress Tracker"
        col_widths = [45, 35, 25, 25, 55, 35, 62] # Fits Landscape width nicely
    else:
        pdf = PDF()
        pdf.title_text = "Lextorah AI - Lesson Transcript"
        col_widths = [35, 30, 20, 20, 35, 25, 25]
        
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Sanitize text for FPDF encoding limitations
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2019', "'").replace('\u2018', "'").replace('\u2013', "-").replace('\u2014', "-")
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    in_table = False
    table_headers = []

    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if not in_table:
                pdf.ln(5)
            continue
            
        # Handle Table Row
        if line.startswith('|'):
            # Check if separator
            if '---' in line:
                continue
                
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if not cells:
                continue
                
            in_table = True
            
            # Print row
            if not table_headers:
                table_headers = cells
                pdf.set_font('Arial', 'B', 8)
                pdf.set_fill_color(230, 240, 230) # Soft green background for header matching brand
                for idx, cell in enumerate(cells):
                    w = col_widths[idx] if idx < len(col_widths) else 20
                    pdf.cell(w, 8, cell, border=1, align='C', fill=True)
                pdf.ln(8)
            else:
                pdf.set_font('Arial', '', 8)
                for idx, cell in enumerate(cells):
                    w = col_widths[idx] if idx < len(col_widths) else 20
                    pdf.cell(w, 8, cell, border=1, align='C')
                pdf.ln(8)
            continue
        else:
            if in_table:
                # Reset table state
                table_headers = []
                in_table = False
                pdf.ln(5)
                
        # Handle Headers (### or ## or #)
        if line.startswith('#'):
            header_level = len(line) - len(line.lstrip('#'))
            clean_text = line.lstrip('#').strip()
            clean_text = clean_text.replace('**', '')
            
            pdf.set_font('Arial', 'B', 14)
            pdf.multi_cell(0, 8, clean_text)
            pdf.ln(2)
            
        # Handle List Items
        elif line.startswith('- ') or line.startswith('* ') or (len(line) > 2 and line[0].isdigit() and line[1] == '.'):
            pdf.set_x(20)
            write_inline_markdown(pdf, line)
            
        # Handle Normal Text
        else:
            pdf.set_x(10)
            write_inline_markdown(pdf, line)
            
    return pdf.output(dest='S').encode('latin-1') # Return bytes

from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Dokumentasi LangChain Stack - ZeroTouch V3', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        # Use multi_cell for text wrapping
        self.multi_cell(0, 8, body)
        self.ln(4)

pdf = PDF()
pdf.add_page()

with open('ZeroTouch_V3_LangChain_Docs.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

current_body = []
for line in lines:
    line = line.strip()
    line = line.replace("🛠️", "").replace("📚", "").replace("⚙️", "").replace("—", "-")
    line = line.replace("**", "").replace("`", "").replace("*", "")
    line = line.replace("<br>", " ")
    
    if line.startswith("# "):
        continue
    elif line.startswith("## "):
        if current_body:
            pdf.chapter_body("\n".join(current_body))
            current_body = []
        pdf.chapter_title(line[3:])
    else:
        current_body.append(line)

if current_body:
    pdf.chapter_body("\n".join(current_body))

pdf.output('ZeroTouch_V3_LangChain_Docs.pdf')
print("Success")

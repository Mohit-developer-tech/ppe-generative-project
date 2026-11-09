"""
Generate realistic sample safety policy documents (DOCX, PDF, and TXT)
ready for uploading to the Policy Transformer Sandbox on the dashboard.
"""
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.abspath("."), "results", ".mplcache")

POLICY_TEXT = """INDUSTRIAL SITE SAFETY POLICY & COMPLIANCE STANDARD
Standard Operating Procedure: PPE-SOP-2026-V4
Applicable Regulatory Codes: OSHA 29 CFR 1926.100, OSHA 29 CFR 1926.651, MUTCD Part 6D.03, ANSI/ISEA 107-2020

1. PURPOSE AND SCOPE
This document establishes mandatory Personal Protective Equipment (PPE) standards across all industrial construction, excavation, and structural refurbishment zones. Compliance is mandatory for all direct employees, subcontractors, delivery drivers, and authorized engineering visitors.

2. HEAD PROTECTION REQUIREMENTS (OSHA 29 CFR 1926.100)
Employees working in areas where there is a possible danger of head injury from impact, falling or flying objects, or from electrical shock and burns, shall be protected by protective helmets.
All personnel entering designated construction perimeters must wear ANSI Z89.1 approved Type I or Type II hard hats at all times.
Hard hats must be worn with the bill facing forward unless approved by the site safety officer for welding operations.
Bump caps and soft headwear are strictly prohibited in structural steel erection and crane swing radiuses.

3. HIGH-VISIBILITY SAFETY APPAREL (MUTCD PART 6D & ANSI/ISEA 107)
All workers within the right-of-way of public highways, traffic corridors, or active earthmoving machinery must wear high-visibility safety apparel.
Apparel must meet ANSI/ISEA 107-2020 Class 2 or Class 3 performance guidelines for daytime and nighttime retroreflective visibility.
Safety vests must remain fully zipped or fastened during working hours to maintain 360-degree visibility.

4. DUAL-PROTECTION MANDATORY ZONES
Both hard hats and high-visibility reflective vests are mandatory within active excavation trenches, underground tunnels, and near mobile crane operations.
Workers performing night operations along transit pathways shall wear both approved hard hats and high-visibility apparel equipped with reflective bands.

5. SPECIALIZED ANCILLARY PPE
Eye and face protection meeting ANSI Z87.1 standards shall be provided and worn during grinding, cutting, chipping, or chemical spraying.
Hearing protection with a minimum Noise Reduction Rating (NRR) of 25 dB must be used when ambient noise levels exceed 85 decibels.
Protective steel-toe footwear conforming to ASTM F2413 must be worn by all personnel on site.

6. RECOGNIZED SITE EXEMPTIONS AND ADMINISTRATIVE PROVISIONS
High-visibility vests and hard hats are not required while personnel are inside fully enclosed, climate-controlled equipment cabs or administrative office trailers.
Protective headwear is exempt in designated break areas and parking lots situated outside active machinery perimeters.
Employees failing to comply with mandatory PPE standards will be subject to immediate site removal and administrative retraining.
"""

def create_documents():
    out_dir = "sample_policy_documents"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plain Text (.txt)
    txt_path = os.path.join(out_dir, "site_safety_policy_osha_compliance.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(POLICY_TEXT.strip())
    print(f"[✓] Created TXT policy at: {txt_path}")
    
    # 2. Microsoft Word (.docx)
    docx_path = os.path.join(out_dir, "site_safety_policy_osha_compliance.docx")
    doc = docx.Document()
    
    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("INDUSTRIAL SITE SAFETY POLICY & COMPLIANCE STANDARD")
    run_title.bold = True
    run_title.font.size = Pt(16)
    run_title.font.color.rgb = RGBColor(15, 23, 42)
    
    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Standard Operating Procedure: PPE-SOP-2026-V4 | OSHA 29 CFR 1926 & MUTCD")
    run_sub.font.size = Pt(10)
    run_sub.font.color.rgb = RGBColor(100, 116, 139)
    doc.add_paragraph() # Spacing
    
    # Parse sections
    for line in POLICY_TEXT.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
            h = doc.add_heading(line, level=2)
            h.runs[0].font.size = Pt(12)
            h.runs[0].font.color.rgb = RGBColor(30, 58, 138)
        elif line.startswith("INDUSTRIAL SITE") or line.startswith("Standard Operating"):
            continue
        else:
            p = doc.add_paragraph(line)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(4)
            p.runs[0].font.size = Pt(10)
            p.runs[0].font.color.rgb = RGBColor(30, 41, 59)
            
    doc.save(docx_path)
    print(f"[✓] Created DOCX policy at: {docx_path}")
    
    # 3. PDF Document (.pdf) via Matplotlib PdfPages
    pdf_path = os.path.join(out_dir, "site_safety_policy_osha_compliance.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1
        fig, ax = plt.subplots(figsize=(8.5, 11), dpi=150)
        ax.axis("off")
        
        # Header banner
        rect = plt.Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes, color="#1E3A8A")
        ax.add_patch(rect)
        ax.text(0.5, 0.965, "INDUSTRIAL SITE SAFETY POLICY & COMPLIANCE STANDARD",
                transform=ax.transAxes, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")
        ax.text(0.5, 0.935, "Standard Operating Procedure: PPE-SOP-2026-V4  |  OSHA 29 CFR 1926 & MUTCD",
                transform=ax.transAxes, ha="center", va="center", color="#BFDBFE",
                fontsize=8.5)
        
        # Body text layout
        y = 0.88
        lines = POLICY_TEXT.strip().split("\n")[2:]
        for line in lines:
            line = line.strip()
            if not line:
                y -= 0.015
                continue
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.")):
                y -= 0.012
                ax.text(0.06, y, line, transform=ax.transAxes, fontsize=10.5,
                        fontweight="bold", color="#1E3A8A")
                y -= 0.022
            else:
                # Wrap text roughly to 85 chars
                words = line.split(" ")
                cur_line = []
                for w in words:
                    cur_line.append(w)
                    if len(" ".join(cur_line)) > 88:
                        ax.text(0.08, y, " ".join(cur_line[:-1]), transform=ax.transAxes,
                                fontsize=8.5, color="#334155")
                        y -= 0.018
                        cur_line = [w]
                if cur_line:
                    ax.text(0.08, y, " ".join(cur_line), transform=ax.transAxes,
                            fontsize=8.5, color="#334155")
                    y -= 0.018
                    
            if y < 0.08:
                break
                
        # Footer
        ax.text(0.5, 0.03, "Page 1 of 1 — Official Site Safety Compliance Document",
                transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#94A3B8")
                
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()
        
    print(f"[✓] Created PDF policy at: {pdf_path}")

if __name__ == "__main__":
    create_documents()


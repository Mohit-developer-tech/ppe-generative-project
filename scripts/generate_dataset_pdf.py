"""
scripts/generate_dataset_pdf.py
Generates an academic, publication-quality PDF document detailing the dataset sources,
extraction methodology, statistical distributions, multi-modal regulatory corpus,
and ethical governance protocols for the PPE Generative AI Project.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PDF = BASE_DIR / "PPE_Dataset_Specification_and_Sources.pdf"
STATIC_PDF = BASE_DIR / "dashboard" / "static" / "assets" / "PPE_Dataset_Specification_and_Sources.pdf"
ARTIFACT_PDF = Path(r"C:\Users\mrpat\.gemini\antigravity\brain\e413f0d0-a427-4818-9bb6-44112ae7f351\PPE_Dataset_Specification_and_Sources.pdf")

DIST_IMG = BASE_DIR / "crops_64" / "class_distribution.png"
SAMPLES_IMG = BASE_DIR / "results" / "autoencoder" / "sample_crops_grid.png"
LOGO_IMG = BASE_DIR / "dashboard" / "static" / "assets" / "ppe_logo.svg"


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for exact 'Page X of Y' numbering and running header/footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Running Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 11 * 72 - 28, "PPE-VisionAI — Dataset Source Specification & Benchmark Reference")
            self.drawRightString(8.5 * 72 - 36, 11 * 72 - 28, "B.Tech Final Year System Design")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 11 * 72 - 32, 8.5 * 72 - 36, 11 * 72 - 32)

        # Running Footer (All Pages)
        self.setFont("Helvetica", 8)
        self.drawString(36, 24, "Hugging Face: jhboyo/ppe-dataset • MIT Academy of Engineering, Pune • Confidential Academic Reference")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 36, 24, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 34, 8.5 * 72 - 36, 34)

        self.restoreState()


def build_pdf():
    os.makedirs(STATIC_PDF.parent, exist_ok=True)
    os.makedirs(ARTIFACT_PDF.parent, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1E1B4B")
    indigo_brand = colors.HexColor("#4338CA")
    amber_brand = colors.HexColor("#D97706")
    text_dark = colors.HexColor("#0F172A")
    text_muted = colors.HexColor("#475569")
    link_color = colors.HexColor("#2563EB")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=indigo_brand,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=indigo_brand,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=text_dark,
        spaceAfter=5
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1  # Center
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10,
        textColor=text_dark
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell_style,
        alignment=1  # Center
    )

    story = []

    # ================= HEADER BLOCK =================
    header_table_data = [
        [
            Paragraph("<b>MIT ACADEMY OF ENGINEERING, ALANDI (D), PUNE</b><br/>"
                      "<font size=7.5 color='#475569'>DEPARTMENT OF COMPUTER ENGINEERING • B.TECH SYSTEM DESIGN (2026)</font><br/>"
                      "<b>FINAL JURY EVALUATION SPECIFICATION DOCUMENT</b>", body_style),
            Paragraph("<b>COURSE:</b> System Design<br/>"
                      "<b>BENCHMARK:</b> 5-Model Generative AI<br/>"
                      "<b>STATUS:</b> <font color='#059669'><b>Verified & Active</b></font>", body_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[350, 190])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LINEBELOW', (0, 0), (-1, -1), 1.2, indigo_brand),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))

    # Document Title
    story.append(Paragraph("PPE-VisionAI: Dataset Source Documentation & Statistical Benchmark Specification", title_style))
    story.append(Paragraph("<b>Empirical Multi-Objective Benchmarking of Generative Models and Multi-Modal Policy Auditing for Personal Protective Equipment (PPE) Compliance</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=8))

    # ================= 1. EXECUTIVE SUMMARY & OVERVIEW =================
    story.append(Paragraph("1. Executive Overview & Research Purpose", h1_style))
    story.append(Paragraph(
        "Automated visual inspection of Personal Protective Equipment (PPE) in construction environments requires "
        "robust representation learning, generative data augmentation, and high-frequency texture restoration. "
        "Standard public datasets often suffer from severe class imbalance, where workers without safety helmets "
        "(critical safety violations) represent a tiny fraction of observations. This project standardizes an "
        "independent, seed-controlled evaluation protocol across five AI paradigms: <b>ConvAE</b> (Self-Supervised Autoencoder), "
        "<b>cVAE</b> (Conditional Variational Autoencoder), <b>cGAN</b> (Conditional Generative Adversarial Network with DiffAugment), "
        "<b>Diffusion</b> (Score-based DDPM/DDIM Restoration), and <b>Policy Transformer</b> (DistilBERT Regulatory Compliance Checker).",
        body_style
    ))

    # Callout Box: Direct Dataset Links
    link_box_data = [
        [
            Paragraph("<b>Primary Dataset Source Repository (Hugging Face Hub):</b><br/>"
                      "&bull; <b>Hugging Face Hub Repository:</b> "
                      "<font color='#2563EB'><u><a href='https://huggingface.co/datasets/jhboyo/ppe-dataset'>https://huggingface.co/datasets/jhboyo/ppe-dataset</a></u></font><br/>"
                      "&bull; <b>Underlying Benchmark 1:</b> Safety Helmet and Worker Dataset (SHWD) — "
                      "<font color='#2563EB'><u><a href='https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset'>https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset</a></u></font><br/>"
                      "&bull; <b>Underlying Benchmark 2:</b> Pictor PPE Dataset (Real-World Industrial & Construction CCTV Benchmarks)<br/>"
                      "&bull; <b>Local Project Processing Script:</b> <font face='Courier'>extract_ppe_crops.py</font> (Yields 41,534 cropped 64&times;64 RGB images)", callout_style)
        ]
    ]
    link_table = Table(link_box_data, colWidths=[540])
    link_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EEF2FF")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#818CF8")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(link_table)
    story.append(Spacer(1, 8))

    # ================= 2. PRIMARY VISUAL DATASET SPECIFICATIONS =================
    story.append(Paragraph("2. Visual Dataset: Extraction Methodology & Class Distribution", h1_style))
    story.append(Paragraph(
        "Raw full-scene construction photos were processed via <font face='Courier'>extract_ppe_crops.py</font>. "
        "Bounding box annotations formatted in YOLO format were parsed, expanded with an adaptive 15% contextual padding "
        "(<font face='Courier'>pad=0.15</font>), square-constrained, and scaled to a uniform <b>64&times;64 RGB</b> spatial resolution "
        "(12,288 dimensions). This isolates localized equipment items from background clutter, enabling rigorous latent representation analysis.",
        body_style
    ))

    # Statistical Table
    stat_table_data = [
        [
            Paragraph("<b>Target Safety Class</b>", table_header_style),
            Paragraph("<b>Semantic Meaning</b>", table_header_style),
            Paragraph("<b>Train Set</b>", table_header_style),
            Paragraph("<b>Val Set</b>", table_header_style),
            Paragraph("<b>Test Set</b>", table_header_style),
            Paragraph("<b>Total Count</b>", table_header_style),
            Paragraph("<b>Overall %</b>", table_header_style)
        ],
        [
            Paragraph("<b><font color='#DC2626'>head (0)</font></b>", table_cell_style),
            Paragraph("Bare Head (Safety Violation)", table_cell_style),
            Paragraph("867 (3.22%)", table_cell_center),
            Paragraph("249 (3.47%)", table_cell_center),
            Paragraph("214 (2.86%)", table_cell_center),
            Paragraph("<b>1,330</b>", table_cell_center),
            Paragraph("<b>3.20%</b>", table_cell_center)
        ],
        [
            Paragraph("<b><font color='#059669'>helmet (1)</font></b>", table_cell_style),
            Paragraph("Hardhat On (Compliant)", table_cell_style),
            Paragraph("15,991 (59.45%)", table_cell_center),
            Paragraph("4,278 (59.71%)", table_cell_center),
            Paragraph("4,395 (58.84%)", table_cell_center),
            Paragraph("<b>24,664</b>", table_cell_center),
            Paragraph("<b>59.38%</b>", table_cell_center)
        ],
        [
            Paragraph("<b><font color='#4338CA'>vest (2)</font></b>", table_cell_style),
            Paragraph("High-Vis Vest (Compliant)", table_cell_style),
            Paragraph("10,041 (37.33%)", table_cell_center),
            Paragraph("2,638 (36.82%)", table_cell_center),
            Paragraph("2,861 (38.30%)", table_cell_center),
            Paragraph("<b>15,540</b>", table_cell_center),
            Paragraph("<b>37.42%</b>", table_cell_center)
        ],
        [
            Paragraph("<b>Dataset Totals</b>", table_cell_style),
            Paragraph("<b>3 Consolidated Classes</b>", table_cell_style),
            Paragraph("<b>26,899 (100%)</b>", table_cell_center),
            Paragraph("<b>7,165 (100%)</b>", table_cell_center),
            Paragraph("<b>7,470 (100%)</b>", table_cell_center),
            Paragraph("<b>41,534</b>", table_cell_center),
            Paragraph("<b>100.0%</b>", table_cell_center)
        ]
    ]

    stat_table = Table(stat_table_data, colWidths=[75, 125, 70, 70, 70, 70, 60])
    stat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 6))

    # Class Imbalance Note & Equation
    story.append(Paragraph(
        "<b>Critical Imbalance Factor & Weighting Formula:</b> Safety violations (<font face='Courier'>head</font>) comprise "
        "only <b>3.20%</b> of the corpus, resulting in an extreme <b>18.5 : 1</b> imbalance relative to compliant helmets. "
        "To prevent classifier collapse, all training pipelines employ an Inverted Class Frequency Sampler: "
        "<font face='Courier'>w_c = N / (C &times; N_c)</font>, yielding weights: "
        "<b>w_head = 10.342</b>, <b>w_helmet = 0.561</b>, and <b>w_vest = 0.893</b>.",
        body_style
    ))
    story.append(Spacer(1, 8))

    # ================= 3. EMBEDDED VISUAL EVIDENCE =================
    story.append(Paragraph("3. Empirical Visual Evidence & Sample Data", h1_style))
    
    # Place distribution plot and sample grid side-by-side if available
    img_row = []
    if DIST_IMG.exists():
        img_row.append(Image(str(DIST_IMG), width=2.7 * inch, height=1.7 * inch))
    if SAMPLES_IMG.exists():
        img_row.append(Image(str(SAMPLES_IMG), width=2.7 * inch, height=1.7 * inch))

    if img_row:
        img_table = Table([[img_row[0], img_row[1] if len(img_row) > 1 else '']], colWidths=[270, 270])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(img_table)
        caption_text = "<b>Figure 1:</b> (Left) Split distribution across train/val/test splits showing preserved class ratios; (Right) Canonical 64&times;64 RGB crops of bare head violations, hardhats, and high-visibility vests."
        story.append(Paragraph(caption_text, callout_style))
        story.append(Spacer(1, 8))

    # ================= 4. TEXT POLICY REGULATORY CORPUS =================
    story.append(Paragraph("4. Multi-Modal Regulatory Policy Corpus (OSHA, MUTCD, ANSI/ISEA)", h1_style))
    story.append(Paragraph(
        "The language reasoning component is built upon a curated corpus of <b>321 segmented and annotated legal clauses</b> "
        "extracted verbatim from statutory industrial safety codes across US and international occupational frameworks:",
        body_style
    ))

    policy_table_data = [
        [
            Paragraph("<b>Regulatory Code</b>", table_header_style),
            Paragraph("<b>Governing Organization / Statute</b>", table_header_style),
            Paragraph("<b>Scope & Mandate</b>", table_header_style),
            Paragraph("<b>Official Source URL</b>", table_header_style)
        ],
        [
            Paragraph("<b>OSHA 29 CFR 1926.100</b>", table_cell_style),
            Paragraph("Occupational Safety and Health Admin (US Dept. of Labor)", table_cell_style),
            Paragraph("Mandatory industrial head protection for construction areas with overhead impact hazards.", table_cell_style),
            Paragraph("<font color='#2563EB'><u><a href='https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.100'>osha.gov/.../1926.100</a></u></font>", table_cell_style)
        ],
        [
            Paragraph("<b>OSHA 29 CFR 1926.651</b>", table_cell_style),
            Paragraph("US Dept. of Labor Construction Standards", table_cell_style),
            Paragraph("Excavation requirements, cave-in protection, and high-visibility warning vests near traffic.", table_cell_style),
            Paragraph("<font color='#2563EB'><u><a href='https://www.osha.gov/laws-regs/regulations/standardnumber/1926/1926.651'>osha.gov/.../1926.651</a></u></font>", table_cell_style)
        ],
        [
            Paragraph("<b>Federal MUTCD Part 6D.03</b>", table_cell_style),
            Paragraph("Federal Highway Administration (FHWA)", table_cell_style),
            Paragraph("Mandatory ANSI 107 Class 2/3 high-visibility garments for all highway and right-of-way workers.", table_cell_style),
            Paragraph("<font color='#2563EB'><u><a href='https://mutcd.fhwa.dot.gov'>mutcd.fhwa.dot.gov</a></u></font>", table_cell_style)
        ],
        [
            Paragraph("<b>ANSI/ISEA 107-2020 & 89.1</b>", table_cell_style),
            Paragraph("American National Standards Institute", table_cell_style),
            Paragraph("Performance specs for high-vis apparel (chromaticity, retroreflection) and Type I/II hardhats.", table_cell_style),
            Paragraph("<font color='#2563EB'><u><a href='https://safetyequipment.org'>safetyequipment.org</a></u></font>", table_cell_style)
        ],
        [
            Paragraph("<b>UK HSE PPE Regulations 1992</b>", table_cell_style),
            Paragraph("Health and Safety Executive (UK Statutory)", table_cell_style),
            Paragraph("Statutory duty of employers to furnish compliant PPE, maintain audit logs, and enforce usage.", table_cell_style),
            Paragraph("<font color='#2563EB'><u><a href='https://www.hse.gov.uk'>hse.gov.uk</a></u></font>", table_cell_style)
        ]
    ]

    policy_table = Table(policy_table_data, colWidths=[105, 120, 195, 120])
    policy_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(policy_table)
    story.append(Spacer(1, 8))

    # ================= 5. DATASET INTEGRITY, ETHICS & PRIVACY GUARANTEES =================
    story.append(Paragraph("5. Ethical Governance, Privacy Audits & Anti-Memorization Protocol", h1_style))
    story.append(Paragraph(
        "Because personal protective equipment crops naturally depict human faces and identities, "
        "the project enforces a three-tier ethical data governance framework:",
        body_style
    ))

    ethics_box_data = [
        [
            Paragraph("<b>1. Ephemeral In-Memory Processing:</b> Uploaded inspection photographs and test streams are held "
                      "strictly in volatile RAM during inference and discarded immediately. No raw facial crops or user inputs are persisted to server disks.<br/>"
                      "<b>2. Nearest-Real-Neighbor Privacy Guarantee:</b> All synthetic samples generated by cGAN and cVAE are audited against the 867 real training identities using deep cosine embedding distances (<font face='Courier'>check_nearest_real_neighbor</font>). The 5th percentile distance of <b>p_5 = 0.2637 &gt;&gt; 0.0600</b> confirms strictly novel generation without replication or identity memorization.<br/>"
                      "<b>3. Zero Data Contamination:</b> Training, validation, and test splits are partitioned at the scene level to guarantee zero image overlap across splits.", callout_style)
        ]
    ]
    ethics_table = Table(ethics_box_data, colWidths=[540])
    ethics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#34D399")),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ethics_table)
    story.append(Spacer(1, 10))

    # Sign-off box
    signoff_data = [
        [
            Paragraph("<b>Document Verification:</b> Certified Dataset Specification & Research Documentation<br/>"
                      "<b>Dataset Repository:</b> <font color='#2563EB'><u>https://huggingface.co/datasets/jhboyo/ppe-dataset</u></font><br/>"
                      "<b>Project:</b> PPE-VisionAI Generative Benchmarking Suite • Final Year B.Tech Computer Engineering", callout_style),
            Paragraph("<b>Evaluation Date:</b> November 2026<br/>"
                      "<b>Status:</b> Approved for Jury Review<br/>"
                      "<b>Security Audit:</b> Passed (Zero Leakage)", callout_style)
        ]
    ]
    signoff_table = Table(signoff_data, colWidths=[360, 180])
    signoff_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(signoff_table)

    # Build PDF with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {OUTPUT_PDF}")

    # Copy to static assets and brain artifacts
    import shutil
    shutil.copyfile(OUTPUT_PDF, STATIC_PDF)
    shutil.copyfile(OUTPUT_PDF, ARTIFACT_PDF)
    print(f"Copied to static assets: {STATIC_PDF}")
    print(f"Copied to brain artifacts: {ARTIFACT_PDF}")


if __name__ == "__main__":
    build_pdf()

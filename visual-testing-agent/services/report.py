from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(report_path, image_path, summary):
    doc = SimpleDocTemplate(report_path)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph("<b>AI Visual Testing Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Summary (split lines properly)
    for line in summary.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 8))

    elements.append(Spacer(1, 20))

    # Image
    elements.append(Paragraph("<b>Detected Differences:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    elements.append(Image(image_path, width=350, height=600))

    doc.build(elements)
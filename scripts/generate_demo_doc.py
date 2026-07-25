"""
Generates a synthetic, fully-fictional "annual report" PDF used as the
demo corpus. It's public-safe (no real company, no real data) and
deliberately spreads related facts across different sections/pages, so
that some natural questions genuinely require pulling from more than
one part of the document -- that's what makes the decompose /
multi-hop-retrieve behavior demonstrable instead of decorative.

Run: python scripts/generate_demo_doc.py
"""
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "northfield_robotics_annual_report.pdf")

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=14)
body = ParagraphStyle("Body", parent=styles["BodyText"], spaceAfter=10, leading=15)

SECTIONS = [
    ("Company Overview", [
        "Northfield Robotics, Inc. (\"Northfield\" or \"the Company\") is a fictional manufacturer of "
        "industrial automation equipment, founded in 2011 and headquartered in Portland, Oregon. The "
        "Company designs, manufactures, and services robotic arms, autonomous mobile robots (AMRs), and "
        "warehouse automation software used by manufacturing and logistics customers.",
        "As of fiscal year 2024, Northfield employed approximately 3,200 people across four manufacturing "
        "sites in the United States, Mexico, and Vietnam, plus regional sales and support offices in "
        "Germany, Japan, and Singapore.",
        "The Company operates in two reportable segments: Robotic Arms and Warehouse Automation. This "
        "document is a fictional composite created for demonstration purposes only and does not describe "
        "any real company.",
    ]),
    ("Business Segments and Products", [
        "The Robotic Arms segment designs six-axis and collaborative ('cobot') robotic arms for welding, "
        "assembly, and material handling. Flagship products include the NF-Series six-axis arms and the "
        "CoboLite collaborative arm line introduced in fiscal year 2023.",
        "The Warehouse Automation segment provides autonomous mobile robots and a fleet-management "
        "software platform called PathFinder OS, which coordinates robot routing, charging, and task "
        "allocation across a customer's warehouse floor.",
        "In fiscal year 2024, the Robotic Arms segment accounted for 61% of total revenue and the "
        "Warehouse Automation segment accounted for 39% of total revenue.",
    ]),
    ("Financial Performance", [
        "Total revenue for fiscal year 2024 was $842 million, up 14% from $739 million in fiscal year "
        "2023. Robotic Arms segment revenue grew 9% year-over-year to $514 million. Warehouse Automation "
        "segment revenue grew 23% year-over-year to $328 million, driven primarily by higher PathFinder "
        "OS software attach rates on new AMR deployments.",
        "Gross margin for fiscal year 2024 was 41.2%, compared with 39.8% in fiscal year 2023. Management "
        "attributes the improvement primarily to software mix shift within Warehouse Automation, offset "
        "partially by higher input costs in the Robotic Arms segment described in the Risk Factors section.",
        "Operating expenses were $268 million, including $112 million in research and development. "
        "Net income for fiscal year 2024 was $96 million, compared with $71 million in fiscal year 2023.",
    ]),
    ("Risk Factors", [
        "Supply chain concentration: A significant portion of the servo motors and precision gearboxes "
        "used in the Robotic Arms segment are sourced from a small number of suppliers in Taiwan and "
        "South Korea. Extended lead times or a supply disruption affecting these components could delay "
        "shipments and increase input costs, which the Company believes contributed to margin pressure "
        "in the Robotic Arms segment during fiscal year 2024.",
        "Competitive pressure: The Company competes with larger, better-capitalized industrial automation "
        "vendors, some of whom have broader product lines and deeper customer relationships.",
        "Customer concentration: The Company's ten largest customers accounted for approximately 34% of "
        "fiscal year 2024 revenue. The loss of one or more of these customers could adversely affect "
        "results.",
        "Regulatory and trade policy: Tariffs or export restrictions affecting robotics components or "
        "finished goods moving between the Company's manufacturing sites in Mexico and Vietnam and its "
        "primary US and European customer base could increase costs or disrupt deliveries.",
    ]),
    ("Research and Development", [
        "Research and development expense was $112 million in fiscal year 2024, representing 13.3% of "
        "revenue, compared with $94 million, or 12.7% of revenue, in fiscal year 2023.",
        "Current R&D priorities include next-generation force-sensing end effectors for the CoboLite "
        "line, and expanding PathFinder OS to support multi-vendor robot fleets rather than only "
        "Northfield-manufactured hardware.",
    ]),
    ("Executive Compensation", [
        "For fiscal year 2024, the Company's named executive officers received total compensation as "
        "follows (base salary, bonus, and equity awards combined): Chief Executive Officer, $8.4 million; "
        "Chief Financial Officer, $4.1 million; Chief Technology Officer, $3.9 million.",
        "A majority of executive compensation is delivered in the form of performance-based restricted "
        "stock units that vest over three years, subject to Company revenue and operating margin targets.",
    ]),
    ("Environmental, Social, and Governance", [
        "The Company's Portland manufacturing site transitioned to 100% renewable electricity in fiscal "
        "year 2024. The Company has set a target to reduce Scope 1 and Scope 2 greenhouse gas emissions "
        "intensity by 30% by fiscal year 2028, relative to a fiscal year 2022 baseline.",
        "The Company's board of directors includes seven members, four of whom are independent. The "
        "board's Audit Committee is composed entirely of independent directors.",
    ]),
    ("Legal Proceedings", [
        "The Company is not currently a party to any material legal proceedings outside the ordinary "
        "course of business. From time to time, the Company is subject to claims and legal actions "
        "arising in the ordinary course of business, none of which are expected, individually or in the "
        "aggregate, to have a material adverse effect on the Company's financial condition.",
    ]),
]


def build():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    doc = SimpleDocTemplate(OUT_PATH, pagesize=LETTER,
                             topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    story = [
        Paragraph("Northfield Robotics, Inc.", styles["Title"]),
        Paragraph("Fiscal Year 2024 Annual Report (Fictional Demo Document)", styles["Heading3"]),
        Spacer(1, 0.3 * inch),
        Paragraph(
            "This document is entirely fictional, created for demonstration purposes only. "
            "Northfield Robotics, Inc. is not a real company. Any resemblance to real companies, "
            "products, or financial results is coincidental.",
            body,
        ),
        PageBreak(),
    ]
    for title, paragraphs in SECTIONS:
        story.append(Paragraph(title, h1))
        for p in paragraphs:
            story.append(Paragraph(p, body))
        story.append(PageBreak())

    doc.build(story)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    build()

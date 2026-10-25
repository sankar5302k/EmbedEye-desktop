"""Generate 20 sample images for OCR mode verification.

10 document images with text
5 screenshots with text
5 plain photos without text (should be discarded as alnum < 4)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

out_dir = Path("data/sample_ocr_test")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. 10 Images with text
text_samples = [
    ("doc_01_electricity_bill.png", [
        "CITY POWER & LIGHT - ELECTRICITY BILL",
        "Account Number: 9876-5432-10",
        "Billing Period: October 2026",
        "Total Amount Due: $142.50",
        "Due Date: November 15, 2026",
        "Payment Status: PENDING",
    ]),
    ("doc_02_tax_invoice.jpg", [
        "TAX INVOICE #INV-2026-889",
        "Customer: ACME Corporation",
        "Item: Cloud Server Compute Hours",
        "Subtotal: $1,250.00 | VAT (18%): $225.00",
        "Grand Total: $1,475.00 USD",
    ]),
    ("doc_03_medical_receipt.png", [
        "METRO GENERAL HOSPITAL - PHARMACY RECEIPT",
        "Patient ID: PT-44029",
        "Prescription: Amoxicillin 500mg, Paracetamol 650mg",
        "Doctor: Dr. Sarah Jenkins, MD",
        "Total Paid: $38.20 (Credit Card)",
    ]),
    ("doc_04_flight_boarding_pass.png", [
        "AIR HORIZON BOARDING PASS",
        "Passenger: John Doe",
        "Flight: AH-402 | Seat: 14B",
        "From: New York (JFK) To: London (LHR)",
        "Gate: B22 | Boarding Time: 18:30",
    ]),
    ("doc_05_rental_contract.png", [
        "APARTMENT LEASE AGREEMENT",
        "Landlord: Oakridge Properties LLC",
        "Tenant: Alice Wonderland",
        "Monthly Rent: $2,100 payable on 1st of month",
        "Security Deposit: $2,100 paid in full",
    ]),
    ("doc_06_shipping_manifest.png", [
        "EXPRESS COURIER SHIPPING LABEL",
        "Tracking: 1Z9999999999999999",
        "Recipient: Tech Solutions Inc, Floor 4",
        "Weight: 4.8 kg | Service: Standard Express",
        "Signature Required on Delivery",
    ]),
    ("doc_07_certificate_award.png", [
        "CERTIFICATE OF COMPLETION",
        "Awarded to: Bob Williams",
        "Course: Advanced Artificial Intelligence & Computer Vision",
        "Issued by: Global Academy of Computing",
        "Grade: Distinction (98%)",
    ]),
    ("doc_08_book_page.jpg", [
        "CHAPTER 4: THE ARCHITECTURE OF NEURAL NETWORKS",
        "Deep learning models leverage hierarchical representations",
        "to automatically discover relevant features from raw data.",
        "Convolutional and transformer architectures have revolutionized",
        "natural language processing and image recognition.",
    ]),
    ("doc_09_coffee_shop_bill.png", [
        "THE DAILY BEAN COFFEE SHOP",
        "Order #104 | Server: Mike",
        "1x Iced Vanilla Latte: $5.50",
        "1x Almond Croissant: $4.25",
        "Total with Tax: $10.58 | Tip: $2.00",
    ]),
    ("doc_10_business_card.png", [
        "ALEX VANCE - SENIOR SOFTWARE ENGINEER",
        "Nova Robotics Laboratories",
        "Email: alex.vance@novarobotics.io",
        "Phone: +1 (555) 019-2834",
        "San Francisco, California",
    ]),
]

for filename, lines in text_samples:
    img = Image.new("RGB", (800, 500), color=(250, 250, 252))
    d = ImageDraw.Draw(img)
    d.rectangle([15, 15, 785, 485], outline=(200, 210, 220), width=2)
    y = 50
    for line in lines:
        d.text((40, y), line, fill=(20, 30, 45))
        y += 45
    img.save(out_dir / filename)

# 2. 5 Screenshots with text
screenshots = [
    ("screen_01_terminal.png", [
        "user@machine:~$ python -m src.ocr_search 'electricity bill'",
        "[INFO] Loaded MiniLM embedder (384-d, CPU)",
        "[INFO] Reciprocal Rank Fusion retrieved 20 candidates",
        "[INFO] Reranking complete in 42ms",
    ], (15, 23, 42), (56, 189, 248)),
    ("screen_02_vscode.png", [
        "def test_gpu_memory_isolation():",
        "    manager = ModelManager()",
        "    with manager.use('florence') as model:",
        "        assert torch.cuda.memory_allocated() > 0",
        "    assert torch.cuda.memory_allocated() == 0",
    ], (30, 30, 30), (220, 220, 220)),
    ("screen_03_browser_tab.png", [
        "https://github.com/UB-Mannheim/tesseract/wiki",
        "Tesseract OCR for Windows - Open Source Engine",
        "Current stable release: 5.5.0 with LSTM neural net models",
        "Supports over 100 languages with fast accuracy",
    ], (240, 242, 245), (15, 23, 42)),
    ("screen_04_dashboard.png", [
        "SYSTEM MONITOR - RTX 3050 4GB",
        "GPU Temperature: 54 C | Utilization: 12%",
        "VRAM Usage: 1240 MB / 4096 MB (30.2%)",
        "Active Task: Neural Cross-Encoder Reranker",
    ], (17, 24, 39), (74, 222, 128)),
    ("screen_05_error_dialog.png", [
        "APPLICATION ERROR: 0x800401F5",
        "Exception: Network connection timeout",
        "Retry count: 3 of 3 exceeded",
        "Please check proxy configuration or run in offline mode",
    ], (254, 242, 242), (185, 28, 28)),
]

for filename, lines, bg_color, text_color in screenshots:
    img = Image.new("RGB", (800, 400), color=bg_color)
    d = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        d.text((30, y), line, fill=text_color)
        y += 45
    img.save(out_dir / filename)

# 3. 5 Plain photos without text (should be discarded by OCR alnum < 4)
plain_photos = [
    ("photo_01_green_forest.jpg", (34, 139, 34)),
    ("photo_02_blue_ocean.jpg", (24, 116, 205)),
    ("photo_03_desert_sand.jpg", (238, 201, 139)),
    ("photo_04_sunset_sky.jpg", (238, 99, 99)),
    ("photo_05_snow_mountain.jpg", (230, 240, 250)),
]

for filename, base_color in plain_photos:
    img = Image.new("RGB", (600, 450), color=base_color)
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    img.save(out_dir / filename)

print(f"Generated 20 sample images successfully in: {out_dir}")

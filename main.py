import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import re
import os
import io
import tempfile
import zipfile
import pandas as pd
from typing import List, Tuple, Dict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import base64
import imaplib

load_dotenv()
import streamlit.components.v1 as components
# ------------------ Configuration defaults ------------------
DEFAULT_POPPLER = r"C:\Program Files\poppler-23.05.0\Library\bin"
DEFAULT_TESSERACT = ""  # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Email Configuration (Your credentials)
SMTP_SERVER = st.secrets["smtp"]["server"]
SMTP_PORT = st.secrets["smtp"]["port"]
SENDER_EMAIL = st.secrets["smtp"]["sender_email"]
SENDER_PASSWORD = st.secrets["smtp"]["sender_password"]
IMAP_SERVER = st.secrets["imap"]["server"]
IMAP_PORT = st.secrets["imap"]["port"]



# Test recipient email (hardcoded for testing)
TEST_RECIPIENT = "ajay@atulsales.com"

INVOICE_KEYWORDS = [r"tax\s+invoice", r"invoice\s+no", r"invoice\s+#", r"invoice\b"]
RECEIVER_LABELS = [
    "details of receiver", "receiver", "billed to", "bill to", "buyer", "consignee", "ship to", "shipped to"
]


# ------------------ MailSender Class ------------------

class MailSender:
    def __init__(self, smtp_server, smtp_port, username, password, imap_server, imap_port):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port

    def send_invoice_email(self, pdf_path: str, receiver_name: str, invoice_no: str,
                           invoice_date: str, net_amount: str, recipient_email: str, doc_type: str = 'Invoice') -> Tuple[bool, str]:
        """
        Sends an email with PDF attachment
        Returns: (success: bool, message: str)
        """
        try:
            # Clean up receiver name for display
            display_name = receiver_name.replace('(not found)', 'Valued Customer')

            # Format date for display (convert from DD-MMM-YYYY to DDth Month YYYY)
            formatted_date = self._format_date_for_display(invoice_date)

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient_email
            msg['Subject'] = f"{doc_type} for {display_name} - {invoice_no}"

            # Email body
            body = f"""Dear {display_name},

Please find attached the {doc_type} ({invoice_no}) dated {formatted_date} for your reference.

Kindly review the details and process the payment as per the agreed terms. Should you have any questions or require clarification, please feel free to get in touch.

Thank you for your continued business.

Best regards,
Kantascrypt Team
{self.username}
            """

            msg.attach(MIMEText(body, 'plain'))

            # Attach PDF
            filename = os.path.basename(pdf_path)
            with open(pdf_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename= {filename}')
                msg.attach(part)

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, recipient_email, text)
            server.quit()

            return True, f"✅ Email sent successfully to {recipient_email}"

        except Exception as e:
            return False, f"❌ Failed to send email: {str(e)}"

    def _format_date_for_display(self, date_str: str) -> str:
        """Convert date from DD-MMM-YYYY to DDth Month YYYY format"""
        try:
            if date_str == '(not found)':
                return 'N/A'

            # Parse the date (e.g., "28-Oct-2025")
            date_obj = datetime.strptime(date_str, "%d-%b-%Y")

            # Get day with suffix (1st, 2nd, 3rd, 4th, etc.)
            day = date_obj.day
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = "th"
            else:
                suffix = ["st", "nd", "rd"][day % 10 - 1]

            # Format as "28th October 2025"
            return date_obj.strftime(f"%d{suffix} %B %Y")
        except:
            return date_str


# ------------------ Helper functions ------------------

def get_poppler_path(cfg_poppler: str):
    if cfg_poppler and os.path.isdir(cfg_poppler):
        return cfg_poppler
    return None

def auto_download_multiple(files: List[Tuple[str, bytes, str]], delay_ms: int = 300):
    """
    files: list of tuples (filename, bytes_data, mime_type)
    delay_ms: milliseconds between subsequent auto-clicks (helps browser)
    This renders a small HTML/JS that clicks hidden anchors to start downloads.
    """
    anchors = []
    for i, (fname, data, mime) in enumerate(files):
        b64 = base64.b64encode(data).decode()
        href = f"data:{mime};base64,{b64}"
        # create unique id for each anchor
        anchors.append(f'<a id="dl{i}" href="{href}" download="{fname}"></a>')
    anchors_html = "\n".join(anchors)

    # JS to click anchors sequentially with small delay
    click_script = "<script>\n"
    for i in range(len(files)):
        click_script += f"setTimeout(function(){{document.getElementById('dl{i}').click();}}, {i * delay_ms});\n"
    click_script += "</script>"

    html = f"""
    {anchors_html}
    {click_script}
    """

    # height small but > 0 so script runs
    components.html(html, height=10)

def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts)
    except Exception as e:
        st.debug(f"extract_text_from_pdf error: {e}")
        return ""


def ocr_pdf_pages(path: str, poppler_path: str = None) -> str:
    try:
        images = convert_from_path(path, dpi=200, poppler_path=poppler_path)
        page_texts = []
        for img in images:
            gray = img.convert("L")
            text = pytesseract.image_to_string(gray)
            page_texts.append(text)
        return "\n".join(page_texts)
    except Exception as e:
        st.debug(f"ocr_pdf_pages error: {e}")
        return ""


def get_pdf_text(path: str, poppler_path: str = None) -> Tuple[str, str]:
    text = extract_text_from_pdf(path)
    if text and len(text.strip()) > 50:
        return text, 'direct'
    text = ocr_pdf_pages(path, poppler_path)
    return text, 'ocr'


# ------------------ Invoice Information Extraction ------------------

def extract_invoice_number(text: str) -> str:
    """Extract invoice number from text - looks for MH followed by any digits"""
    if not text:
        return '(not found)'

    # Primary Pattern: MH followed by any number of digits (at least 8 digits for invoice number)
    # This will match MH0125638975 even if it appears at the end of other text
    pattern = r'MH\d{8,}'
    match = re.search(pattern, text)
    if match:
        return match.group(0).strip()

    return '(not found)'


def extract_net_amount(text: str) -> str:
    """Extract net amount from text"""
    if not text:
        return '(not found)'

    # Pattern 1: Net Amount 52,500.00 or Net Amount: 52,500.00
    pattern1 = r'Net\s+Amount\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)'
    match = re.search(pattern1, text, flags=re.IGNORECASE)
    if match:
        amount = match.group(1).strip()
        return amount

    # Pattern 2: Total Amount or Grand Total
    pattern2 = r'(?:Total\s+Amount|Grand\s+Total)\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)'
    match = re.search(pattern2, text, flags=re.IGNORECASE)
    if match:
        amount = match.group(1).strip()
        return amount

    # Pattern 3: Amount Payable
    pattern3 = r'Amount\s+Payable\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)'
    match = re.search(pattern3, text, flags=re.IGNORECASE)
    if match:
        amount = match.group(1).strip()
        return amount

    return '(not found)'


def extract_invoice_date(text: str) -> str:
    """Extract invoice date from text"""
    if not text:
        return '(not found)'

    # Pattern 1: Invoice Date 28-Oct-2025 or Invoice Date: 28-Oct-2025
    pattern1 = r'Invoice\s+Date\s*[:\-]?\s*(\d{1,2}[-/]\w{3}[-/]\d{4})'
    match = re.search(pattern1, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: Date: 28-Oct-2025
    pattern2 = r'Date\s*[:\-]\s*(\d{1,2}[-/]\w{3}[-/]\d{4})'
    match = re.search(pattern2, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 3: DD/MM/YYYY format
    pattern3 = r'Invoice\s+Date\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})'
    match = re.search(pattern3, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 4: Date in DD-MM-YYYY format
    pattern4 = r'Date\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})'
    match = re.search(pattern4, text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return '(not found)'


def extract_receiver_name(text: str):
    if not text:
        return None

    receiver = None

    # Method 1
    pattern = r"(?:details\s+of\s+receiver|billed\s+to|receiver|buyer)[\s\S]{0,200}?name\s*[:\-]\s*([A-Z][A-Z\s&\.,\-\(\)]+\[\s*\d+\s*\])"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        receiver = match.group(1).strip()
        receiver = ' '.join(receiver.split())
        if ']' in receiver:
            receiver = receiver[:receiver.rindex(']') + 1].strip()
        return receiver

    # Method 2
    pattern = r"name\s*[:\-]\s*([A-Z][A-Z\s&\.,\-\(\)]+\[\s*\d+\s*\])"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        receiver = match.group(1).strip()
        receiver = ' '.join(receiver.split())
        if ']' in receiver:
            receiver = receiver[:receiver.rindex(']') + 1].strip()
        return receiver

    # Method 3
    pattern = r'([A-Z][A-Z\s&\.,\-\(\)]{10,80}\[\s*\d{5,}\s*\])'
    matches = re.findall(pattern, text)
    if matches:
        exclude_words = ['INVOICE', 'TAX INVOICE', 'GSTIN', 'DETAILS OF']
        for m in matches:
            if not any(ex in m.upper() for ex in exclude_words):
                receiver = m.strip()
                receiver = ' '.join(receiver.split())
                if ']' in receiver:
                    receiver = receiver[:receiver.rindex(']') + 1].strip()
                return receiver

    # Method 4 - search within labels
    for label in RECEIVER_LABELS:
        pattern = rf"{re.escape(label)}[\s\S]{{1,700}}?(?=\n\s*\n|DL NO|PAN NO|State|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            block = match.group(0)
            name_match = re.search(r"name\s*[:\-]\s*([A-Z][A-Z\s&\.,\-\(\)]+\[\s*\d+\s*\])", block, flags=re.IGNORECASE)
            if name_match:
                candidate = ' '.join(name_match.group(1).split())
                if ']' in candidate:
                    candidate = candidate[:candidate.rindex(']') + 1].strip()
                return candidate

    # Method 5
    pattern = r'\n([A-Z]\.?.\s+[A-Z][A-Z\s&\.,\-\(\)]{5,70}\[\s*\d+\s*\])'
    matches = re.findall(pattern, text)
    if matches:
        exclude_words = ['INVOICE', 'TAX', 'DETAILS', 'GSTIN', 'STATE']
        for m in matches:
            if not any(w in m.upper() for w in exclude_words):
                receiver = ' '.join(m.split())
                if ']' in receiver:
                    receiver = receiver[:receiver.rindex(']') + 1].strip()
                return receiver

    # Method 6
    pattern = r'([A-Z][A-Z\s\.\&,\-\(\)]{8,80}\[\s*\d{4,}\s*\])'
    matches = re.findall(pattern, text)
    for m in matches:
        words_before_bracket = m.split('[')[0].strip().split()
        if len(words_before_bracket) >= 2:
            exclude = ['INVOICE', 'GSTIN', 'STATE', 'DETAILS']
            if not any(e in m.upper() for e in exclude):
                receiver = ' '.join(m.split())
                if ']' in receiver:
                    receiver = receiver[:receiver.rindex(']') + 1].strip()
                return receiver

    return receiver


def sanitize_filename(name: str):
    if not name:
        return None
    if ']' in name:
        bracket_end = name.rindex(']')
        name = name[:bracket_end + 1]
    name = ' '.join(name.split())
    bracket_number = None
    bracket_match = re.search(r'\[\s*(\d+)\s*\]', name)
    if bracket_match:
        bracket_number = bracket_match.group(1)
        name = name[:bracket_match.start()].strip()
    name = re.sub(r'\(([A-Za-z0-9\s]+)\)', r'_\1', name)
    name = name.replace('.', '')
    name = re.sub(r'[^\w\s\-]', '', name)
    name = ' '.join(name.split())
    name = name.replace(' ', '_')
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if bracket_number:
        name = f"{name}_{bracket_number}"
    return name if name else None


def extract_all_invoice_info(text: str) -> Dict[str, str]:
    """Extract all invoice information from text"""
    receiver = extract_receiver_name(text)
    customer_code = '(not found)'
    if receiver:
        m = re.search(r'\[\s*(\d+)\s*\]', receiver)
        if m:
            customer_code = m.group(1)

    return {
        'receiver': receiver or '(not found)',
        'customer_code': customer_code,
        'invoice_no': extract_invoice_number(text),
        'net_amount': extract_net_amount(text),
        'invoice_date': extract_invoice_date(text)
    }


# ------------------ PDF operations ------------------

def merge_pdfs_bytes(files: List[io.BytesIO]) -> io.BytesIO:
    writer = PdfWriter()
    for b in files:
        b.seek(0)
        reader = PdfReader(b)
        for p in reader.pages:
            writer.add_page(p)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out


def split_pdf_by_pages_file(src_path: str, pages_per_invoice: int, output_folder: str) -> List[dict]:
    """Returns list of dicts with path, receiver, customer_code, invoice details"""
    reader = PdfReader(src_path)
    total = len(reader.pages)
    exported = []
    idx = 0
    counter = 1
    while idx < total:
        writer = PdfWriter()
        start = idx
        text_chunk = ""
        for _ in range(pages_per_invoice):
            if idx >= total:
                break
            p = reader.pages[idx]
            writer.add_page(p)
            try:
                text_chunk += p.extract_text() or ""
            except Exception:
                pass
            idx += 1

        # Extract all invoice information
        info = extract_all_invoice_info(text_chunk)
        safe = sanitize_filename(info['receiver'])

        if safe:
            fname = f"{safe}.pdf"
            path = os.path.join(output_folder, fname)
            if os.path.exists(path):
                fname = f"{safe}_{counter}.pdf"
                path = os.path.join(output_folder, fname)
                counter += 1
        else:
            fname = f"invoice_pages_{start + 1}_to_{idx}.pdf"
            path = os.path.join(output_folder, fname)

        with open(path, 'wb') as f:
            writer.write(f)

        exported.append({
            'path': path,
            'filename': fname,
            **info  # Unpack all invoice info
        })
    return exported


def merge_pdf_paths(paths: List[str], out_path: str):
    """Merge a list of existing PDF files (paths) into a single out_path file."""
    writer = PdfWriter()
    for p in paths:
        try:
            r = PdfReader(p)
            for pg in r.pages:
                writer.add_page(pg)
        except Exception as e:
            st.debug(f"merge_pdf_paths: failed to read {p}: {e}")
    with open(out_path, "wb") as f:
        writer.write(f)


def split_pdf_auto_detect_file(src_path: str, output_folder: str, poppler_path: str = None) -> List[dict]:
    """Improved auto-detect splitting:
    - groups pages by extracted invoice number (if available)
    - merges split-chunks that resolved to the same invoice number
    - reduces duplicate sending for multi-page invoices
    """
    reader = PdfReader(src_path)
    num_pages = len(reader.pages)

    # Extract text per page
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")

    # If pages are empty, try OCR once and split returned OCR text into page-like blocks
    if all(not t.strip() for t in page_texts):
        ocr_text = ocr_pdf_pages(src_path, poppler_path)
        # keep same heuristic as original but ensure length matches approx pages by splitting on double newline
        page_texts = ocr_text.split('\n\n')

    # Build page_info: has_kw, invoice_no, raw text
    page_info = []
    for t in page_texts:
        tl = t.lower()
        has_kw = any(re.search(kw, tl) for kw in INVOICE_KEYWORDS)
        inv_no = extract_invoice_number(t)
        page_info.append({'text': t, 'has_kw': has_kw, 'invoice_no': inv_no})

    # Propagate invoice numbers forward then backward so pages that don't contain the number inherit nearby number
    # Forward pass
    for i in range(1, len(page_info)):
        if page_info[i]['invoice_no'] == '(not found)' and page_info[i - 1]['invoice_no'] != '(not found)':
            page_info[i]['invoice_no'] = page_info[i - 1]['invoice_no']
    # Backward pass
    for i in range(len(page_info) - 2, -1, -1):
        if page_info[i]['invoice_no'] == '(not found)' and page_info[i + 1]['invoice_no'] != '(not found)':
            page_info[i]['invoice_no'] = page_info[i + 1]['invoice_no']

    # Decide split start indices:
    # Start at page 0, then start a new chunk only when invoice_no changes.
    starts = [0]
    for i in range(1, len(page_info)):
        prev = page_info[i - 1]
        cur = page_info[i]

        # Primary rule: invoice number changed -> new invoice
        if cur['invoice_no'] != prev['invoice_no']:
            starts.append(i)
            continue

        # Fallback: when both pages lack invoice_no but current page has an invoice keyword while previous didn't,
        # we treat that as a likely new invoice beginning (covers some header-only cases).
        if prev['invoice_no'] == '(not found)' and cur['invoice_no'] == '(not found)' and cur['has_kw'] and not prev[
            'has_kw']:
            starts.append(i)
            continue

        # Otherwise, treat as continuation of same invoice
        # (this avoids creating separate files when header/footer contains "Invoice" on every page)

    # Build chunks based on starts
    exported = []
    for idx, start in enumerate(starts):
        if idx + 1 < len(starts):
            end = starts[idx + 1] - 1
        else:
            end = len(page_info) - 1

        writer = PdfWriter()
        chunk_text = ""
        for pnum in range(start, end + 1):
            try:
                writer.add_page(reader.pages[pnum])
                chunk_text += page_info[pnum]['text']
            except Exception:
                pass

        # Extract info for the chunk
        info = extract_all_invoice_info(chunk_text)
        safe = sanitize_filename(info['receiver'])

        if safe:
            fname = f"{safe}.pdf"
            path = os.path.join(output_folder, fname)
            # avoid overwriting, add suffix if exists
            counter = 1
            base_no_ext = os.path.splitext(fname)[0]
            ext = os.path.splitext(fname)[1]
            while os.path.exists(path):
                path = os.path.join(output_folder, f"{base_no_ext}_{counter}{ext}")
                counter += 1
        else:
            fname = f"invoice_detected_{start + 1}_to_{end + 1}.pdf"
            path = os.path.join(output_folder, fname)
            counter = 1
            base_no_ext = os.path.splitext(fname)[0]
            ext = os.path.splitext(fname)[1]
            while os.path.exists(path):
                path = os.path.join(output_folder, f"{base_no_ext}_{counter}{ext}")
                counter += 1

        with open(path, 'wb') as f:
            writer.write(f)

        exported.append({
            'path': path,
            'filename': os.path.basename(path),
            **info
        })

    # Post-process: merge any exported items that resolved to the same invoice_no (non '(not found)')
    # This avoids creating multiple files (and duplicate emails) for the same invoice number.
    merged = []
    by_inv = {}
    for item in exported:
        key = item['invoice_no'] if item['invoice_no'] != '(not found)' else None
        if key:
            by_inv.setdefault(key, []).append(item)
        else:
            merged.append(item)  # keep unknown-invoice-number items as-is

    for inv, items in by_inv.items():
        if len(items) == 1:
            merged.append(items[0])
        else:
            # merge their PDFs into a single file named Invoice_<inv>.pdf (or receiver if safe)
            first = items[0]
            receiver_safe = sanitize_filename(first.get('receiver') or '')
            out_name = f"Invoice_{inv}.pdf" if not receiver_safe else f"{receiver_safe}_{inv}.pdf"
            out_path = os.path.join(output_folder, out_name)
            merge_pdf_paths([it['path'] for it in items], out_path)

            merged.append({
                'path': out_path,
                'filename': os.path.basename(out_path),
                'receiver': first.get('receiver', '(not found)'),
                'customer_code': first.get('customer_code', '(not found)'),
                'invoice_no': inv,
                'net_amount': first.get('net_amount', '(not found)'),
                'invoice_date': first.get('invoice_date', '(not found)')
            })

    return merged


# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Invoice PDF Splitter", layout='wide')

st.title("Atul Sales PDF Splitter")
# st.markdown(
#     "A Streamlit UI for merging, splitting and extracting invoiced receiver names. Works with text-based PDFs and scanned PDFs (OCR).")

# Initialize MailSender
mailer = MailSender(
    smtp_server=SMTP_SERVER,
    smtp_port=SMTP_PORT,
    username=SENDER_EMAIL,
    password=SENDER_PASSWORD,
    imap_server=IMAP_SERVER,
    imap_port=IMAP_PORT
)

## Settings removed (hardcoded)
# Settings moved into script for fast workflow
poppler_path = DEFAULT_POPPLER
tess_cmd = DEFAULT_TESSERACT
ocr_enabled = True

if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

col1, col2 = st.columns([10, 2])

# ---- Merge PDFs ----
# with col1:
#     st.subheader('1) Merge PDFs')
#     uploaded_for_merge = st.file_uploader('Upload PDFs to merge (multiple)', type=['pdf'], accept_multiple_files=True)
#     if uploaded_for_merge:
#         st.write(f"Selected: {len(uploaded_for_merge)} files")
#         for f in uploaded_for_merge:
#             st.write(f"📄 {f.name}")
#     merge_name = st.text_input('Merged filename', value='merged_invoices.pdf')
#     if st.button('🔗 Merge and Download'):
#         if not uploaded_for_merge:
#             st.warning('Please upload PDF files to merge first.')
#         else:
#             st.info('Merging...')
#             buffers = []
#             for up in uploaded_for_merge:
#                 buffers.append(io.BytesIO(up.read()))
#             merged = merge_pdfs_bytes(buffers)
#             st.success('✅ Merge complete — ready to download')
#             st.download_button('⬇️ Download Merged PDF', merged, file_name=merge_name, mime='application/pdf')

# ---- Split PDF ----
with col1:
    st.subheader('2) Split PDF into invoices')

    # NEW: Document type dropdown (showcase)
    doc_type = st.selectbox('Document type (for splitting / naming)',
                            options=['Invoice', 'Challan', 'Debit Note', 'Receipt', 'Other'], index=0)

    uploaded_split = st.file_uploader('Upload a single PDF to split', type=['pdf'])
    split_mode = st.radio('Split mode', ('Auto-detect invoices', 'Every N pages'))
    pages_per = st.number_input('Pages per invoice (when using Every N pages)', min_value=1, max_value=200, value=1)

    if st.button('✂️ Split PDF, Generate Excel & Send Emails'):
        if not uploaded_split:
            st.warning('Please upload a PDF to split.')
        else:
            with st.spinner('Processing...'):
                tmpdir = tempfile.mkdtemp()
                src_path = os.path.join(tmpdir, uploaded_split.name)
                with open(src_path, 'wb') as f:
                    f.write(uploaded_split.read())

                poppler = get_poppler_path(poppler_path) if ocr_enabled else None
                try:
                    if split_mode == 'Every N pages':
                        exported = split_pdf_by_pages_file(src_path, pages_per, tmpdir)
                    else:
                        exported = split_pdf_auto_detect_file(src_path, tmpdir, poppler)

                    if exported:
                        # Create Excel file
                        excel_data = []
                        for item in exported:
                            excel_data.append({
                                'Doc Type': doc_type,
                                'Filename': item['filename'],
                                'Customer Code': item['customer_code'],
                                'Receiver Name': item['receiver'],
                                'Invoice No': item['invoice_no'],
                                'Invoice Date': item['invoice_date'],
                                # 'Net Amount': item['net_amount']
                            })

                        df = pd.DataFrame(excel_data)
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Invoices')
                        excel_buffer.seek(0)

                        # Create ZIP with PDFs
                        zipbuf = io.BytesIO()
                        with zipfile.ZipFile(zipbuf, 'w') as z:
                            for item in exported:
                                z.write(item['path'], arcname=item['filename'])
                            # Add Excel to ZIP
                            z.writestr('invoice_summary.xlsx', excel_buffer.getvalue())
                        zipbuf.seek(0)

                        st.success(f'✅ Split complete — {len(exported)} file(s) created')

                        # Display summary table
                        st.dataframe(df, use_container_width=True)

                        # Debug: Show extracted text from first invoice
                        # with st.expander('🔍 Debug: View extracted text and extraction tests'):
                        #     if exported:
                        #         first_item = exported[0]
                        #         debug_text, _ = get_pdf_text(first_item['path'], poppler)
                        #
                        #         st.subheader('Raw Text Extracted:')
                        #         st.text_area('Full text:', debug_text[:3000], height=300)
                        #
                        #         st.subheader('Testing Extraction Patterns:')
                        #
                        #         # Test invoice number patterns
                        #         st.write("**Invoice Number Test:**")
                        #         test_patterns = [
                        #             (r'Invoice\s+No\s*:\s*([A-Z]{2}\d{10})', 'Pattern 1: Invoice No : MH...'),
                        #             (r'Invoice\s+No\s*:\s*(\S+)', 'Pattern 2: Invoice No : (any non-space)'),
                        #             (r'Invoice No.*?([A-Z]{2}\d{10})', 'Pattern 3: Invoice No ... MH...'),
                        #             (r'([A-Z]{2}\d{10})', 'Pattern 4: Just MH... pattern'),
                        #         ]
                        #
                        #         for pattern, desc in test_patterns:
                        #             match = re.search(pattern, debug_text, re.IGNORECASE)
                        #             if match:
                        #                 st.success(f"✅ {desc} -> Found: {match.group(1)}")
                        #             else:
                        #                 st.error(f"❌ {desc} -> Not found")
                        #
                        #         # Test date patterns
                        #         st.write("**Invoice Date Test:**")
                        #         date_patterns = [
                        #             (r'Invoice\s+Date\s*:\s*(\d{1,2}[-/]\w{3}[-/]\d{4})',
                        #              'Pattern 1: Invoice Date : DD-Mon-YYYY'),
                        #             (r'Invoice\s+Date\s*:\s*(\S+)', 'Pattern 2: Invoice Date : (any non-space)'),
                        #             (r'(\d{1,2}[-/]\w{3}[-/]\d{4})', 'Pattern 3: Just DD-Mon-YYYY pattern'),
                        #         ]
                        #
                        #         for pattern, desc in date_patterns:
                        #             match = re.search(pattern, debug_text, re.IGNORECASE)
                        #             if match:
                        #                 st.success(f"✅ {desc} -> Found: {match.group(1)}")
                        #             else:
                        #                 st.error(f"❌ {desc} -> Not found")
                        #
                        #         # Show what was actually extracted
                        #         st.write("**Actually Extracted:**")
                        #         st.json({
                        #             'Invoice No': first_item['invoice_no'],
                        #             'Invoice Date': first_item['invoice_date'],
                        #             'Net Amount': first_item['net_amount'],
                        #             'Receiver': first_item['receiver']
                        #         })

                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            st.download_button('📦 Download all as ZIP', zipbuf, file_name='split_invoices.zip',
                                               mime='application/zip')
                        with col_dl2:
                            st.download_button('📊 Download Excel Summary', excel_buffer,
                                               file_name='invoice_summary.xlsx',
                                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                        # Send emails if enabled
                        # ---- Replace the old `if send_emails:` block with this ----
                        st.markdown('---')
                        st.subheader('📧 Sending Emails (automated) ...')
                        email_results = []

                        progress_bar = st.progress(0.0)
                        status_text = st.empty()

                        total = len(exported)
                        for idx, item in enumerate(exported):
                            try:
                                status_text.text(f"📤 Sending email {idx + 1} of {total} to {TEST_RECIPIENT}...")
                                success, message = mailer.send_invoice_email(
                                    item['path'],
                                    item.get('receiver', '(not found)'),
                                    item.get('invoice_no', '(not found)'),
                                    item.get('invoice_date', '(not found)'),
                                    item.get('net_amount', '(not found)'),
                                    TEST_RECIPIENT,
                                    doc_type
                                )
                            except Exception as e:
                                # Catch unexpected exceptions during send and record as failed
                                success = False
                                message = f"Exception while sending: {e}"

                            email_results.append({
                                'Filename': item.get('filename', ''),
                                'Receiver': item.get('receiver', '(not found)'),
                                'Invoice No': item.get('invoice_no', '(not found)'),
                                'Status': '✅ Sent' if success else '❌ Failed',
                                'Message': message
                            })

                            # Update progress (use float between 0.0 and 1.0)
                            progress_bar.progress((idx + 1) / total)

                        # After email loop completes
                        status_text.text('✅ Email sending complete!')

                        # Prepare files for automatic download (use getvalue() on BytesIO)
                        try:
                            zip_bytes = zipbuf.getvalue()
                            excel_bytes = excel_buffer.getvalue()

                            # Trigger auto-download of ZIP first, then Excel a little after
                            auto_download_multiple([
                                ('split_invoices.zip', zip_bytes, 'application/zip'),
                                ('invoice_summary.xlsx', excel_bytes,
                                 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                            ], delay_ms=700)  # 700ms gap to be safe
                        except Exception as e:
                            st.warning(f"Auto-download failed (browser limits?): {e}")

                        # Display email results
                        email_df = pd.DataFrame(email_results)
                        st.dataframe(email_df, use_container_width=True)

                        success_count = sum(1 for r in email_results if r['Status'] == '✅ Sent')
                        if success_count == total:
                            st.success(f"🎉 Successfully sent all {total} emails to {TEST_RECIPIENT}")
                        else:
                            st.warning(f"⚠️ Sent {success_count} out of {total} emails to {TEST_RECIPIENT}")

                        # Show filenames
                        with st.expander('📄 View all generated filenames'):
                            for item in exported:
                                st.write(
                                    f"- {item['filename']} | Invoice: {item['invoice_no']} | Date: {item['invoice_date']} | Amount: ₹{item['net_amount']}")
                    else:
                        st.info('No files exported.')
                except Exception as e:
                    st.error(f'❌ Error during split: {e}')
                    import traceback

                    st.code(traceback.format_exc())

# ---- Extract info from multiple PDFs ----
st.markdown('---')
st.subheader('3) Extract invoice info from uploaded PDFs')
uploaded_for_extract = st.file_uploader('Upload multiple PDFs to extract info from', type=['pdf'],
                                        accept_multiple_files=True, key='extract')

if st.button('🔍 Extract & Download Excel'):
    if not uploaded_for_extract:
        st.warning('Please upload one or more PDFs to extract.')
    else:
        with st.spinner('Extracting...'):
            rows = []
            tmpdir = tempfile.mkdtemp()
            poppler = get_poppler_path(poppler_path) if ocr_enabled else None
            for up in uploaded_for_extract:
                path = os.path.join(tmpdir, up.name)
                with open(path, 'wb') as f:
                    f.write(up.read())
                text, method = get_pdf_text(path, poppler)
                info = extract_all_invoice_info(text)
                rows.append({
                    'Filename': up.name,
                    'Customer Code': info['customer_code'],
                    'Receiver': info['receiver'],
                    'Invoice No': info['invoice_no'],
                    'Invoice Date': info['invoice_date'],
                    # 'Net Amount': info['net_amount'],
                    'Method': method
                })

            df = pd.DataFrame(rows)

            # Create Excel file
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Invoice Info')
            excel_buffer.seek(0)

            st.success('✅ Extraction complete')
            st.download_button('📊 Download Excel', excel_buffer, file_name='invoice_summary.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            st.dataframe(df, use_container_width=True)

st.markdown('---')

# Information boxes
col_info1, col_info2 = st.columns(2)

# with col_info1:
#     st.info('''
#     **📧 Email Configuration:**
#     - Sender: sakshi@kantascrypt.com
#     - All invoices are sent to the test recipient
#     - Enable "Send emails after splitting" in sidebar
#
#     **Email Format:**
#     - Subject: "Invoice for [Customer Name] - [Invoice No]"
#     - Body includes: Invoice No, Date, Net Amount
#     - Professional business format
#     ''')
#
# with col_info2:
#     st.success('''
#     **✨ Features:**
#     - Auto-detect receiver names
#     - Extract Invoice No, Date & Amount
#     - Generate Excel summary with all details
#     - Auto-send professional emails with attachments
#     - All in one click!
#     ''')

# st.markdown('### 🔧 Troubleshooting & Hints')
# st.markdown('''
# - **Filenames wrong?** Tweak the regex patterns in `extract_receiver_name` and `sanitize_filename`
# - **Email not sending?** Check SMTP credentials and test recipient email
# - **Invoice details not extracted?** Check the regex patterns in extraction functions
# - **OCR not working?** Install Tesseract and Poppler on your system
# - **Running on cloud?** System packages for Poppler and Tesseract must be installed on the server
# ''')

# st.markdown('### 📋 Extraction Patterns')
# with st.expander('View expected invoice format patterns'):
#     st.code('''
# Expected patterns in PDF:
# - Invoice No: MH0125638978
# - Net Amount: 52,500.00 or Net Amount: ₹52,500.00
# - Invoice Date: 28-Oct-2025 or 28-10-2025
# - Receiver Name: [COMPANY NAME] [12345]
#     ''')

# st.markdown('---')
# st.caption(
#     '🚀 Enhanced with Invoice No, Date & Amount extraction + Professional email formatting via Kantascrypt mail server (sakshi@kantascrypt.com)')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import shutil
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
import threading
import time
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # In production, replace with your frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Email Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
TEST_RECIPIENT = os.getenv("TEST_RECIPIENT")
# PDF Configuration
DEFAULT_POPPLER = shutil.which('pdftoppm')  # Returns path or None
if DEFAULT_POPPLER:
    DEFAULT_POPPLER = os.path.dirname(DEFAULT_POPPLER)

DEFAULT_TESSERACT = shutil.which('tesseract')  # Returns path or None

if DEFAULT_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

INVOICE_KEYWORDS = [r"tax\s+invoice", r"invoice\s+no", r"invoice\s+#", r"invoice\b"]
RECEIVER_LABELS = [
    "details of receiver", "receiver", "billed to", "bill to",
    "buyer", "consignee", "ship to", "shipped to"
]

# Global storage for progress tracking
processing_status = {}

if DEFAULT_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT


# ============= Helper Classes =============

class MailSender:
    def __init__(self, smtp_server, smtp_port, username, password, imap_server, imap_port):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port

    def send_invoice_email(self, pdf_path: str, receiver_name: str, invoice_no: str,
                           invoice_date: str, net_amount: str, recipient_email: str,
                           doc_type: str = 'Invoice') -> Tuple[bool, str]:
        try:
            display_name = receiver_name.replace('(not found)', 'Valued Customer')
            formatted_date = self._format_date_for_display(invoice_date)

            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient_email
            msg['Subject'] = f"{doc_type} for {display_name} - {invoice_no}"

            body = f"""Dear {display_name},

Please find attached the {doc_type} ({invoice_no}) dated {formatted_date} for your reference.

Kindly review the details and process the payment as per the agreed terms. Should you have any questions or require clarification, please feel free to get in touch.

Thank you for your continued business.

Best regards,
Kantascrypt Team
{self.username}
            """

            msg.attach(MIMEText(body, 'plain'))

            filename = os.path.basename(pdf_path)
            with open(pdf_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename= {filename}')
                msg.attach(part)

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, recipient_email, text)
            server.quit()

            return True, f"Email sent successfully to {recipient_email}"

        except Exception as e:
            return False, f"Failed to send email: {str(e)}"

    def _format_date_for_display(self, date_str: str) -> str:
        try:
            if date_str == '(not found)':
                return 'N/A'
            date_obj = datetime.strptime(date_str, "%d-%b-%Y")
            day = date_obj.day
            if 4 <= day <= 20 or 24 <= day <= 30:
                suffix = "th"
            else:
                suffix = ["st", "nd", "rd"][day % 10 - 1]
            return date_obj.strftime(f"%d{suffix} %B %Y")
        except:
            return date_str


# ============= PDF Processing Functions =============

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(texts)
    except Exception:
        return ""


def ocr_pdf_pages(path: str, poppler_path: str = None) -> str:
    try:
        images = convert_from_path(path, dpi=200, poppler_path=poppler_path)
        page_texts = [pytesseract.image_to_string(img.convert("L")) for img in images]
        return "\n".join(page_texts)
    except Exception:
        return ""


def get_pdf_text(path: str, poppler_path: str = None) -> Tuple[str, str]:
    text = extract_text_from_pdf(path)
    if text and len(text.strip()) > 50:
        return text, 'direct'
    text = ocr_pdf_pages(path, poppler_path)
    return text, 'ocr'


def extract_invoice_number(text: str) -> str:
    if not text:
        return '(not found)'
    pattern = r'MH\d{8,}'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else '(not found)'


def extract_net_amount(text: str) -> str:
    if not text:
        return '(not found)'
    patterns = [
        r'Net\s+Amount\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)',
        r'(?:Total\s+Amount|Grand\s+Total)\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)',
        r'Amount\s+Payable\s*[:\-]?\s*₹?\s*([\d,]+\.?\d*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return '(not found)'


def extract_invoice_date(text: str) -> str:
    if not text:
        return '(not found)'
    patterns = [
        r'Invoice\s+Date\s*[:\-]?\s*(\d{1,2}[-/]\w{3}[-/]\d{4})',
        r'Date\s*[:\-]\s*(\d{1,2}[-/]\w{3}[-/]\d{4})',
        r'Invoice\s+Date\s*[:\-]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        r'Date\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return '(not found)'


def extract_receiver_name(text: str):
    if not text:
        return None

    patterns = [
        r"(?:details\s+of\s+receiver|billed\s+to|receiver|buyer)[\s\S]{0,200}?name\s*[:\-]\s*([A-Z][A-Z\s&\.,\-\(\)]+\[\s*\d+\s*\])",
        r"name\s*[:\-]\s*([A-Z][A-Z\s&\.,\-\(\)]+\[\s*\d+\s*\])",
        r'([A-Z][A-Z\s&\.,\-\(\)]{10,80}\[\s*\d{5,}\s*\])'
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            receiver = match.group(1).strip()
            receiver = ' '.join(receiver.split())
            if ']' in receiver:
                receiver = receiver[:receiver.rindex(']') + 1].strip()
            return receiver

    return None


def sanitize_filename(name: str):
    if not name:
        return None
    if ']' in name:
        bracket_end = name.rindex(']')
        name = name[:bracket_end + 1]
    name = ' '.join(name.split())
    bracket_match = re.search(r'\[\s*(\d+)\s*\]', name)
    bracket_number = bracket_match.group(1) if bracket_match else None
    if bracket_match:
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


def split_pdf_auto_detect_file(src_path: str, output_folder: str, poppler_path: str = None) -> List[dict]:
    reader = PdfReader(src_path)
    page_texts = [page.extract_text() or "" for page in reader.pages]

    if all(not t.strip() for t in page_texts):
        ocr_text = ocr_pdf_pages(src_path, poppler_path)
        page_texts = ocr_text.split('\n\n')

    page_info = []
    for t in page_texts:
        tl = t.lower()
        has_kw = any(re.search(kw, tl) for kw in INVOICE_KEYWORDS)
        inv_no = extract_invoice_number(t)
        page_info.append({'text': t, 'has_kw': has_kw, 'invoice_no': inv_no})

    # Propagate invoice numbers
    for i in range(1, len(page_info)):
        if page_info[i]['invoice_no'] == '(not found)' and page_info[i - 1]['invoice_no'] != '(not found)':
            page_info[i]['invoice_no'] = page_info[i - 1]['invoice_no']

    for i in range(len(page_info) - 2, -1, -1):
        if page_info[i]['invoice_no'] == '(not found)' and page_info[i + 1]['invoice_no'] != '(not found)':
            page_info[i]['invoice_no'] = page_info[i + 1]['invoice_no']

    starts = [0]
    for i in range(1, len(page_info)):
        prev = page_info[i - 1]
        cur = page_info[i]
        if cur['invoice_no'] != prev['invoice_no']:
            starts.append(i)
            continue
        if prev['invoice_no'] == '(not found)' and cur['invoice_no'] == '(not found)' and cur['has_kw'] and not prev[
            'has_kw']:
            starts.append(i)

    exported = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else len(page_info) - 1

        writer = PdfWriter()
        chunk_text = ""
        for pnum in range(start, end + 1):
            writer.add_page(reader.pages[pnum])
            chunk_text += page_info[pnum]['text']

        info = extract_all_invoice_info(chunk_text)
        safe = sanitize_filename(info['receiver'])

        if safe:
            fname = f"{safe}.pdf"
            path = os.path.join(output_folder, fname)
            counter = 1
            base_no_ext = os.path.splitext(fname)[0]
            ext = os.path.splitext(fname)[1]
            while os.path.exists(path):
                path = os.path.join(output_folder, f"{base_no_ext}_{counter}{ext}")
                counter += 1
        else:
            fname = f"invoice_detected_{start + 1}_to_{end + 1}.pdf"
            path = os.path.join(output_folder, fname)

        with open(path, 'wb') as f:
            writer.write(f)

        exported.append({
            'path': path,
            'filename': os.path.basename(path),
            **info
        })

    return exported


# ============= Background Processing Function =============

def process_pdf_task(task_id, file_path, doc_type):
    """Background task to process PDF and send emails"""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Splitting PDF...',
            'email_results': [],
            'total_invoices': 0,
            'emails_sent': 0
        }

         poppler = None
        if DEFAULT_POPPLER and os.path.isdir(DEFAULT_POPPLER):
            poppler = DEFAULT_POPPLER

        # Split PDF
        exported = split_pdf_auto_detect_file(file_path, tmpdir, poppler)

        processing_status[task_id]['total_invoices'] = len(exported)
        processing_status[task_id]['progress'] = 20
        processing_status[task_id]['message'] = f'Split complete. Found {len(exported)} invoices. Sending emails...'

        # Initialize mailer
        mailer = MailSender(SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, IMAP_SERVER, IMAP_PORT)

        # Send emails
        email_results = []
        for idx, item in enumerate(exported):
            success, message = mailer.send_invoice_email(
                item['path'],
                item.get('receiver', '(not found)'),
                item.get('invoice_no', '(not found)'),
                item.get('invoice_date', '(not found)'),
                item.get('net_amount', '(not found)'),
                TEST_RECIPIENT,
                doc_type
            )

            email_results.append({
                'filename': item.get('filename', ''),
                'receiver': item.get('receiver', '(not found)'),
                'invoice_no': item.get('invoice_no', '(not found)'),
                'status': 'sent' if success else 'failed',
                'message': message
            })

            processing_status[task_id]['emails_sent'] = idx + 1
            processing_status[task_id]['progress'] = 20 + int((idx + 1) / len(exported) * 60)
            processing_status[task_id]['email_results'] = email_results

        processing_status[task_id]['message'] = 'Creating ZIP and Excel files...'
        processing_status[task_id]['progress'] = 85

        # Create Excel
        excel_data = [{
            'Doc Type': doc_type,
            'Filename': item['filename'],
            'Customer Code': item['customer_code'],
            'Receiver Name': item['receiver'],
            'Invoice No': item['invoice_no'],
            'Invoice Date': item['invoice_date']
        } for item in exported]

        df = pd.DataFrame(excel_data)
        excel_path = os.path.join(tmpdir, 'invoice_summary.xlsx')
        df.to_excel(excel_path, index=False, sheet_name='Invoices', engine='openpyxl')

        # Create ZIP
        zip_path = os.path.join(tmpdir, 'split_invoices.zip')
        with zipfile.ZipFile(zip_path, 'w') as z:
            for item in exported:
                z.write(item['path'], arcname=item['filename'])
            z.write(excel_path, arcname='invoice_summary.xlsx')

        processing_status[task_id]['status'] = 'completed'
        processing_status[task_id]['progress'] = 100
        processing_status[task_id]['message'] = 'Processing complete!'
        processing_status[task_id]['zip_path'] = zip_path
        processing_status[task_id]['excel_path'] = excel_path
        processing_status[task_id]['summary'] = excel_data

    except Exception as e:
        processing_status[task_id]['status'] = 'error'
        processing_status[task_id]['message'] = str(e)


# ============= API Endpoints =============

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF allowed'}), 400

    try:
        # Get document type from request or default to 'Invoice'
        doc_type = request.form.get('doc_type', 'Invoice')

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Generate task ID
        task_id = f"task_{timestamp}"

        # Start background processing
        thread = threading.Thread(target=process_pdf_task, args=(task_id, file_path, doc_type))
        thread.start()

        return jsonify({
            'success': True,
            'message': 'PDF uploaded successfully. Processing started.',
            'task_id': task_id
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Get processing status and email progress"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    status = processing_status[task_id]
    return jsonify(status), 200


@app.route('/api/download/<task_id>', methods=['GET'])
def download_zip(task_id):
    """Download the generated ZIP file"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404

    status = processing_status[task_id]

    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed yet'}), 400

    zip_path = status.get('zip_path')
    if not zip_path or not os.path.exists(zip_path):
        return jsonify({'error': 'ZIP file not found'}), 404

    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name='split_invoices.zip'
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'PDF Splitter API is running'}), 200


# ============= Main =============

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

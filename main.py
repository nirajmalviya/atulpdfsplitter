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
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import threading
import time
import socket

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # In production, replace with your frontend URL
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
UPLOAD_FOLDER = "/tmp/uploads"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Email Configuration - Rediffmail Pro
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.rediffmailpro.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "docs.mail@atulsales.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.rediffmailpro.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
TEST_RECIPIENT = os.getenv("TEST_RECIPIENT", "docs.mail@atulsales.com")

# Email retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Customer API Configuration
CUSTOMER_API_URL = "http://192.168.0.102/api/Customer/LoadCustomerDetailsByCode"

# PDF Configuration - Auto-detect system binaries
DEFAULT_POPPLER = shutil.which('pdftoppm')
if DEFAULT_POPPLER:
    DEFAULT_POPPLER = os.path.dirname(DEFAULT_POPPLER)

DEFAULT_TESSERACT = shutil.which('tesseract')
if DEFAULT_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

INVOICE_KEYWORDS = [r"tax\s+invoice", r"invoice\s+no", r"invoice\s+#", r"invoice\b"]
RECEIVER_LABELS = [
    "details of receiver", "receiver", "billed to", "bill to",
    "buyer", "consignee", "ship to", "shipped to"
]

# Global storage for progress tracking
processing_status = {}


# ============= Helper Classes =============

class MailSender:
    def __init__(self, smtp_server, smtp_port, username, password, imap_server, imap_port):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.imap_server = imap_server
        self.imap_port = imap_port
        self._sent_folder_cache = None  # Cache the detected folder

    def send_invoice_email(self, pdf_path: str, receiver_name: str, invoice_no: str,
                           invoice_date: str, net_amount: str, recipient_email: str,
                           doc_type: str = 'Invoice') -> Tuple[bool, str]:
        """Send email with retry logic and detailed error reporting"""

        for attempt in range(MAX_RETRIES):
            try:
                print(f"\n{'=' * 60}")
                print(f"[ATTEMPT {attempt + 1}/{MAX_RETRIES}] Sending email to {recipient_email}")
                print(f"SMTP Server: {self.smtp_server}:{self.smtp_port}")
                print(f"From: {self.username}")
                print(f"{'=' * 60}\n")

                display_name = receiver_name.replace('(not found)', 'Valued Customer')
                formatted_date = self._format_date_for_display(invoice_date)

                msg = MIMEMultipart()
                msg['From'] = self.username
                msg['To'] = recipient_email
                msg['Subject'] = f"{doc_type} for {display_name} - {invoice_no}"
                msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
                msg['Message-ID'] = self._generate_message_id()

                body = f"""Dear {display_name},

Please find attached the {doc_type} ({invoice_no}) dated {formatted_date} for your reference.

Kindly review the details and process the payment as per the agreed terms. Should you have any questions or require clarification, please feel free to get in touch.

Thank you for your continued business.

Best regards,
Atul Sales Team
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

                # Test DNS resolution first
                try:
                    socket.getaddrinfo(self.smtp_server, self.smtp_port)
                    print(f"✓ DNS resolution successful for {self.smtp_server}")
                except socket.gaierror as e:
                    raise Exception(f"DNS resolution failed for {self.smtp_server}: {str(e)}")

                # Send email via SMTP with detailed logging
                print(f"[1/4] Connecting to SMTP server...")
                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30)

                print(f"[2/4] Starting TLS...")
                server.starttls()

                print(f"[3/4] Authenticating...")
                server.login(self.username, self.password)

                print(f"[4/4] Sending message...")
                text = msg.as_string()
                server.sendmail(self.username, recipient_email, text)

                print(f"✓ Email sent successfully via SMTP")
                server.quit()

                # Save to Sent folder via IMAP (MailKit-style approach)
                imap_success = self._append_to_sent_async(msg)
                if imap_success:
                    print(f"✓ Email saved to Sent folder")
                else:
                    print(f"⚠ Warning: Email sent but not saved to Sent folder")

                return True, f"Email sent successfully to {recipient_email}"

            except smtplib.SMTPAuthenticationError as e:
                error_msg = f"Authentication failed. Please check email/password: {str(e)}"
                print(f"✗ {error_msg}")
                return False, error_msg  # Don't retry auth errors

            except smtplib.SMTPServerDisconnected as e:
                error_msg = f"Server disconnected: {str(e)}"
                print(f"✗ {error_msg}")

                if attempt < MAX_RETRIES - 1:
                    print(f"⟳ Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, f"Failed after {MAX_RETRIES} attempts: {error_msg}"

            except smtplib.SMTPException as e:
                error_msg = f"SMTP error: {str(e)}"
                print(f"✗ {error_msg}")

                if attempt < MAX_RETRIES - 1:
                    print(f"⟳ Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, f"Failed after {MAX_RETRIES} attempts: {error_msg}"

            except socket.timeout as e:
                error_msg = f"Connection timeout: {str(e)}"
                print(f"✗ {error_msg}")

                if attempt < MAX_RETRIES - 1:
                    print(f"⟳ Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, f"Failed after {MAX_RETRIES} attempts: {error_msg}"

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                print(f"✗ {error_msg}")
                import traceback
                print(traceback.format_exc())

                if attempt < MAX_RETRIES - 1:
                    print(f"⟳ Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue
                return False, f"Failed after {MAX_RETRIES} attempts: {error_msg}"

        return False, f"Failed to send email after {MAX_RETRIES} attempts"

    def _generate_message_id(self) -> str:
        """Generate a unique Message-ID for the email"""
        import random
        import string
        domain = self.username.split('@')[1] if '@' in self.username else 'localhost'
        unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=20))
        timestamp = int(time.time())
        return f"<{timestamp}.{unique_id}@{domain}>"

    def _detect_sent_folder_async(self, imap) -> str:
        """
        Detect the Sent folder - MailKit style approach
        Returns the folder name if found, None otherwise
        """

        # If we already detected it, use cached value
        if self._sent_folder_cache:
            print(f"[IMAP] Using cached Sent folder: '{self._sent_folder_cache}'")
            return self._sent_folder_cache

        # Comprehensive list of Sent folder patterns
        sent_folder_candidates = [
            'Sent',
            'Sent Items',
            'Sent Messages',
            'Sent Mail',
            'INBOX.Sent',
            'INBOX/Sent',
            'INBOX.Sent Items',
            'sent',
            'SENT',
        ]

        print(f"[IMAP] Trying to detect Sent folder from {len(sent_folder_candidates)} candidates...")

        for idx, folder_name in enumerate(sent_folder_candidates):
            try:
                # Try to select the folder (MailKit GetFolder equivalent)
                status, data = imap.select(folder_name, readonly=True)
                if status == 'OK':
                    # Successfully found the folder
                    try:
                        imap.close()  # Close the folder after testing
                    except:
                        pass

                    # Cache it for future use
                    self._sent_folder_cache = folder_name
                    print(f"[IMAP] ✓ Detected Sent folder: '{folder_name}' (attempt {idx + 1})")
                    return folder_name
            except Exception:
                # This folder doesn't exist or can't be accessed
                continue

        print(f"[IMAP] ⚠ Could not auto-detect Sent folder")
        return None

    def _get_or_create_sent_folder(self, imap) -> str:
        """
        Get or create the Sent folder - mimics MailKit's approach:
        1. Try to detect existing folder
        2. Try to create 'Sent' folder
        3. Try common fallbacks
        """

        # Step 1: Try to detect existing folder
        sent_folder = self._detect_sent_folder_async(imap)
        if sent_folder:
            return sent_folder

        # Step 2: List all folders to help with debugging
        print("[IMAP] Listing available folders for debugging:")
        try:
            status, folder_list = imap.list()
            if status == 'OK':
                for folder_info in folder_list[:20]:  # Show first 20
                    folder_str = folder_info.decode('utf-8') if isinstance(folder_info, bytes) else str(folder_info)
                    print(f"[IMAP]   {folder_str}")
        except Exception as e:
            print(f"[IMAP] Could not list folders: {e}")

        # Step 3: Try to create 'Sent' folder (MailKit CreateAsync equivalent)
        print("[IMAP] Attempting to create 'Sent' folder...")
        try:
            status, data = imap.create('Sent')
            if status == 'OK':
                print("[IMAP] ✓ Created 'Sent' folder")
                self._sent_folder_cache = 'Sent'
                return 'Sent'
            else:
                print(f"[IMAP] Could not create 'Sent' folder: {status}")
        except Exception as e:
            print(f"[IMAP] Exception creating 'Sent' folder: {e}")

        # Step 4: Try 'Sent Items' as fallback
        print("[IMAP] Attempting to create 'Sent Items' folder...")
        try:
            status, data = imap.create('Sent Items')
            if status == 'OK':
                print("[IMAP] ✓ Created 'Sent Items' folder")
                self._sent_folder_cache = 'Sent Items'
                return 'Sent Items'
        except Exception as e:
            print(f"[IMAP] Exception creating 'Sent Items' folder: {e}")

        print("[IMAP] ❌ Failed to get or create Sent folder")
        return None

    def _append_to_sent_async(self, msg: MIMEMultipart) -> bool:
        """
        Append message to Sent folder - MailKit-style approach
        Returns True on success, False on failure

        This closely follows the C# MailKit implementation:
        1. Connect via SSL
        2. Authenticate
        3. Detect/Create Sent folder
        4. Open folder in ReadWrite mode (simulated via non-readonly select)
        5. Append with Seen flag
        """
        import imaplib
        import email.utils

        imap = None

        try:
            # Step 1: Connect IMAP over SSL (MailKit: ConnectAsync with SslOnConnect)
            print(f"[IMAP] Connecting to {self.imap_server}:{self.imap_port} with SSL...")
            imap = imaplib.IMAP4_SSL(self.imap_server, self.imap_port, timeout=30)
            print("[IMAP] ✓ Connected via SSL")

            # Step 2: Authenticate (MailKit: AuthenticateAsync)
            print(f"[IMAP] Authenticating as {self.username}...")
            result = imap.login(self.username, self.password)
            print(f"[IMAP] ✓ Authenticated: {result}")

            # Step 3: Get or create Sent folder (MailKit: GetFolder/CreateAsync)
            sent_folder = self._get_or_create_sent_folder(imap)

            if not sent_folder:
                print("[IMAP] ❌ Cannot proceed without a Sent folder")
                return False

            print(f"[IMAP] Using Sent folder: '{sent_folder}'")

            # Step 4: Open folder in ReadWrite mode (MailKit: OpenAsync with ReadWrite)
            # In imaplib, we just select without readonly=True
            print(f"[IMAP] Opening folder '{sent_folder}' in ReadWrite mode...")
            status, data = imap.select(sent_folder, readonly=False)

            if status != 'OK':
                print(f"[IMAP] ❌ Failed to open folder: {status}")
                return False

            print(f"[IMAP] ✓ Folder opened: {data}")

            # Step 5: Prepare the message
            if 'Date' not in msg:
                msg['Date'] = email.utils.formatdate(localtime=True)

            if 'Message-ID' not in msg:
                msg['Message-ID'] = self._generate_message_id()

            # Convert message to bytes
            msg_string = msg.as_string()
            msg_bytes = msg_string.encode('utf-8')

            # Get current time for IMAP internal date
            import time as time_module
            current_time = time_module.time()
            internal_date = imaplib.Time2Internaldate(current_time)

            # Step 6: Append message with \Seen flag (MailKit: AppendAsync with MessageFlags.Seen)
            print(f"[IMAP] Appending message ({len(msg_bytes)} bytes) with \\Seen flag...")

            result = imap.append(
                sent_folder,
                '\\Seen',  # MessageFlags.Seen equivalent
                internal_date,
                msg_bytes
            )

            if result[0] == 'OK':
                print(f"[IMAP] ✅ Message successfully appended to '{sent_folder}'")
                print(f"[IMAP] Server response: {result}")

                # Close the folder after appending (MailKit does this in Dispose)
                try:
                    imap.close()
                    print(f"[IMAP] ✓ Folder closed")
                except:
                    pass

                return True
            else:
                print(f"[IMAP] ❌ APPEND failed: {result}")
                return False

        except imaplib.IMAP4.abort as e:
            print(f"[IMAP] ❌ Connection aborted: {str(e)}")
            return False

        except imaplib.IMAP4.error as e:
            print(f"[IMAP] ❌ IMAP protocol error: {str(e)}")
            return False

        except socket.timeout as e:
            print(f"[IMAP] ❌ Connection timeout: {str(e)}")
            return False

        except Exception as e:
            print(f"[IMAP] ❌ Unexpected error: {str(e)}")
            print(f"[IMAP] Error type: {type(e).__name__}")
            import traceback
            print(f"[IMAP] Traceback:\n{traceback.format_exc()}")
            return False

        finally:
            # Disconnect (MailKit: DisconnectAsync)
            if imap:
                try:
                    # Properly disconnect with expunge
                    imap.logout()
                    print("[IMAP] ✓ Disconnected")
                except Exception as logout_error:
                    print(f"[IMAP] Warning: Disconnect failed: {logout_error}")

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


# ============= Customer API Functions =============

def fetch_customer_details_from_api(customer_codes: List[str]) -> Dict[str, dict]:
    """
    Fetch customer details from the API

    Args:
        customer_codes: List of customer codes to fetch

    Returns:
        Dictionary mapping customer codes to customer details
    """
    try:
        # Filter out invalid customer codes
        valid_codes = [code for code in customer_codes if code != '(not found)']

        if not valid_codes:
            return {}

        # Prepare request payload
        payload = {
            "customerdetails": [
                {"customercode": int(code)} for code in valid_codes
            ]
        }

        # Make API request
        response = requests.post(
            CUSTOMER_API_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )

        response.raise_for_status()

        # Parse response
        data = response.json()

        if not data.get('status', False):
            print(f"API returned error: {data.get('message', 'Unknown error')}")
            return {}

        # Map customer codes to details
        customer_map = {}
        for customer in data.get('data', []):
            customer_code = str(customer.get('customercode', ''))
            customer_map[customer_code] = {
                'customermasterid': customer.get('customermasterid', ''),
                'customername': customer.get('customername', ''),
                'emailid': customer.get('emailid', ''),
                'customercode': customer.get('customercode', '')
            }

        return customer_map

    except requests.exceptions.RequestException as e:
        print(f"Error fetching customer details: {str(e)}")
        return {}
    except Exception as e:
        print(f"Unexpected error in fetch_customer_details_from_api: {str(e)}")
        return {}


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

        tmpdir = tempfile.mkdtemp()

        # Check if DEFAULT_POPPLER exists and is a directory
        poppler = None
        if DEFAULT_POPPLER and os.path.isdir(DEFAULT_POPPLER):
            poppler = DEFAULT_POPPLER

        # Split PDF
        exported = split_pdf_auto_detect_file(file_path, tmpdir, poppler)

        processing_status[task_id]['total_invoices'] = len(exported)
        processing_status[task_id]['progress'] = 15
        processing_status[task_id][
            'message'] = f'Split complete. Found {len(exported)} invoices. Fetching customer details...'

        # Extract all customer codes
        customer_codes = [item['customer_code'] for item in exported if item['customer_code'] != '(not found)']

        # Fetch customer details from API
        customer_details_map = {}
        if customer_codes:
            processing_status[task_id]['message'] = f'Fetching customer details for {len(customer_codes)} customers...'
            customer_details_map = fetch_customer_details_from_api(customer_codes)
            processing_status[task_id]['progress'] = 25
            processing_status[task_id][
                'message'] = f'Customer details fetched. Found {len(customer_details_map)} customers. Sending emails...'
        else:
            processing_status[task_id]['progress'] = 25
            processing_status[task_id]['message'] = 'No valid customer codes found. Proceeding with test emails...'

        # Initialize mailer
        mailer = MailSender(SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, IMAP_SERVER, IMAP_PORT)

        # Send emails
        email_results = []
        for idx, item in enumerate(exported):
            customer_code = item.get('customer_code', '(not found)')

            # Get customer details from API or use defaults
            if customer_code in customer_details_map:
                customer_info = customer_details_map[customer_code]
                recipient_email = customer_info.get('emailid', TEST_RECIPIENT)
                customer_name = customer_info.get('customername', item.get('receiver', '(not found)'))
            else:
                # Fallback to test recipient if customer not found in API
                recipient_email = TEST_RECIPIENT
                customer_name = item.get('receiver', '(not found)')

            # Send email
            success, message = mailer.send_invoice_email(
                item['path'],
                customer_name,
                item.get('invoice_no', '(not found)'),
                item.get('invoice_date', '(not found)'),
                item.get('net_amount', '(not found)'),
                recipient_email,
                doc_type
            )

            email_results.append({
                'filename': item.get('filename', ''),
                'receiver': customer_name,
                'customer_code': customer_code,
                'invoice_no': item.get('invoice_no', '(not found)'),
                'recipient_email': recipient_email,
                'status': 'sent' if success else 'failed',
                'message': message
            })

            processing_status[task_id]['emails_sent'] = idx + 1
            processing_status[task_id]['progress'] = 25 + int((idx + 1) / len(exported) * 55)
            processing_status[task_id]['email_results'] = email_results

        processing_status[task_id]['message'] = 'Creating ZIP and Excel files...'
        processing_status[task_id]['progress'] = 85

        # Create Excel with enhanced data
        excel_data = []
        for idx, item in enumerate(exported):
            customer_code = item['customer_code']

            # Get customer info from API if available
            if customer_code in customer_details_map:
                customer_info = customer_details_map[customer_code]
                excel_data.append({
                    'Doc Type': doc_type,
                    'Filename': item['filename'],
                    'Customer Code': customer_code,
                    'Customer Name (API)': customer_info.get('customername', ''),
                    'Receiver Name (PDF)': item['receiver'],
                    'Email': customer_info.get('emailid', ''),
                    'Invoice No': item['invoice_no'],
                    'Invoice Date': item['invoice_date'],
                    'Net Amount': item.get('net_amount', '(not found)'),
                    'Email Status': email_results[idx]['status']
                })
            else:
                excel_data.append({
                    'Doc Type': doc_type,
                    'Filename': item['filename'],
                    'Customer Code': customer_code,
                    'Customer Name (API)': 'Not Found in API',
                    'Receiver Name (PDF)': item['receiver'],
                    'Email': 'Not Available',
                    'Invoice No': item['invoice_no'],
                    'Invoice Date': item['invoice_date'],
                    'Net Amount': item.get('net_amount', '(not found)'),
                    'Email Status': email_results[idx]['status']
                })

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
        processing_status[task_id]['customers_found_in_api'] = len(customer_details_map)
        processing_status[task_id]['total_customers'] = len(customer_codes)

    except Exception as e:
        processing_status[task_id]['status'] = 'error'
        processing_status[task_id]['message'] = str(e)
        import traceback
        processing_status[task_id]['traceback'] = traceback.format_exc()


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


@app.route('/api/test-email', methods=['POST'])
def test_email():
    """Test endpoint to verify email configuration"""
    try:
        data = request.json or {}
        recipient = data.get('recipient', TEST_RECIPIENT)

        print(f"\n{'=' * 60}")
        print(f"EMAIL CONFIGURATION TEST")
        print(f"{'=' * 60}")
        print(f"SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
        print(f"IMAP Server: {IMAP_SERVER}:{IMAP_PORT}")
        print(f"Sender: {SENDER_EMAIL}")
        print(f"Recipient: {recipient}")
        print(f"Password configured: {'Yes' if SENDER_PASSWORD else 'No'}")
        print(f"{'=' * 60}\n")

        if not SENDER_PASSWORD:
            return jsonify({
                'error': 'Email password not configured. Please set SENDER_PASSWORD environment variable.'
            }), 400

        mailer = MailSender(SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, IMAP_SERVER, IMAP_PORT)

        # Create a test PDF
        tmpdir = tempfile.mkdtemp()
        test_pdf_path = os.path.join(tmpdir, 'test_invoice.pdf')

        # Create a simple PDF for testing
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(test_pdf_path)
        c.drawString(100, 750, "TEST INVOICE")
        c.drawString(100, 700, "This is a test email from the PDF Splitter system")
        c.save()

        success, message = mailer.send_invoice_email(
            test_pdf_path,
            "Test Customer",
            "TEST-001",
            "01-Jan-2025",
            "1000.00",
            recipient,
            "Test Invoice"
        )

        # Cleanup
        try:
            os.remove(test_pdf_path)
            os.rmdir(tmpdir)
        except:
            pass

        return jsonify({
            'success': success,
            'message': message,
            'config': {
                'smtp_server': SMTP_SERVER,
                'smtp_port': SMTP_PORT,
                'imap_server': IMAP_SERVER,
                'imap_port': IMAP_PORT,
                'sender': SENDER_EMAIL,
                'recipient': recipient
            }
        }), 200 if success else 500

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/test-customer-api', methods=['POST'])
def test_customer_api():
    """Test endpoint to verify customer API integration"""
    try:
        data = request.json
        customer_codes = data.get('customer_codes', [])

        if not customer_codes:
            return jsonify({'error': 'Please provide customer_codes array'}), 400

        customer_details = fetch_customer_details_from_api(customer_codes)

        return jsonify({
            'success': True,
            'customer_codes_requested': customer_codes,
            'customers_found': len(customer_details),
            'customer_details': customer_details
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============= Main =============

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("PDF INVOICE SPLITTER - MAILKIT-STYLE IMAP")
    print("=" * 60)
    print(f"SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
    print(f"IMAP Server: {IMAP_SERVER}:{IMAP_PORT}")
    print(f"Sender Email: {SENDER_EMAIL}")
    print(f"Password Configured: {'Yes' if SENDER_PASSWORD else 'NO - PLEASE SET SENDER_PASSWORD!'}")
    print(f"Test Recipient: {TEST_RECIPIENT}")
    print("=" * 60 + "\n")

    if not SENDER_PASSWORD:
        print("⚠️  WARNING: SENDER_PASSWORD environment variable is not set!")
        print("⚠️  Email sending will fail without a password.")
        print("⚠️  Please set it before running the application.\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

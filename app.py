import os
import cv2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from plagiarism_detector import check_plagiarism_online
import base64
import requests
import logging
import tempfile
from io import BytesIO
from datetime import datetime
from urllib.parse import urlparse
from flask import Flask, request, render_template, send_file, url_for, flash
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from skimage.metrics import structural_similarity as ssim
import fitz  # PyMuPDF

# --- Config ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
GOOGLE_API_KEY = 'AIzaSyA5VzwiG_fLFhGKrrXxVrGm1o6BVXtgxyo'
VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_API_KEY}"
MAX_WORKERS = 2
BLACKLISTED_DOMAINS = {'landacbio.ipn.mx'}

# --- App Setup ---
app = Flask(__name__)
app.secret_key = 'change-me'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ORB = cv2.ORB_create(nfeatures=3000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def sanitize_filename(fname):
    return os.path.basename(fname)


def get_web_urls(image_path):
    """
    Fetch matching image URLs via Google Vision API's WEB_DETECTION.
    Uses fullMatchingImages, partialMatchingImages, visuallySimilarImages.
    """
    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        'requests': [{
            'image': {'content': img_b64},
            'features': [{'type': 'WEB_DETECTION'}]
        }]
    }
    try:
        resp = requests.post(VISION_URL, json=payload, timeout=10)
        resp.raise_for_status()
        web = resp.json().get('responses', [{}])[0].get('webDetection', {})
        items = []
        items.extend(web.get('fullMatchingImages', []))
        items.extend(web.get('partialMatchingImages', []))
        items.extend(web.get('visuallySimilarImages', []))
        urls = [item.get('url') for item in items if item.get('url')]
        top_urls = urls[:5]
        logger.info(f"[Vision API] Found {len(top_urls)} image URLs for {image_path}")
        return top_urls
    except Exception as e:
        logger.warning(f"[Vision API ERROR] {e}")
        return []


def download_image(url):
    domain = urlparse(url).netloc
    if domain in BLACKLISTED_DOMAINS:
        logger.info(f"[Skipped - Blacklisted Domain] {url}")
        return None, url
    try:
        r = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        content_type = r.headers.get('Content-Type', '')
        if 'image' not in content_type:
            logger.info(f"[Skipped - Not Image] {url} (Content-Type: {content_type})")
            return None, url
        img = Image.open(BytesIO(r.content)).convert('RGB')
        fname = f"comp_{sanitize_filename(urlparse(url).path)}"
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fname += '.jpg'
        path = os.path.join(UPLOAD_FOLDER, fname)
        img.save(path)
        logger.info(f"[Downloaded] {url} -> {path}")
        return path, url
    except Exception as e:
        logger.warning(f"[Download Failed] {url}: {e}")
        return None, url


def compare_images(p1, p2):
    img1, img2 = cv2.imread(p1), cv2.imread(p2)
    if img1 is None or img2 is None:
        logger.warning(f"[Image Read Failed] {p1 if img1 is None else p2}")
        return 0.0, 0.0
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    g1 = cv2.cvtColor(cv2.resize(img1, (w, h)), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, (w, h)), cv2.COLOR_BGR2GRAY)
    s, _ = ssim(g1, g2, full=True)
    k1, d1 = ORB.detectAndCompute(g1, None)
    k2, d2 = ORB.detectAndCompute(g2, None)
    o = 0.0
    if d1 is not None and d2 is not None and k1 and k2:
        matches = BF.match(d1, d2)
        o = len(matches) / max(len(k1), len(k2))
    logger.debug(f"[Compare] SSIM: {s:.3f}, ORB: {o:.3f} | {p1} vs {p2}")
    return s, o


def calculate_plagiarism(ssim_list, orb_list, total_urls):
    logger.debug(f"[Calculate] SSIMs: {ssim_list}, ORBs: {orb_list}, URLs: {total_urls}")
    if not ssim_list or not orb_list:
        return {'final': 0.0}
    m_s, m_o = max(ssim_list), max(orb_list)
    content_sim = 0.6 * m_s + 0.4 * m_o
    web_pres = min(total_urls / 5.0, 1.0)
    final_pct = min((0.7 * content_sim + 0.3 * web_pres) * 100, 100)
    logger.info(
        f"[Plagiarism Score] SSIM: {m_s:.3f}, ORB: {m_o:.3f}, "
        f"ContentSim: {content_sim:.3f}, WebPres: {web_pres:.3f}, Final: {final_pct:.1f}%"
    )
    return {'final': round(final_pct, 1)}


def extract_images_from_pdf(pdf_path):
    images = []
    doc = fitz.open(pdf_path)
    for i in range(len(doc)):
        for img in doc.load_page(i).get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base['ext']
            fname = f"page_{i}_{xref}.{ext}"
            path = os.path.join(UPLOAD_FOLDER, fname)
            with open(path, 'wb') as f:
                f.write(base['image'])
            images.append(path)
    return images


@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'image_results': [],
        'total': 0,
        'plag_count': 0,
        'overall_score': 0.0
    }

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash("Invalid file type.", 'danger')
            return render_template('index.html', **context)

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = sanitize_filename(file.filename)
        saved = os.path.join(UPLOAD_FOLDER, filename)
        file.save(saved)

        # <-- FIXED: use .endswith (lowercase) -->
        paths = extract_images_from_pdf(saved) if filename.lower().endswith('.pdf') else [saved]

        context['total'] = len(paths)
        max_score = 0.0

        for p in paths:
            logger.info(f"[Processing] {p}")
            urls = get_web_urls(p)

            with ThreadPoolExecutor(MAX_WORKERS) as pool:
                results = pool.map(lambda u: download_image(u), urls)

            downloaded = [(pp, u) for pp, u in results if pp]
            s_list, o_list, details = [], [], []

            for pp, u in downloaded:
                s, o = compare_images(p, pp)
                s_list.append(s)
                o_list.append(o)
                details.append({'url': u, 'ssim': round(s, 3), 'orb': round(o, 3)})

            score = calculate_plagiarism(s_list, o_list, len(urls))['final']
            logger.info(f"[Result] {score}% for {os.path.basename(p)}")

            context['image_results'].append({
                'path': url_for('static', filename=f'uploads/{os.path.basename(p)}'),
                'score': score,
                'details': details
            })

            if score >= 50:
                context['plag_count'] += 1
            max_score = max(max_score, score)

        context['overall_score'] = max_score

        # cleanup
        try:
            os.remove(saved)
        except Exception:
            pass

    return render_template('index.html', **context)


@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        results = request.json.get('results', [])
        fd, out = tempfile.mkstemp(suffix='.pdf')

        doc = SimpleDocTemplate(
            out, pagesize=letter,
            leftMargin=0.5*inch, rightMargin=0.5*inch,
            topMargin=0.5*inch, bottomMargin=0.5*inch
        )
        styles = getSampleStyleSheet()
        flow = [
            Paragraph(
                "Plagiarism Detection Report",
                ParagraphStyle(
                    'Title', parent=styles['Title'],
                    fontSize=18, spaceAfter=14,
                    textColor=colors.HexColor('#4361ee')
                )
            )
        ]

        meta = [
            ["Report Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ["Total Images:", str(len(results))],
            ["Plagiarized Images:", str(len([r for r in results if r['score'] >= 50]))]
        ]
        tbl = Table(meta, colWidths=[2*inch, 4*inch])
        tbl.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 10)]))
        flow.extend([tbl, Spacer(1, 24)])

        for img in results:
            try:
                ip = os.path.join('static', img['path'].replace('/static/', ''))
                pil = Image.open(ip)
                pil.thumbnail((400, 400))
                buf = BytesIO()
                pil.save(buf, 'PNG')
                flow.append(RLImage(buf, width=3*inch, height=3*inch))
            except Exception as e:
                flow.append(Paragraph(f"Image unavailable: {e}", styles['Normal']))

            info = [
                ["Match Score:", f"{img['score']}%"],
                ["Detection Method:", "Visual + Feature Matching"]
            ]
            src = [["Source URL", "Scores"]] + [
                [d['url'], f"SSIM: {d['ssim']}, ORB: {d['orb']}"] 
                for d in img['details']
            ]
            flow.extend([
                Table(info, colWidths=[1.5*inch, 4.5*inch]),
                Spacer(1, 12),
                Paragraph("Matched Sources:", styles['Heading3']),
                Table(src, colWidths=[4*inch, 2*inch]),
                Spacer(1, 36)
            ])

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.drawString(
                0.5*inch, 0.5*inch,
                f"Page {doc.page} • Generated by Plagiarism Checker"
            )
            canvas.restoreState()

        doc.build(flow, onFirstPage=footer, onLaterPages=footer)
        return send_file(
            out, as_attachment=True,
            download_name='plagiarism-report.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        return str(e), 500

    finally:
        os.close(fd)
UPLOAD_FOLDER = 'stored_pdfs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/check_online', methods=['POST'])
def check_online():
    """Performs online plagiarism detection."""
    text = request.form.get('text', '')
    upload_file = request.files.get('file')

    # If a file is uploaded, extract text from the PDF
    if upload_file:
        filename = secure_filename(upload_file.filename)
        upload_file.save(filename)  # Save temporarily
        extracted_text = extract_text_from_pdf(filename)  #
        os.remove(filename)  # Remove temporary file
    else:
        extracted_text = text 
    # Check plagiarism using online sources
    online_results = check_plagiarism_online(extracted_text)
    online_results['extracted_text'] = extracted_text 
    return jsonify(online_results)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ''.join([page.extract_text() or '' for page in reader.pages])
        return text
    except Exception as e:
        return ''

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, send_file, jsonify
import cv2, numpy as np, io, img2pdf, os, json, time, subprocess, base64
from PIL import Image
from pdf2docx import Converter
from werkzeug.utils import secure_filename
import google.generativeai as genai

try:
    import fitz  # PyMuPDF for PDF compression
except ImportError:
    fitz = None

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp_storage')
os.makedirs(TEMP_DIR, exist_ok=True)

# --- AI INITIALIZATION ---
print("✨ Connecting to Gemini Context Engine...")
# IMPORTANT: Put your actual Gemini API Key in environment variable GEMINI_API_KEY.
# On Render set this in Web Service Environment variables.
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))
llm_model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ All Systems Ready!")

# --- UTILITIES ---
def cleanup_temp_dir(max_age_seconds=3600):
    """Deletes files in TEMP_DIR older than 1 hour to prevent server OOM/Storage exhaustion."""
    now = time.time()
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            if os.stat(file_path).st_mtime < now - max_age_seconds:
                try: os.remove(file_path)
                except Exception: pass

# --- ROUTES ---
@app.route('/')
def index(): 
    cleanup_temp_dir()
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        file = request.files.get('image')
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        
        ratio = w / 500.0
        small_img = cv2.resize(img, (500, int(h / ratio)))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        
        document_contour = None
        
        # 1. THE MULTI-PASS SYSTEM
        # We test 3 different vision filters. If a dark shadow ruins one, the next one will catch it.
        filters = [
            cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 75, 200),   # Pass 1: Standard Document
            cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 30, 100),   # Pass 2: High Sensitivity (for low contrast)
            cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 10, 50)   # Pass 3: Extreme Blur (for heavy shadows/wrinkles)
        ]
        
        for edge_filter in filters:
            # Thicken the lines slightly to close small gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edge_filter, cv2.MORPH_CLOSE, kernel)
            
            cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
            
            for c in cnts:
                peri = cv2.arcLength(c, True)
                
                # 2. THE EPSILON LOOP
                # We test 3 levels of mathematical smoothing to force the shape into exactly 4 corners
                for eps in [0.02, 0.04, 0.06]:
                    approx = cv2.approxPolyDP(c, eps * peri, True)
                    
                    # Is it 4 points? Is it at least 15% of the image size?
                    if len(approx) == 4 and cv2.contourArea(approx) > (500 * (h/ratio) * 0.15):
                        document_contour = approx.reshape(-1, 2)
                        break # Found it!
                        
                if document_contour is not None:
                    break # Stop checking other shapes
                    
            if document_contour is not None:
                break # Stop running filters!
                
        # 3. Snap the dots to the found document
        if document_contour is not None:
            pts = document_contour
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            
            # Organize corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            rect = [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]]
            return jsonify({"points": [[int(p[0]*ratio), int(p[1]*ratio)] for p in rect]})
            
        # 4. Fallback only if the image is literally a blank wall
        pad_x, pad_y = int(w * 0.1), int(h * 0.1)
        return jsonify({"points": [[pad_x, pad_y], [w - pad_x, pad_y], [w - pad_x, h - pad_y], [pad_x, h - pad_y]]})
        
    except Exception as e:
        print(f"OpenCV Error: {e}")
        return jsonify({"points": [[0,0], [w,0], [w,h], [0,h]]})

@app.route('/api/process', methods=['POST'])
def process():
    file = request.files.get('image')
    pts = np.array(json.loads(request.form.get('points')), dtype="float32")
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    _, buf = cv2.imencode(".png", enhanced)
    return send_file(io.BytesIO(buf), mimetype='image/png')

@app.route('/api/enhance_image', methods=['POST'])
def enhance_image():
    file = request.files.get('image')
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    _, buf = cv2.imencode(".png", enhanced)
    return send_file(io.BytesIO(buf), mimetype='image/png')

@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    try:
        file = request.files.get('image')
        smart_format = request.form.get('smart_format') == 'true'

        if not file:
            return jsonify({"state": "FAILURE", "error": "No image provided"}), 400

        # Read image bytes
        image_bytes = file.read()

        # Use Gemini to extract text
        prompt = "Extract all text from this image exactly, preserving formatting. Return plain text only, no extra commentary."
        response = llm_model.generate_content(
            prompt,
            images=[{
                'mimeType': 'image/png',  # Assume PNG, but Gemini handles various
                'content': base64.b64encode(image_bytes).decode('utf-8')
            }]
        )

        raw_text = (response.text or "").strip()

        if not raw_text:
            return jsonify({"state": "SUCCESS", "result": {"text": "No clear text or handwriting could be detected."}})

        if smart_format:
            format_prompt = f"Format this extracted text clearly and fix obvious typos. Return only text.\n\n{raw_text}"
            formatted_response = llm_model.generate_content(format_prompt)
            final_text = (formatted_response.text or "").strip()
        else:
            final_text = raw_text

        return jsonify({"state": "SUCCESS", "result": {"text": final_text}})

    except Exception as e:
        return jsonify({"state": "FAILURE", "error": str(e)})



@app.route('/api/pdf', methods=['POST'])
def pdf_api():
    files = request.files.getlist("images")
    pdf_bytes = img2pdf.convert([f.read() for f in files])
    return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf')

@app.route('/api/pdf_to_word', methods=['POST'])
def pdf_to_word():
    cleanup_temp_dir()
    file = request.files.get('file')
    filename = secure_filename(file.filename)
    p_in = os.path.join(TEMP_DIR, f"in_{filename}")
    p_out = os.path.join(TEMP_DIR, f"out_{filename}.docx")
    file.save(p_in)
    try:
        cv = Converter(p_in); cv.convert(p_out, multi_processing=False); cv.close()
        return send_file(p_out, as_attachment=True)
    except Exception as e:
        return str(e), 500

@app.route('/api/word_to_pdf', methods=['POST'])
def word_to_pdf():
    cleanup_temp_dir()
    file = request.files.get('file')
    filename = secure_filename(file.filename)
    p_in = os.path.abspath(os.path.join(TEMP_DIR, f"in_{filename}"))
    file.save(p_in)
    
    try:
        subprocess.run([
            'soffice', '--headless', '--convert-to', 'pdf', p_in, '--outdir', TEMP_DIR
        ], check=True)
        p_out = os.path.abspath(os.path.join(TEMP_DIR, f"in_{filename.rsplit('.', 1)[0]}.pdf"))
        return send_file(p_out, as_attachment=True)
    except FileNotFoundError:
        return "LibreOffice not found. Please install LibreOffice on the server.", 500
    except Exception as e:
        return f"Conversion Error: {str(e)}", 500

@app.route('/api/compress', methods=['POST'])
def compress_file():
    cleanup_temp_dir()
    file = request.files.get('file')
    level = request.form.get('level', 'medium') 
    if not file: return "No file uploaded.", 400
    
    filename = secure_filename(file.filename.lower())
    ext = filename.split('.')[-1]
    
    try:
        file_bytes = file.read()
        original_size = len(file_bytes)
        
        if ext in ['jpg', 'jpeg', 'png']:
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode in ("RGBA", "P"): img = img.convert("RGB")
            
            if level == 'small': target_size = original_size * 0.15
            elif level == 'medium': target_size = original_size * 0.50
            else: target_size = original_size * 0.85
            
            scale = 1.0 if level == 'large' else (0.7 if level == 'medium' else 0.4)
            q = 85 if level == 'large' else (50 if level == 'medium' else 20)
            
            new_w, new_h = int(img.width * scale), int(img.height * scale)
            if new_w < 10 or new_h < 10: new_w, new_h = img.width, img.height
            
            curr_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            out_io = io.BytesIO()
            curr_img.save(out_io, format="JPEG", quality=q, optimize=True)
            
            while out_io.tell() > target_size and q > 10:
                q -= 10
                out_io = io.BytesIO()
                curr_img.save(out_io, format="JPEG", quality=q, optimize=True)
                
            out_io.seek(0)
            return send_file(out_io, mimetype='image/jpeg', as_attachment=True, download_name=f"Compressed_{filename.rsplit('.', 1)[0]}.jpg")
            
        elif ext == 'pdf':
            if not fitz: return "PyMuPDF not installed.", 500
            in_path = os.path.join(TEMP_DIR, f"temp_in_{filename}")
            out_path = os.path.join(TEMP_DIR, f"Compressed_{filename}")
            
            with open(in_path, 'wb') as f: f.write(file_bytes)
            doc = fitz.open(in_path)
            
            if level in ['small', 'medium']:
                new_pdf = fitz.open()
                zoom = 1.3 if level == 'medium' else 0.7 
                mat = fitz.Matrix(zoom, zoom)
                jpg_q = 45 if level == 'medium' else 15
                
                for page in doc:
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("jpeg", jpg_quality=jpg_q)
                    imgdoc = fitz.open("pdf", fitz.open("jpeg", img_data).convert_to_pdf())
                    new_pdf.insert_pdf(imgdoc)
                
                new_pdf.save(out_path, garbage=4, deflate=True)
                new_pdf.close()
            else:
                doc.save(out_path, garbage=3, deflate=True)
                
            doc.close()
            return send_file(out_path, as_attachment=True, download_name=f"Compressed_{filename}")
        else:
            return "Unsupported file type.", 400
    except Exception as e:
        return f"Compression Error: {str(e)}", 500

if __name__ == '__main__':
    # 0.0.0.0 exposes the app to your local machine from inside the Docker container
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
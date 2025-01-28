import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import time

# Function to remove file with retry mechanism
def remove_file(filepath):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            os.remove(filepath)
            break  # File removed successfully, exit the loop
        except PermissionError:
            if attempt < max_retries - 1:
                print("File is in use, retrying...")
                time.sleep(1)  # Wait for a short period before retrying
            else:
                print("Failed to delete the file after several attempts.")
                raise

# Configuration of Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_text = None
    file_url = None  # URL for the uploaded file

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate URL for the uploaded file
            file_url = f"/uploads/{filename}"

            try:
                with Image.open(filepath) as image:
                    # Preprocessing
                    image = image.convert('L').filter(ImageFilter.SHARPEN)
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(3)
                    threshold = 128
                    image = image.point(lambda p: 255 if p > threshold else 0)
                    image.save("debug_image.png")

                    # OCR
                    config = '--psm 6 --oem 3 --dpi 300'
                    extracted_text = pytesseract.image_to_string(image, lang='eng', config=config).strip()

            except Exception as e:
                return f"Error processing image: {str(e)}", 500

    return render_template('index.html', extracted_text=extracted_text, file_url=file_url)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure the uploads folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

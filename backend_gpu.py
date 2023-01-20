import os
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask import send_file
import base64

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
OUTPUT_FOLDER = 'output'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'styleFile' not in request.files or 'contentFile' not in request.files:
        return 'Missing one or both files in the request', 400
    styleFile = request.files['styleFile']
    contentFile = request.files['contentFile']

    if styleFile.filename == '' or contentFile.filename == '':
        return 'One or both files not selected', 400

    if styleFile and contentFile and allowed_file(styleFile.filename) and allowed_file(contentFile.filename):
        styleFile.save(os.path.join(app.config['UPLOAD_FOLDER'], "styleFile.jpg"))
        contentFile.save(os.path.join(app.config['UPLOAD_FOLDER'], "contentFile.jpg"))
        os.system('python test_microAST.py --content ./uploads/contentFile.jpg --style ./uploads/styleFile.jpg')
        with open(os.path.join(app.config['OUTPUT_FOLDER'], "stylized_image.jpg"), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    else:
        return 'File type not allowed', 400


if __name__ == '__main__':
    app.run(debug=True)

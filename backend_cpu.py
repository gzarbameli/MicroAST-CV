import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask import send_file, send_from_directory
import base64
import shutil

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

    if request.headers.get("predefinedStyle") == "":
        styleFile = request.files['styleFile']
        styleFile.save(os.path.join(app.config['UPLOAD_FOLDER'], "styleFile.jpg"))
    else:
        if request.headers.get("predefinedStyle") == "Picasso":
            shutil.copyfile('./predefined_styles/picasso.jpg',os.path.join(app.config['UPLOAD_FOLDER'], "styleFile.jpg"))
        elif request.headers.get("predefinedStyle") == "Monet":
            shutil.copyfile('./predefined_styles/monet.jpg',os.path.join(app.config['UPLOAD_FOLDER'], "styleFile.jpg"))
        elif request.headers.get("predefinedStyle") == "Kandinskij":
            shutil.copyfile('./predefined_styles/kandinskij.jpg',os.path.join(app.config['UPLOAD_FOLDER'], "styleFile.jpg")) 
            
    model = request.headers.get("model")
    print(model)
    contentFile = request.files['contentFile']

    contentFile.save(os.path.join(app.config['UPLOAD_FOLDER'], "contentFile.jpg"))
    os.system('python test_microAST_cpu_only.py \
                --content ./uploads/contentFile.jpg \
                --style ./uploads/styleFile.jpg \
                --content_encoder models/' + model + '/content_encoder_iter.pth \
                --style_encoder models/' + model + '/style_encoder_iter.pth \
                --modulator models/' + model + '/modulator_iter.pth \
                --decoder models/' + model + '/decoder_iter.pth \
                --model ' + model
                )
    with open(os.path.join(app.config['OUTPUT_FOLDER'], "stylized_image.jpg"), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

@app.route('/gallery', methods=['GET'])
def get_predefined_styles():
    gallery = './gallery'
    image_list = []
    for image in os.listdir(gallery):
        with open(os.path.join(gallery, image), 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            image_list.append(encoded_string)
    return jsonify(image_list)

@app.route('/gallery/<path:filename>')
def serve_images(filename):
    return send_from_directory('gallery', filename)

@app.route('/predefined_styles/<path:filename>')
def serve_styles(filename):
    return send_from_directory('predefined_styles', filename)

if __name__ == '__main__':
    app.run(debug=True)

import cv2
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, flash, request, redirect, url_for
import os

from werkzeug.exceptions import abort
from werkzeug.utils import secure_filename
from keras.utils import custom_object_scope
from keras.layers import Activation

app = Flask(__name__)

IMG_SIZE = 64

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

tumor_class_name = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# load the tumor model
with custom_object_scope({'Activation': Activation}):
    tumor_model = tf.keras.models.load_model('models/brain-tumor-model/model.h5',
                                             custom_objects={'Activation': Activation})

breast_cancer_class_name = {1: 'Malignant', 0: 'Benign'}

# load the breast cancer model
breast_cancer_model = pickle.load(open("models/breast-cancer-model/model.pkl", 'rb'))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(filepath):
    img = cv2.imread(filepath)
    image = cv2.resize(img, (128, 128))
    img_array = np.expand_dims(image, axis=0)
    reshaped_image = np.reshape(img_array, (-1, 128, 128, 3))
    return reshaped_image


def prediction(texture_mean, area_mean, concavity_mean, area_se, concavity_se, fractal_dimension_se,
               smoothness_worst, concavity_worst, symmetry_worst, fractal_dimension_worst):
    data = np.array([[texture_mean, area_mean, concavity_mean, area_se, concavity_se, fractal_dimension_se,
                      smoothness_worst, concavity_worst, symmetry_worst, fractal_dimension_worst]])
    y_pred = breast_cancer_model.predict(data)
    return breast_cancer_class_name.get(y_pred[0])


@app.route('/')
def splash_screnn():
    return render_template('index.html')


@app.route('/default')
def default():
    return render_template('default.html')


@app.route('/breast_cancer')
def breast_cancer():
    return render_template('breastCancer.html')


@app.route('/breast_cancer', methods=["GET", "POST"])
def get_data():
    if request.method == "POST":
        texture_mean = request.form.get("texture_mean")
        area_mean = request.form.get("area_mean")
        concavity_mean = request.form.get("concavity_mean")
        area_se = request.form.get("area_se")
        concavity_se = request.form.get("concavity_se")
        fractal_dimension_se = request.form.get("fractal_dimension_se")
        smoothness_worst = request.form.get("smoothness_worst")
        concavity_worst = request.form.get("concavity_worst")
        symmetry_worst = request.form.get("symmetry_worst")
        fractal_dimension_worst = request.form.get("fractal_dimension_worst")

        result = prediction(texture_mean, area_mean, concavity_mean, area_se, concavity_se, fractal_dimension_se,
                            smoothness_worst, concavity_worst, symmetry_worst, fractal_dimension_worst)
    return render_template("breastCancer.html", result=result)


@app.errorhandler(403)
def not_found(e):
    return render_template("403.html"), 403


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


@app.route('/default', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file.filename = "img." + file.filename.rsplit('.', 1)[1]
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        image = prepare_image("static/uploads/img." + file.filename.rsplit('.', 1)[1])
        result = tumor_model.predict(image)[0]
        pnb = np.argmax(result)
        class_name = tumor_class_name[pnb]

        filename = "static/uploads/img." + file.filename.rsplit('.', 1)[1]
        return render_template('default.html', filename=filename, class_name=class_name)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)

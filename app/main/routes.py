'''
Module contains flask routing logic for application
'''
# Pylint Configurations
# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=wrong-import-order
# pylint: disable=bare-except
# pylint: disable=global-statement

from datetime import datetime
from flask import render_template, flash, redirect, url_for, request, g, \
    jsonify, current_app, make_response, has_app_context


from app.main import bp
from app.errors import ValidationError
from app.models import User, Note, InsertCountMetric, DeleteCountMetric
from config import Config
import requests
import sys
from config import myclassvariables


import json
import numpy as np
import cv2

# Initalize Variables
# Used as flag to store application health
HEALTH_STATUS_OK = True
# Used as flag to store application readiness
READY_STATUS_OK = True






def check_health():
    """ Simulates Application Health Status for Kubernetes Probes
        Will Store Application Health in Variable HEALTH_STATUS_OK for now """
    if HEALTH_STATUS_OK:
        current_app.logger.info(
            "Checking if Application is Healthy - Application is Healthy")
    else:
        current_app.logger.error(
            "Checking if Application is Healthy - Application is Unhealthy")
    return HEALTH_STATUS_OK


def check_ready():
    """ Simulates Application Readiness for Kubernetes Probes
        Will Store Application Health in Variable READY_STATUS_OK for now """
    if READY_STATUS_OK:
        current_app.logger.info(
            "Checking if Application is Ready - Application is Ready")
    else:
        current_app.logger.error(
            "Checking if Application is Ready - Application is not Ready")
    return READY_STATUS_OK


@bp.route('/health', methods=['GET'])
@custom_authmodule
def health(**kwargs):
    '''Provide Application Health Status to the Outside World'''
    if check_health():
        return make_response(jsonify({'Status': 'OK'}), 200)
    return make_response(jsonify({'Status': 'Unavailable'}), 503)


@bp.route('/ready', methods=['GET'])
@custom_authmodule
def ready(**kwargs):
    '''Provide Application Ready Status to the Outside World'''
    if check_ready():
        return make_response(jsonify({'Status': 'Ready'}), 200)
    return make_response(jsonify({'Status': 'Unavailable'}), 503)


# app = Flask(__name__)

# Function to preprocess image (replace this with your actual preprocessing function)
def preprocess_image(img):
    # Resize image to minimum size of 256x256 while maintaining aspect ratio
    min_size = min(img.shape[:2])
    scale_factor = 256 / min_size
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    img_resized = cv2.resize(img, new_size)
    # Crop 224x224 from the center
    center_x = new_size[0] // 2
    center_y = new_size[1] // 2
    half_crop = 112
    img_cropped = img_resized[center_y - half_crop:center_y + half_crop, center_x - half_crop:center_x + half_crop]
    # Normalize pixel values
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    img_normalized = (img_cropped - mean) / std
    # Transpose image from HWC to CHW layout
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    return img_transposed
    
def load_image(image_path):
    return cv2.imread(image_path)
    
# Function to convert image data to flat array
def image_to_flat_array(image_data):
    return image_data.flatten().tolist()
    
# Function to convert image data to JSON format
def image_to_json(image_data):
    return json.dumps({"inputs": [{"name": "data", "shape": [1, 3, 224, 224], "datatype": "FP32", "data": image_data}]})
    
# Function to load class labels
def load_class_labels():
    with open('./app/imagenet_classes.txt', 'r') as f:
        class_labels = f.read().splitlines()
    return class_labels
    
# Function to perform inference
def perform_inference(image):
    # Preprocess image
    image_processed = preprocess_image(image)
    
    # Convert image to flat array and JSON format
    image_flat = image_to_flat_array(image_processed)
    image_json = image_to_json(image_flat)
    # Send request to OpenVINO server
    url = 'https://data-server-demo-ai-gig.openshiftai-cluster-gig-e78f23787c77651c3692e4428c776eaa-0000.eu-gb.containers.appdomain.cloud/v2/models/data-server/infer'
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, data=image_json, headers=headers)
        if response.status_code == 200:
            # Parse response
            results = json.loads(response.text)
            # Get class labels
            class_labels = load_class_labels()
            # Get the top-1 prediction
            predictions = np.array(response.json()['outputs'][0]['data'])
            top_prediction_idx = np.argmax(predictions)
            top_prediction_label = class_labels[top_prediction_idx]
            return top_prediction_label
    except Exception as e:
        return "Error: {}".format(e)


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            # Perform inference
            file.save('image.jpg')
            img = load_image('image.jpg')
            result = perform_inference(img)
            return render_template('result.html', prediction=result)
    return render_template('index.html')


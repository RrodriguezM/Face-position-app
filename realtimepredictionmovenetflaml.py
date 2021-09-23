from botocore import exceptions
from GetImagesCameraSv3c.getImage import takePicture
import pandas as pd
import numpy as np
import pickle
import lightgbm
import time
import cv2
import boto3
import os
from helpers.mqtt_aws import MQTT_AWS

client = MQTT_AWS(os.environ.get('AMAZON_ROOT_CA'), os.environ.get('PRIVATE_PEM_KEY'),
                  os.environ.get('CERTIFICATE_PEM_CTR'), os.environ.get('IOT_ENDPOINT'))

# Load Pickle
with open('lgbm_front_1000.pkl', 'rb') as f:
    automl = pickle.load(f)

import tensorflow as tf
# Topic
topic = "device/02/data"
BUCKET_NAME = "face-detector-images"

# class Names and Positions
class_names = ['left', 'center', 'right', 'nada']

print("Loading the model")
#model = tf.saved_model.load('model_saved')
#movenet = model.signatures['serving_default']
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def preprocess_movenet_img(image):
    # image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize(image, [192, 192])
    return image


def preprocess_movenet(img_path):
    image_path = img_path
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize(image, 192, 192)
    return image


def feature_extractor(img_path):
    img_processed = preprocess_movenet(img_path)
    outputs = movenet(img_processed)
    features = np.reshape(tf.squeeze(
        outputs['output_0'][:, :, :5, :]).numpy(), 15)
    return features


def feature_extractor_img(image):
    print("preprocessing image")
    img_processed = preprocess_movenet_img(image)
    outputs = movenet(img_processed)
    features = np.reshape(tf.squeeze(outputs[:, :, :5, :]), 15)
    return features




s3_client = boto3.client('s3')
client.MQTTConnect()
try:
    while True:
        image = takePicture("192.168.101.172")
        features = feature_extractor_img(image)
        prediction = automl.predict(pd.DataFrame(features.reshape(1, 15)))
        file_name = f"{class_names[prediction[0]]}/{time.time()}.jpg"
        payload = {
            "status": class_names[prediction[0]],
            "S3_uri": f"s3://{BUCKET_NAME}/{file_name}"
        }
        # Send the data to AWS IoT
        client.MQTTPublish(topic, payload)
        cv2.imwrite("tmp.jpg", image)

        response = s3_client.upload_file(
            "tmp.jpg", BUCKET_NAME, file_name)

        print(payload)
        time.sleep(1)
except Exception as e:
    print(e)
    client.MQTTDisconnect()

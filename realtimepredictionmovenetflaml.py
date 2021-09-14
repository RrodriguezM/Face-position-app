from botocore import exceptions
from GetImagesCameraSv3c.getImage import takePicture
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import time
import cv2
import boto3
import os
from helpers.mqtt_aws import MQTT_AWS

print(os.environ.get('AMAZON_ROOT_CA'))
client = MQTT_AWS(os.environ.get('AMAZON_ROOT_CA'), os.environ.get('PRIVATE_PEM_KEY'),
                  os.environ.get('CERTIFICATE_PEM_CTR'), os.environ.get('IOT_ENDPOINT'))

# Topic
topic = "device/02/data"
BUCKET_NAME = "face-detector-images"

# class Names and Positions
class_names = ['left', 'center', 'right', 'nada']


model = tf.saved_model.load('model_saved')
movenet = model.signatures['serving_default']


def preprocess_movenet_img(image):
    # image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)
    return image


def preprocess_movenet(img_path):
    image_path = img_path
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)
    return image


def feature_extractor(img_path):
    img_processed = preprocess_movenet(img_path)
    outputs = movenet(img_processed)
    features = np.reshape(tf.squeeze(
        outputs['output_0'][:, :, :5, :]).numpy(), 15)
    return features


def feature_extractor_img(image):
    img_processed = preprocess_movenet_img(image)
    outputs = movenet(img_processed)
    features = np.reshape(tf.squeeze(
        outputs['output_0'][:, :, :5, :]).numpy(), 15)
    return features


# Load Pickle
with open('automl_front.pkl', 'rb') as f:
    automl = pickle.load(f)

s3_client = boto3.client('s3')
client.MQTTConnect()
try:
    while True:
        time.sleep(10)
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
except Exception as e:
    print(e)
    client.MQTTDisconnect()

from PIL import Image
import time
import streamlit as st
from dynamodb_json import json_util
import numpy as np
import pandas as pd
import boto3


# AWS Client connections
client = boto3.client('dynamodb')
s3_client = boto3.client('s3')

# Page Title
st.title("Available time Rafael Rodriguez")

# Dynamo DB Query get the las 12 Hours of Data
response = client.query(
    ExpressionAttributeValues={
        ':device': {
            'N': '2',
        },
        ':time': {
            'N': str(int((time.time()-43200)*1000)),
        },
    },
    KeyConditionExpression='device_id = :device and sample_time > :time',
    ProjectionExpression="device_data,sample_time",
    TableName='device_data_table',
)

# Data Processing
df = pd.DataFrame(json_util.loads(response["Items"]))
df["S3_uri"] = df["device_data"].apply(lambda x: x["S3_uri"])
df["device_data"] = df["device_data"].apply(lambda x: x["status"])
df["available"] = df["device_data"].apply(
    lambda x: "1" if x in ["left", "right", "center"] else "0")
df["available"] = df["available"].apply(np.int8)
df["sample_time"] = pd.to_datetime(df["sample_time"], unit='ms')

# Conditionally show the DF using the chackbox
if st.checkbox('Show Details Table'):
    df

# TODO: add the time in the bottom of the chart
# Show the graph of availability
st.line_chart(df["available"])


# Show Time Available
left_column_1, center_column_1, right_column_1 = st.columns(3)
left_column_1.metric(label="Minutes Out Site Count", value=str(
    round((df["available"].value_counts()[0])*10.65/60, 3)))
right_column_1.metric(label="Minutes In Site Count", value=str(
    round((df["available"].value_counts()[1])*10.65/60, 3)))

if 'left' not in st.session_state:
    st.session_state['left'] = 0
if 'center' not in st.session_state:
    st.session_state['center'] = 0
if 'right' not in st.session_state:
    st.session_state['right'] = 0

left_time = round((df["device_data"].value_counts()["left"])*10.65/60, 3)
center_time = round((df["device_data"].value_counts()["center"])*10.65/60, 3)
right_time = round((df["device_data"].value_counts()["right"])*10.65/60, 3)

delta_left = round((left_time - st.session_state.left)*60, 2)
delta_center = round((center_time - st.session_state.center)*60, 2)
delta_right = round((right_time - st.session_state.right)*60, 2)

st.session_state.left = left_time
st.session_state.center = center_time
st.session_state.right = right_time

# Show Each monitor Time
left_column_2, center_column_2, right_column_2 = st.columns(3)
left_column_2.metric(label="Seen Left Monitor", value=str(
    left_time), delta=f"{delta_left} sec")
center_column_2.metric(label="Seen Center Monitor", value=str(
    center_time), delta=f"{delta_center} sec")
right_column_2.metric(label="Seen Right Monitor", value=str(
    right_time), delta=f"{delta_right} sec")

# Show last know position
df.sort_values(by='sample_time')
left_column_3, center_column_3, right_column_3 = st.columns(3)
if df.iloc[-1]["available"]:
    center_column_3.metric(label="Last Known", value="In Site")
else:
    center_column_3.metric(label="Last Known", value="Out of Site")

# Get the image and show it
uri_splited = df.iloc[-1]["S3_uri"].split("/")
s3 = boto3.client('s3')
with open('tmpo.jpg', 'wb') as f:
    s3.download_fileobj(
        str(uri_splited[2]), f"{uri_splited[3]}/{uri_splited[4]}", f)
image = Image.open('tmpo.jpg')
st.image(image, caption=f"{uri_splited[3]}", width=620)

from PIL import Image
import time
import streamlit as st
from dynamodb_json import json_util
import numpy as np
import pandas as pd
import boto3

st.title("Available time Rafael Rodriguez")

client = boto3.client('dynamodb')
s3_client = boto3.client('s3')

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

df = pd.DataFrame(json_util.loads(response["Items"]))
df["S3_uri"] = df["device_data"].apply(lambda x: x["S3_uri"])
df["device_data"] = df["device_data"].apply(lambda x: x["status"])
df["available"] = df["device_data"].apply(
    lambda x: "1" if x in ["left", "right", "center"] else "0")
df["available"] = df["available"].apply(np.int8)
df["sample_time"] = pd.to_datetime(df["sample_time"], unit='ms')
if st.checkbox('Show Details Table'):
    df

st.line_chart(df["available"])

left_column, right_column = st.columns(2)
left_column.metric(label="Minutes Out Site Count", value=str(
    (df["available"].value_counts()[0])*10.65/60))
right_column.metric(label="Minutes In Site Count", value=str(
    (df["available"].value_counts()[1])*10.65/60))

left_column, center_column, right_column = st.columns(3)
left_column.metric(label="Seen Left Monitor", value=str(
    (df["device_data"].value_counts()["left"])*10.65/60))
center_column.metric(label="Seen Center Monitor", value=str(
    (df["device_data"].value_counts()["center"])*10.65/60))
right_column.metric(label="Seen Right Monitor", value=str(
    (df["device_data"].value_counts()["right"])*10.65/60))

# Ensure are ordered
df.sort_values(by='sample_time')
if df.iloc[-1]["available"]:
    st.metric(label="Last Known", value="In Site")
else:
    st.metric(label="Last Known", value="Out of Site")

uri_splited = df.iloc[-1]["S3_uri"].split("/")
s3 = boto3.client('s3')
with open('tmpo.jpg', 'wb') as f:
    s3.download_fileobj(
        str(uri_splited[2]), f"{uri_splited[3]}/{uri_splited[4]}", f)
image = Image.open('tmpo.jpg')
st.image(image, caption=f"{uri_splited[3]}")

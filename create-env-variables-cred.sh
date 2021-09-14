#!/bin/bash

set -e

export AWS_ACCESS_KEY_ID="xxx"
export AWS_SECRET_ACCESS_KEY="xxx"
export AWS_DEFAULT_REGION=us-east-1
export AMAZON_ROOT_CA="Certificates/AmazonRootCA1.cer"
export PRIVATE_PEM_KEY="Certificates/xxx-private.pem.key"
export CERTIFICATE_PEM_CTR="Certificates/xxx-certificate.pem.crt"
export IOT_ENDPOINT="xxx-ats.iot.us-east-1.amazonaws.com"

import time as t
import json
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT


class MQTT_AWS:
    def __init__(self, CA1cer, privatepem, certpem, endpoint):
        self.myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient("testDevice")
        self.myAWSIoTMQTTClient.configureEndpoint(endpoint, 8883)
        self.myAWSIoTMQTTClient.configureCredentials(
            CA1cer, privatepem, certpem)
        print("Inicialized")

    def MQTTConnect(self):
        """Establish connection between AWS and the device
        """
        self.myAWSIoTMQTTClient.connect()

    def MQTTPublish(self, topic, payload):
        """Publish a message to AWS IoT Core to the especified Endpoint 

        Args:
            topic (String): Name of the Topic to publish
            payload (Dict): Payload to publish
        """
        self.myAWSIoTMQTTClient.publish(topic, json.dumps(payload), 1)

    def MQTTDisconnect(self):
        """Brake connection between AWS and the device
        """
        print("Disconnecting from AWS")
        self.myAWSIoTMQTTClient.disconnect()

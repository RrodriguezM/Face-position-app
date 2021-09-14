import requests
import cv2
import numpy as np
import time


url = "http://admin:admin@"
url2 = "/web/tmpfs/snap.jpg"


def takePicture(ip, img_class="", dataset_type=""):
    """[summary]

    Args:
        ip ([String]): IP address provided in string to connect to the CAM

    Returns:
        [np.array]: snapshoot of the camera in jpg named in ns timestamp
    """

    urlReq = url + str(ip) + url2
    imgResp = requests.get(str(urlReq))
    imgNp = np.array(bytearray(imgResp.content), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = cv2.resize(img, (512, 512))
    if dataset_type != "":
        cv2.imwrite(
            f"images/{dataset_type}/{img_class}/{time.time_ns()}.jpg", img)
    return img


if __name__ == "__main__":
    # TODO: Create arg parse to del the ip
    takePicture("192.168.101.172", "right", "train")

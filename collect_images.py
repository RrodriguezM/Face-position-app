from GetImagesCameraSv3c.getImage import takePicture
import os

position = "empty"
os.system(f'say Starting {position}')
for i in range(100):
    takePicture("192.168.101.172", position, "train")
os.system(f'say End {position}')

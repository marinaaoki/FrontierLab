from __future__ import print_function
from time       import sleep
from PIL        import Image
from io         import StringIO

import urllib.request
import os

# Save 2 frames per second from the webcam.
def save_frame(count):
    # Default URL in M5Camera.
    url = 'http://192.168.4.1/jpg'
    frame = Image.open(urllib.request.urlopen(url))
    rotated = frame.rotate(180)
    rotated.save(f"{count:03}"+'.png')

    print("Saved frame " + f"{count:03}")

# Run loop to save stream.
def create_stream(source):
    os.chdir(source)
    count = 0
    while True:
        save_frame(count)
        # 4 FPS
        sleep(0.25)
        count+=1






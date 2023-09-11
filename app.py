from importlib import import_module
import os
from flask import Flask, render_template, Response,url_for

import cv2
import time
import numpy as np

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

#import numpy as np
import datetime
#import cv2
import torch
from absl import app, flags, logging
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models
from super_gradients.common.object_names import Models
import tkinter as tk
import threading

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        #frame = main()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #app.run(host='0.0.0.0', threaded=True)
    #try:
    #    app.run(main)
    #except SystemExit:
    #    pass
    app.run(host='0.0.0.0', threaded=True)
import os
import cv2
from base_camera import BaseCamera

import numpy as np
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

class Camera(BaseCamera):
    video_source = 0
 
    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()
 
    @staticmethod
    def set_video_source(source):
        Camera.video_source = source
 
    @staticmethod
    def frames():
        
        #cv2.namedWindow('video')

        #frame = camera.get_frame()

        #video_cap = cv2.VideoCapture(FLAGS.video)
        video_cap = cv2.VideoCapture(0)
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))

        # Initialize the video writer object（初始化视频写入器对象）
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

        # Initialize the DeepSort tracker（初始化DeepSort跟踪器）
        tracker = DeepSort(max_age=50)

        # Check if GPU is available, otherwise use CPU（检查是否有GPU可用，否则使用CPU）
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load the YOLO model（加载YOLO模型）
        #model = models.get(FLAGS.model, pretrained_weights="coco").to(device)
        model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

        # Load the COCO class labels the YOLO model was trained on
        #加载YOLO模型所训练的COCO类标签
        classes_path = "./configs/coco.names"
        with open(classes_path, "r") as f:
            class_names = f.read().strip().split("\n")

        # Create a list of random colors to represent each class
        #创建一个随机颜色列表来表示每个类
        np.random.seed(42)  # to get the same colors（得到相同的颜色）
        colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

        jisuan=1

        while True:

            start = cv2.getTickCount()
            ret, frame =video_cap.read()

            # If there is no frame, we have reached the end of the video
            #如果没有帧，我们就到达了视频的结尾
            #if not ret:
            #    print("End of the video file...")
            #    break

            jisuan-=1

            if(jisuan==0):
                # Run the YOLO model on the frame
                #在框架上运行YOLO模型
                # Perform object detection using the YOLO model on the current frame
                #在当前帧上使用YOLO模型执行对象检测
                #detect = next(iter(model.predict(frame, iou=0.5, conf=FLAGS.conf)))
                detect = next(iter(model.predict(frame, iou=0.5, conf=0.50)))

                # Extract the bounding box coordinates, confidence scores, and class labels from the detection results
                #从检测结果中提取边界框坐标、置信度分数和类标签
                bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
                confidence = torch.from_numpy(detect.prediction.confidence).tolist()
                labels = torch.from_numpy(detect.prediction.labels).tolist()
                # Combine the bounding box coordinates and confidence scores into a single list
                #将边界框坐标和置信度分数合并到单个列表中
                concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
                # Combine the concatenated list with the class labels into a final prediction list
                #将连接的列表与类标签组合成最终的预测列表
                final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]

                # Initialize the list of bounding boxes and confidences
                #初始化边界框和信任列表
                results = []

                # Loop over the detections（循环检测）
                for data in final_prediction:
                    # Extract the confidence (i.e., probability) associated with the detection
                    #提取与检测相关的置信度(即概率)
                    confidence = data[4]

                    # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
                    #通过确保置信度大于最小置信度来过滤掉弱检测
                    #if float(confidence) < FLAGS.conf:
                    if float(confidence) < 0.5:
                        continue

                    # If the confidence is greater than the minimum confidence, draw the bounding box on the frame
                    #如果置信度大于最小置信度，则在框架上绘制边界框
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    class_id = int(data[5])

                    # Add the bounding box (x, y, w, h), confidence, and class ID to the results list
                    #将边界框(x, y, w, h)、置信度和类ID添加到结果列表中
                    results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
                
                jisuan=2


            # Update the tracker with the new detections
            #用新的检测更新跟踪器
            tracks = tracker.update_tracks(results, frame=frame)
            
            # Loop over the tracks
            #在轨道上循环
            for track in tracks:
                # If the track is not confirmed, ignore it
                #如果轨道未被确认，则忽略它
                if not track.is_confirmed():
                    continue

                # Get the track ID and the bounding box
                #获取轨道ID和边界框
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

                # Get the color for the class
                #选择适合班级的颜色
                color = colors[class_id]
                B, G, R = int(color[0]), int(color[1]), int(color[2])

                # Create text for track ID and class name
                #为曲目ID和类名创建文本
                text = str(track_id) + " - " + str(class_names[class_id])
                
                # Draw bounding box and text on the frame
                #在框架上绘制边界框和文本
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                #temp.image(cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR, channels="RGB",use_column_width=True))

                stop = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (stop - start)
                fps = '{}: {:.3f}'.format('FPS', fps)
                (fps_w, fps_h), baseline = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (2, 20 - fps_h - baseline), (2 + fps_w, 18), color=(0, 0, 0), thickness=-1)
                cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=(255, 255, 255), thickness=2)

                #tframe=cv2.imencode('.jpg', frame)[1].tobytes()

                #cv2.imshow('video', frame)

                #writer.write(frame)


                if cv2.waitKey(1) == ord("q"):
                    break

                #yield (b'--frame\r\n'
                #    b'Content-Type: image/jpeg\r\n\r\n' + tframe + b'\r\n')
                yield cv2.imencode('.jpg', frame)[1].tobytes()

                
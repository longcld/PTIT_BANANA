#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import numpy as np
import cv2
import detectSign
from tensorflow.keras.models import load_model
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import segment
import tensorflow as tf
import math
import keras
import dlib

# __________________________________________________PATH______________________________________________________
models_path = '../../catkin_ws/src/catkin_ws/src/team805/models/'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

keras.backend.set_session(session)

speed = rospy.Publisher('/team805/set_speed', Float32)
angle = rospy.Publisher('/team805/set_angle', Float32)

# Weights
weightsPath = models_path + "MobileNetSSD_deploy.caffemodel"

# Architecture of model
proto = models_path + "MobileNetSSD_deploy.prototxt"

# Load car detection model
net = cv2.dnn.readNetFromCaffe(proto, weightsPath)

lane = segment.load_model(models_path + "lane.h5")
sign = load_model(models_path + "sign1.h5")
# print(lane.summary())

lane._make_predict_function()
sign._make_predict_function()

# ______________________________________________________DEFAULT VALUE___________________________________________
ready_turn = -1
turn = 'none'
turns = []
counts = 15
end_track = 0
s = 0
a = 0
total_time = 0
image_count = 0
is_car = False
tracker = dlib.correlation_tracker()
startX, startY, endX, endY = [0, 0, 0, 0]
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


# ___________________________________________________________DETECT CAR_________________________________________
def detect_car(image):
    (H, W) = (image.shape[0], image.shape[1])
    # Convert image to blob types
    blob = cv2.dnn.blobFromImage(image, size=(W, H), ddepth=cv2.CV_8U)
    # Push blob image to model
    net.setInput(blob, scalefactor=1.0 / 127.5, mean=[127.5, 127.5, 127.5])
    # Run model
    detections = net.forward()
    # Get 1st box from 100 detected box(3rd index)
    box = detections[0, 0, 0, 3:7] * np.array([W, H, W, H])
    # 3:7 be a coordinate of box detected
    (startX, startY, endX, endY) = box.astype("int")
    rect = [startX, startY, endX, endY]

    # Get label encode of output model
    idx = int(detections[0, 0, 0, 1])

    # Check if detected box is car
    if CLASSES[idx] != "car":
        return [0, 0, 0, 0]

    return rect


def classify(box):
    box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)

    with session.as_default():
        with session.graph.as_default():
            y_pred = sign.predict_classes(cv2.resize(box, (32, 32)).reshape(1, 32, 32, 1))[0]

    if y_pred == 0:
        return 'none'
    elif y_pred == 1:
        return 'left'
    else:
        return 'right'


def get_center_lane_point(y_pred, x_cor):
    temp = np.where(y_pred[x_cor] == 1)[0]

    i = 0
    j = 1
    start = 0
    end = 0

    while (j < len(temp)):
        x = j - i

        if (temp[j] - temp[i]) == x:
            if (x) > end - start:
                start = temp[i]
                end = temp[j]
            j += 1

        else:
            i = j
            j += 1

    return x_cor, round(start + (end - start) / 2)


def getAngle(x, y):
    return - math.degrees(math.atan((160 - y) / (240 - x)))


def predict(img):
    return np.squeeze(lane.predict(img.reshape(1, 240, 320, 3).astype('float32')))


def callback(ros_data):
    global ready_turn, turn, turns, counts, s, a, image_count, tracker, is_car, \
        startX, startY, endX, endY, end_track, total_time

    image_count += 1
    # ____________________________________________________________TURN___________________________________________________
    if counts == 0:
        counts = 5
        l = turns.count('left')
        r = turns.count('right')

        if len(turns) > 2:
            if (l > r):
                ready_turn = 21
                turn = 'left'
            elif (r > l):
                ready_turn = 21
                turn = 'right'
            else:
                turn = 'none'
        turns = []

    np_arr = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, 1)

    # ____________________________________________________SIGN________________________________________________
    rects = detectSign.getRects(image_np)

    for rect in rects:
        x, y, w, h = rect
        y = y - 2
        x = x - 2
        roi = image_np[y:y + h + 7, x:x + w + 7]

        turn_pred = classify(roi)

        if (turn_pred == 'left') or (turn_pred == 'right'):
            turns.append(turn_pred)
            cv2.rectangle(image_np, (x, y), (x + w, y + h), [0, 255, 0], 1)
            cv2.putText(image_np, turn_pred, (x, y + h + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))

    with session.as_default():
        with session.graph.as_default():
            y_pred = np.squeeze(lane.predict(image_np.reshape(1, 240, 320, 3).astype('float32')))

    # _____________________________________________________READY TURN______________________________________________
    if 0 < ready_turn < 19:
        if turn == 'left':
            s = 0
            a = -60
        elif turn == 'right':
            s = 0
            a = 60

    else:
        # __________________________________________________DETECT CAR_____________________________________________
        # Set frequent detect car
        if image_count % 3 == 0:
            startX1, startY1, endX1, endY1 = detect_car(image_np)

            if [startX1, startY1, endX1, endY1] != [0, 0, 0, 0]:
                startX, startY, endX, endY = [startX1, startY1, endX1, endY1]
                # if detected box is car
                is_car = True
                # Convert car box to dlib type
                rect_car = dlib.rectangle(startX, startY, endX, endY)
                # Start track by this box for 10 next frames
                tracker.start_track(image_np, rect_car)

                startX -= 15
                startY = 0
                endX += 15
                endY += 20

        else:
            # Update tracking
            if is_car == True:
                # Set last frame to original
                if image_count % 3 == 2:
                    is_car = False
                    end_track = image_count

                # Update tracking image
                tracker.update(image_np)

                # Get new positon of detected car
                pos = tracker.get_position()

                # Get coordinate
                startX = int(pos.left() - 15)
                startY = 0
                # startY = int(pos.top() - 10)
                endX = int(pos.right() + 15)
                endY = int(pos.bottom() + 20)
        # _________________________________________________________TRACKING CAR_________________________________________
        if not is_car and (image_count - end_track == 7) and endY != 0:
            startX, startY, endX, endY = [0, 0, 0, 0]
        if endY > 240:
            startY = 0
            endY = 240
        if startX < 0:
            startX = 0
        if endX > 320:
            endX = 320

        # Draw
        image_np = cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Set detected region not lane
        y_pred[startY:endY, startX:endX] = 0

        # _____________________________________________________SPEED & ANGLE__________________________________________
        if (y_pred[140].sum() > 280):
            x, y = (120, 160)
        else:
            temp = np.where(y_pred == 1)
            image_np[temp[0], temp[1]] = [0, 0, 80.5]

            # x, y = get_center_lane_point(y_pred, 152)
            x1, y1 = get_center_lane_point(y_pred, 120)
            x2, y2 = get_center_lane_point(y_pred, 150)
            #
            x = np.mean([x1, x2])
            y = np.mean([y1, y2])

        cv2.line(image_np, (160, 240), (int(y), int(x)), [0, 255, 0], 2)

        if (159 < y < 161):
            s = 120
        elif (y < 140 or y > 180):
            s = 10
        else:
            s = 60
        # ________________________________________________________SLOW___________________________________________
        if len(turns) > 0 and turns[-1] != 'none':
            s = 0
        a = getAngle(x, y)

        cv2.putText(image_np, str(s) + " km/h", (170, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))

    # if image_count % 10 == 0:
    #     print("FPS: ", image_count / total_time)
    # cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('box', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('segmented', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('origin', 640, 480)
    # cv2.resizeWindow('box', 640, 480)
    # cv2.resizeWindow('segmented', 640, 480)
    # cv2.imshow('origin', image_np)
    # cv2.imshow('box', box_img)
    # cv2.imshow('segmented', y_pred)
    # cv2.waitKey(1)

    speed.publish(s)
    angle.publish(round(a))
    # print(ready_turn)
    ready_turn -= 1
    counts -= 1


def depth_callback(data):
    np_arr = np.fromstring(data.data, np.uint8)
    image_np = cv2.imdecode(np_arr, 1)
    print("xxxxxxxxxxxxxxxxxxxxxxx")
    cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    cv2.imshow('depth', image_np)
    cv2.waitKey(1)
    speed.publish(50)
    angle.publish(round(2))
    # speed.publish(s)
    # angle.publish(round(a))


def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    print('READY')
    rospy.Subscriber('/team805/camera/rgb/compressed', CompressedImage, callback)
    # rospy.Subscriber('/team805_image/compressed', CompressedImage, callback)
    # rospy.Subscriber('team805/camera/depth/compressed', CompressedImage, depth_callback)
    cv2.destroyAllWindows()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    print('END');


if __name__ == '__main__':
    listener()

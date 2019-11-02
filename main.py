import time

import cv2
import sys
import os
import socket
import json
import math
from dotenv import load_dotenv
import numpy as np
from threading import Thread, Lock
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import cognitive_face as CF

load_dotenv()
KEY = os.getenv('FACE_API_SUBSCRIPTION_KEY')
ENDPOINT = os.getenv('FACE_API_ENDPOINT')

CF.Key.set(KEY)
CF.BaseUrl.set(ENDPOINT)
face_cascade = cv2.CascadeClassifier("./algorithm.xml")

p0 = None
face_center = None
frame_gray = None
font = cv2.FONT_HERSHEY_SIMPLEX
gesture = False
x_movement = 0
y_movement = 0
gesture_show = 10  # number of frames a gesture is shown
max_head_movement = 20
movement_threshold = 50
gesture_threshold = 150
# gesture_lock is used when a gesture is recorded and an action has to be taken based on it
gesture_lock = Lock()
handling_action = False
# face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))


# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_coords(p1):
    try: return int(p1[0][0][0]), int(p1[0][0][1])
    except: return int(p1[0][0]), int(p1[0][1])


def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


class StreamServer:
    def __init__(self, host=None, port=10001):
        self.thread = Thread(target=self.get_items, args=())
        self.host = host
        self.port = port
        self.started = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.update_lock = Lock()
        self.current_item = None

    def start(self):
        if self.started:
            print("already started")
            return None
        self.started = True
        self.thread.start()

    def get_items(self):
        print("connecting to the socket server")
        self.sock.connect((self.host, self.port))
        while self.started:
            try:
                data = self.sock.recv(1024)
            except:
                return
            self.update_lock.acquire()
            if data and len(data) > 0:
                self.current_item = json.loads(data)
            self.update_lock.release()

    def ack_item(self, gesture_in):
        print("acking item...")
        self.sock.sendall(gesture_in)

    def stop(self):
        print("closing streaming server...")
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        self.thread.join()


class WebcamVideoStream:
    def __init__(self, src=0, width=320, height=240):
        self.thread = Thread(target=self.update, args=())
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print("already started!!")
            return None
        self.started = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()


def find_face():
    try:
        # find the face in the image
        face_found = False
        frame_num = 0
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        while frame_num < 30:
            # Take first frame and find corners in it
            frame_num += 1
            ret, frame = cap.read()
            global frame_gray
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
            faces = face_cascade.detectMultiScale(frame_gray,
                                                  scaleFactor=1.1,
                                                  minNeighbors=5,
                                                  minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_found = True
            # cv2.imshow('image', frame)
            cv2.imwrite('face.jpg', frame)
            cv2.waitKey(1)
        global face_center
        face_center = x + w / 2, y + h / 3
        global p0
        p0 = np.array([[face_center]], np.float32)
    except:
        sys.exit(1)


def take_action(gesture, current_item):
    if handling_action:
        return
    print(gesture, current_item)


def detect_emotions():
    attr = "age,gender,emotion"
    detected_faces = CF.face.detect('./face.jpg', attributes=attr)
    if not detected_faces:
        print("could not find face")
        sys.exit(1)
    for face in detected_faces:
        print(face)


if __name__ == "__main__":
    print("attempting to find face in video...")
    find_face()
    vs = WebcamVideoStream().start()
    ss = StreamServer("localhost")
    ss.start()
    print("Face center: {}".format(p0))
    f_counter = 1
    detect_emotions()
    while True:
        frame = vs.read()
        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if not err:
            sys.exit(1)
        cv2.circle(frame, get_coords(p1), 4, (0, 0, 255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

        # get the xy coordinates for points p0 and p1
        a, b = get_coords(p0), get_coords(p1)
        x_movement += abs(a[0] - b[0])
        y_movement += abs(a[1] - b[1])
        text = 'x_movement: ' + str(x_movement)
        if not gesture: cv2.putText(frame, text, (50, 50), font, 0.8, (0, 0, 255), 2)
        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame, text, (50, 100), font, 0.8, (0, 0, 255), 2)

        if x_movement > gesture_threshold:
            gesture = 'No'
        if y_movement > gesture_threshold:
            gesture = 'Yes'
        if gesture and gesture_show > 0:
            cv2.putText(frame, 'Gesture Detected: ' + gesture, (50, 50), font, 1.2, (0, 0, 255), 3)
            take_action(gesture, ss.current_item)
            if gesture_show == 10:
                gesture_lock.acquire()
                handling_action = True
                ss.ack_item(gesture)
                gesture_lock.release()
            gesture_show -= 1
        if gesture_show == 0:
            gesture = False
            x_movement = 0
            y_movement = 0
            gesture_show = 10  # number of frames a gesture is shown
            if gesture_show == 10:
                gesture_lock.acquire()
                handling_action = False
                gesture_lock.release()
        # p0 = p1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(5) & 0xFF == ord('r'):
            print("finding ya face...")
            find_face()
            x_movement = 0
            y_movement = 0
        f_counter += 1
    vs.stop()
    cv2.destroyAllWindows()
    ss.stop()

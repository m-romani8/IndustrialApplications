import cv2
import dlib
import argparse
import time
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import socket
import os

def AutoWebCamFinder():
    index = 0
    while index < 100:  # Prova fino a 10 dispositivi
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:  # Se la webcam risponde
            print(f"Webcam trovata all'indice: {index}")
            return index,cap.release()
        index += 1

# Funzione per calcolare l'eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Argomenti

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-w", "--webcam", type=int, default=0,
#                 help="index of webcam on system")
# args = vars(ap.parse_args())

# Soglia e numero di frame consecutivi
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
MAX_OPEN_FRAMES = 3  # Numero massimo di frame "aperti" prima di azzerare il contatore

# Variabili globali
COUNTER = 0
OPEN_FRAMES = 0

# Caricamento del rilevatore di volti e predittore di punti
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor_path = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)

# Indici degli occhi
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Attesa della connessione
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 12345))
server.listen(0)
print("In attesa di connessioni...")

conn, addr = server.accept()
print(f"Connessione da {addr}")

# Avvio del video stream
print("[INFO] starting video stream...")
#vs = VideoStream(src=args["webcam"]).start()
#vs = VideoStream(src=0).start() # my system's webcam index 
vs = VideoStream(src=0, usePiCamera=False).start() # my system's webcam index 


#webcam_index,_ = AutoWebCamFinder()
#vs = VideoStream(src=webcam_index).start()
time.sleep(1.0)




while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Disegna contorni occhi
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            OPEN_FRAMES = 0  # Reset del contatore di occhi aperti

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # Placeholder per funzione personalizzata
                #trigger_drowsiness_function()

                msg = "DROWSY"
                conn.send(msg.encode())

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if COUNTER > 0:
                OPEN_FRAMES += 1
                if OPEN_FRAMES >= MAX_OPEN_FRAMES:
                    COUNTER = 0  # Azzeramento se gli occhi restano aperti per troppo tempo

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
conn.close()
server.close()
cv2.destroyAllWindows()
vs.stop()

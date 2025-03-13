import cv2

index = 0
while index < 10:  # Prova fino a 10 dispositivi
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:  # Se la webcam risponde
        print(f"Webcam trovata all'indice: {index}")
        cap.release()
    index += 1
    
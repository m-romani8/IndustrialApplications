import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import os
import pickle
import torch
import torch.nn as nn
import time

# --- Configurazioni e Costanti ---
SEQ_LENGTH = 60        
NUM_FEATURES = 1
HIDDEN_SIZE = 128

class EyeStateLSTM_96(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateLSTM_96, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 32, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size - 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 32, num_classes)

        self.name = "lstm_1_layer_96"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

# Etichetta/e considerata/e come stato di sonnolenza
DROWSY_LABELS = {'close'}

# Indici landmark per gli occhi (da MediaPipe Face Mesh)
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    """Calcola l'Eye Aspect Ratio per un set di landmark oculari."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + 1, y + text_h + 1), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

# --- Inizializzazione ---
print("[INFO] Inizializzazione...")

# Verifica dispositivo (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilizzo del dispositivo: {device}")

OUTPUT_DIR = "nn_output_pytorch"
SCALER_FILENAME = os.path.join(OUTPUT_DIR, "ear_scaler.pkl")
ENCODER_FILENAME = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

# Caricamento Label Encoder
try:
    with open(ENCODER_FILENAME, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"[INFO] Label Encoder caricato da: {ENCODER_FILENAME}")
    print(f"[INFO] Classi note: {label_encoder.classes_}")
    num_classes = len(label_encoder.classes_)
except FileNotFoundError:
    print(f"[ERRORE] File Label Encoder non trovato: {ENCODER_FILENAME}")
    exit()
except Exception as e:
    print(f"[ERRORE] Impossibile caricare il Label Encoder: {e}")
    exit()

MODEL_FILENAME = os.path.join(OUTPUT_DIR, "lstm_1_layer_relu_96.pth")

# Caricamento Modello
print("[INFO] Caricamento modello")
try:
    model = EyeStateLSTM_96(input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE, num_classes=num_classes)

    model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Modello caricato con successo.")
except FileNotFoundError:
    print(f"[ERRORE] File modello non trovato: {MODEL_FILENAME}")
    exit()
except RuntimeError as e:
    print(f"[ERRORE] Errore durante il caricamento del modello. Possibile discrepanza nella definizione?")
    print(e)
    exit()
except Exception as e:
    print(f"[ERRORE] Impossibile caricare il modello: {e}")
    exit()

# Caricamento Scaler
try:
    with open(SCALER_FILENAME, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[INFO] Scaler caricato da: {SCALER_FILENAME}")
except FileNotFoundError:
    print(f"[ERRORE] File scaler non trovato: {SCALER_FILENAME}")
    exit()
except Exception as e:
    print(f"[ERRORE] Impossibile caricare lo scaler: {e}")
    exit()

# Coppie di colori (scuro, chiaro) per i testi informativi
# Formato: {"dark": (B, G, R), "light": (B, G, R)}

# Colore per EAR e Stato
# BGR: (Blu, Verde, Rosso)
EAR_STATE_TEXT_COLORS = {
    "dark": (200, 120, 0),    
    "light": (255, 180, 100)
}
TIMES_TEXT_COLORS = {
    "dark": (30, 110, 130),
    "light": (70, 220, 255)   
}
FPS_TEXT_COLORS = {
    "dark": (30, 50, 130),    
    "light": (70, 100, 255)   
}

BACKGROUND_BRIGHTNESS_THRESHOLD = 200
TEXT_PADDING_FOR_BG_CHECK = 5

def get_adaptive_text_color(frame, text_content, position, font, scale, thickness, color_pair, threshold, padding):
    (text_w, text_h), baseline_offset = cv2.getTextSize(text_content, font, scale, thickness)
    x, y_baseline = position

    roi_x1 = max(0, x - padding)
    roi_y1 = max(0, y_baseline - text_h - padding)
    roi_x2 = min(frame.shape[1], x + text_w + padding)
    roi_y2 = min(frame.shape[0], y_baseline + baseline_offset + padding)

    if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
        return color_pair["light"]  # Default a colore chiaro

    background_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if background_roi.size == 0:
        return color_pair["light"] # Default se la ROI è vuota

    try:
        gray_roi = cv2.cvtColor(background_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_roi)
    except cv2.error:
        return color_pair["light"]

    if avg_brightness > threshold:
        return color_pair["dark"]  # Sfondo chiaro -> testo scuro
    else:
        return color_pair["light"] # Sfondo scuro -> testo chiaro

# Inizializzazione Webcam
print("[INFO] Avvio stream webcam...")
cap = cv2.VideoCapture(0)

# Inizializzazione MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

if not cap.isOpened():
    print("[ERRORE] Impossibile aprire la webcam.")
    face_mesh.close()
    exit()

# Deque per memorizzare gli ultimi SEQ_LENGTH valori EAR
ear_sequence = deque(maxlen=SEQ_LENGTH)
total_processing_times = []
nn_processing_times = []

current_prediction = "Inizializzazione..."
current_ear = 0.0

print("[INFO] Avvio rilevamento in tempo reale. Premere 'q' per uscire.")

while True:
    start_time_total = time.time()

    ret, frame = cap.read()
    if not ret:
        print("[ERRORE] Impossibile leggere il frame dalla webcam.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False 

    results = face_mesh.process(rgb_frame)

    rgb_frame.flags.writeable = True

    face_found = False
    if results.multi_face_landmarks:
        face_found = True
        # Prendi i landmark del primo volto trovato
        face_landmarks = results.multi_face_landmarks[0]

        # Estrai coordinate landmark per gli occhi
        left_eye = np.array([(face_landmarks.landmark[i].x * frame_width, face_landmarks.landmark[i].y * frame_height) for i in LEFT_EYE_IDXS])
        right_eye = np.array([(face_landmarks.landmark[i].x * frame_width, face_landmarks.landmark[i].y * frame_height) for i in RIGHT_EYE_IDXS])

        # Calcola EAR - Eye Aspect Ratio
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        current_ear = (left_ear + right_ear) / 2.0

        # Aggiungi l'EAR corrente alla sequenza
        ear_sequence.append(current_ear)

        # Disegna i contorni degli occhi (opzionale)
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 0), 1)

        # --- Predizione con LSTM/GRU ---
        if len(ear_sequence) == SEQ_LENGTH:
            # 1. Prendi la sequenza completa
            sequence_data_np = np.array(list(ear_sequence)).reshape(-1, 1) # Shape (SEQ_LENGTH, 1)

            # 2. Scala la sequenza
            try:
                scaled_sequence = scaler.transform(sequence_data_np) # Applica lo scaler
            except Exception as e:
                 print(f"[WARN] Errore nello scaling della sequenza: {e}")
                 scaled_sequence = None

            if scaled_sequence is not None:
                input_tensor = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0).to(device) # Shape (1, SEQ_LENGTH, 1)

                # 3. Esegui Inferenza
                start_time_nn = time.time()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted_idx_tensor = torch.max(outputs.data, 1)
                    predicted_idx = predicted_idx_tensor.cpu().item()

                try:
                    current_prediction = label_encoder.inverse_transform([predicted_idx])[0]
                except IndexError:
                    current_prediction = "Errore Decode"
                    print(f"[WARN] Indice predetto {predicted_idx} fuori range per label encoder?")
                except Exception as e:
                    current_prediction = "Errore Decode"
                    print(f"[WARN] Errore nel decodificare la predizione: {e}")
                end_time_nn = time.time()
                nn_processing_time = (end_time_nn - start_time_nn) * 1000
                nn_processing_times.append(nn_processing_time)
        else:
            current_prediction = f"Raccogliendo ({len(ear_sequence)}/{SEQ_LENGTH})"

    else:
        # Nessun volto rilevato
        current_prediction = "Nessun volto"
        current_ear = 0.0 # Resetta EAR se non c'è volto
        ear_sequence.clear()

    end_time_total = time.time()
    total_processing_time = (end_time_total - start_time_total) * 1000
    total_processing_times.append(total_processing_time)

    # --- Visualizzazione ---
    font_info = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_ear_state = 0.7
    font_scale_times = 0.6
    font_scale_fps = 0.5
    font_thickness_info = 2

    # EAR Text
    ear_text = f"EAR: {current_ear:.2f}"
    ear_pos = (10, 30)
    ear_color = get_adaptive_text_color(frame, ear_text, ear_pos, font_info, font_scale_ear_state, font_thickness_info, EAR_STATE_TEXT_COLORS, BACKGROUND_BRIGHTNESS_THRESHOLD, TEXT_PADDING_FOR_BG_CHECK)
    cv2.putText(frame, ear_text, ear_pos, font_info, font_scale_ear_state, ear_color, font_thickness_info)

    # Stato Text
    state_text = f"State: {current_prediction}"
    state_pos = (10, 60)
    state_color = get_adaptive_text_color(frame, state_text, state_pos, font_info, font_scale_ear_state, font_thickness_info, EAR_STATE_TEXT_COLORS, BACKGROUND_BRIGHTNESS_THRESHOLD, TEXT_PADDING_FOR_BG_CHECK)
    cv2.putText(frame, state_text, state_pos, font_info, font_scale_ear_state, state_color, font_thickness_info)

    # Calcola i tempi medi
    mean_total_time = np.mean(total_processing_times) if len(total_processing_times) > 0 else 0.0
    mean_nn_time = np.mean(nn_processing_times) if len(nn_processing_times) > 0 else 0.0

    if current_prediction in DROWSY_LABELS:
        cv2.rectangle(frame, (frame_width // 2 - 200, frame_height - 50),
                      (frame_width // 2 + 200, frame_height - 10), (0, 0, 255), -1) # Sfondo rosso
        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (frame_width // 2 - 175, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 255), 2, cv2.LINE_AA) # Testo giallo/bianco

    # Mostra il frame
    cv2.imshow("Drowsiness detection by LSTM", frame)

    # Uscita con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Uscita richiesta.")
        break

# --- Pulizia ---
print("[INFO] Pulizia e chiusura...")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("[INFO] Terminato.")
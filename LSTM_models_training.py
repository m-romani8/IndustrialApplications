import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import json
import pandas as pd
from collections import deque
import os
import sys
import traceback
import time
import pickle
import glob
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilizzo dispositivo: {device}")

BASE_DIR = "dmd"
VIDEO_EXT = "_rgb_face.mp4"
JSON_EXT = "_rgb_ann_drowsiness.json"

VIDEO_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, '*', '*', '*', f'*{VIDEO_EXT}')))
JSON_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, '*', '*', '*', f'*{JSON_EXT}')))

aligned_video_paths = []
aligned_json_paths = []
for video_path in VIDEO_PATHS:
    expected_json_path = video_path.replace(VIDEO_EXT, JSON_EXT)
    if expected_json_path in JSON_PATHS:
        aligned_video_paths.append(video_path)
        aligned_json_paths.append(expected_json_path)
    else:
        pass
VIDEO_PATHS = aligned_video_paths
JSON_PATHS = aligned_json_paths

print(f"[INFO] Trovati {len(VIDEO_PATHS)} video e {len(JSON_PATHS)} file JSON. Verifica che siano allineati.")

if not VIDEO_PATHS or not JSON_PATHS:
    print("[ERRORE] Nessun video o file JSON trovato nella struttura attesa sotto 'dmd'. Controlla i percorsi.")
    sys.exit(1)
print(f"[INFO] Trovati {len(VIDEO_PATHS)} video con file JSON corrispondenti.")

SEQ_LENGTH = 60
NUM_FEATURES = 1
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 96
HIDDEN_SIZE_DENSE = 64

RANDOM_STATE = 99

TEST_SIZE_VIDEOS = 0.1
VALIDATION_SPLIT_VIDEOS = 0.2

OUTPUT_DIR = "nn_output_pytorch"
MODEL_FILENAME = os.path.join(OUTPUT_DIR, "eye_state_lstm.pth")

TRAIN_VIDEO_PATHS_FILE = os.path.join(OUTPUT_DIR, "train_video_paths.pkl")
TEST_VIDEO_PATHS_FILE = os.path.join(OUTPUT_DIR, "test_video_paths.pkl")

SCALER_FILENAME = os.path.join(OUTPUT_DIR, "ear_scaler.pkl")
ENCODER_FILENAME = os.path.join(OUTPUT_DIR, "label_encoder.pkl")

RAW_DATA_DIR = os.path.join(OUTPUT_DIR, "raw_data")
TRAIN_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "train")
TEST_RAW_DATA_DIR = os.path.join(RAW_DATA_DIR, "test")

RAW_DATA_EXTRACTED_MARKER = os.path.join(RAW_DATA_DIR, "extracted_complete.marker")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_RAW_DATA_DIR, exist_ok=True)
os.makedirs(TEST_RAW_DATA_DIR, exist_ok=True)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

def load_ground_truth(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        actions = data['openlabel']['actions']
        frame_state = {}
        for action in actions.values():
            if 'eyes_state' in action['type']:
                state_type = action['type'].split('/')[-1]
                if state_type == 'closed':
                    state_type = 'close'
                for interval in action['frame_intervals']:
                    for frame in range(interval['frame_start'], interval['frame_end'] + 1):
                        frame_state[frame] = state_type
        return frame_state
    except FileNotFoundError:
        print(f"[ERRORE] File JSON non trovato: {json_path}")
        return None
    except Exception as e:
        print(f"[ERRORE] Errore nel caricare/parsare {json_path}: {e}")
        return None

def extract_ear_from_video(video_path):
    print(f"   [INFO] Estrazione EAR da video: {os.path.basename(video_path)}...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5, refine_landmarks=False
    )
    left_eye_idxs = [33, 160, 158, 133, 153, 144]
    right_eye_idxs = [362, 385, 387, 263, 373, 380]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERRORE] Impossibile aprire il video: {video_path}")
        face_mesh.close()
        return None

    ear_data = {}
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = face_mesh.process(rgb_frame)
            current_ear = np.nan

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_eye = np.array([(face_landmarks.landmark[i].x * frame_width, face_landmarks.landmark[i].y * frame_height) for i in left_eye_idxs])
                right_eye = np.array([(face_landmarks.landmark[i].x * frame_width, face_landmarks.landmark[i].y * frame_height) for i in right_eye_idxs])
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                current_ear = (left_ear + right_ear) / 2.0
        except Exception as e:
            current_ear = np.nan

        ear_data[frame_index] = current_ear
        frame_index += 1

    cap.release()
    face_mesh.close()
    return ear_data

# --- PyTorch Dataset---
class EarDataset(Dataset):
    def __init__(self, sequences, labels):
        if isinstance(sequences, np.ndarray):
                self.sequences = torch.tensor(sequences, dtype=torch.float32)
        elif isinstance(sequences, torch.Tensor):
                self.sequences = sequences.float()
        else:
                raise TypeError("Sequences must be numpy array or torch tensor")

        if isinstance(labels, np.ndarray):
                self.labels = torch.tensor(labels, dtype=torch.long)
        elif isinstance(labels, torch.Tensor):
                self.labels = labels.long()
        else:
                raise TypeError("Labels must be numpy array or torch tensor")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def process_or_load_raw_data_for_video(video_path, json_path, output_subdir):
    video_basename = os.path.basename(video_path).replace(VIDEO_EXT, "")
    ear_raw_file = os.path.join(output_subdir, f"{video_basename}_ear_raw.pkl")
    gt_labels_file = os.path.join(output_subdir, f"{video_basename}_gt_labels.pkl")

    if os.path.exists(ear_raw_file) and os.path.exists(gt_labels_file):
        try:
            with open(ear_raw_file, 'rb') as f: raw_ear_data = pickle.load(f)
            with open(gt_labels_file, 'rb') as f: ground_truth = pickle.load(f)
            return raw_ear_data, ground_truth
        except Exception as e:
            print(f"[ERRORE] Errore nel caricare i dati RAW salvati per {video_basename}: {e}. Riprovo estrazione per questo video.")
            pass

    raw_ear_data = extract_ear_from_video(video_path)
    ground_truth = load_ground_truth(json_path)

    if raw_ear_data is None or ground_truth is None:
        print(f"[ERRORE] Dati RAW (EAR o GT) mancanti per {video_basename}. Saltando.")
        return None, None

    try:
        os.makedirs(output_subdir, exist_ok=True)
        with open(ear_raw_file, 'wb') as f: pickle.dump(raw_ear_data, f)
        with open(gt_labels_file, 'wb') as f: pickle.dump(ground_truth, f)
    except Exception as e:
        print(f"[ERRORE] Errore nel salvare i dati RAW per {video_basename}: {e}")
        traceback.print_exc()
        print("[WARN] I dati RAW per questo video non sono stati salvati correttamente.")

    return raw_ear_data, ground_truth

def load_raw_data_for_video_list(video_paths, raw_data_subdir):
    raw_ear_dict_list = []
    raw_gt_dict_list = []
    print(f"[INFO] Caricamento dati RAW da '{os.path.basename(raw_data_subdir)}' per {len(video_paths)} video...")
    json_paths = [p.replace(VIDEO_EXT, JSON_EXT) for p in video_paths]

    for video_path, json_path in zip(video_paths, json_paths):
         raw_ear, gt = process_or_load_raw_data_for_video(video_path, json_path, raw_data_subdir)
         if raw_ear is not None and gt is not None:
              raw_ear_dict_list.append(raw_ear)
              raw_gt_dict_list.append(gt)

    print(f"[INFO] Dati RAW caricati/processati per {len(raw_ear_dict_list)} video in '{os.path.basename(raw_data_subdir)}'.")
    return raw_ear_dict_list, raw_gt_dict_list

def create_sequences_from_raw_data(raw_ear_dict_list, raw_gt_dict_list, scaler, label_encoder, seq_length):
    all_sequences_raw = []
    all_labels_str = []

    print(f"\n[INFO] Creazione sequenze con SEQ_LENGTH={seq_length} dai dati RAW...")

    total_ear_values_processed = 0
    total_sequences_created = 0

    # Processa ogni video (lista di dizionari) INDIPENDENTEMENTE per creare sequenze
    for video_index, (raw_ear_data, ground_truth) in enumerate(zip(raw_ear_dict_list, raw_gt_dict_list)):
        if raw_ear_data is None or ground_truth is None: continue # Salta video con dati RAW mancanti

        # Ottieni i frame comuni con dati validi EAR e Ground Truth
        common_frames = sorted(list(set(raw_ear_data.keys()) & set(ground_truth.keys())))

        # Filtra solo i frame con EAR non NaN e label conosciuta dall'encoder
        frame_indices_valid_ear_and_label = [
            f for f in common_frames
            if not np.isnan(raw_ear_data.get(f)) and ground_truth.get(f) in label_encoder.classes_
        ]
        frame_indices_valid_and_sorted = sorted(frame_indices_valid_ear_and_label)

        ear_values_video_filtered = [raw_ear_data.get(f) for f in frame_indices_valid_and_sorted]
        labels_video_filtered = [ground_truth.get(f) for f in frame_indices_valid_and_sorted] # Stringhe

        total_ear_values_processed += len(ear_values_video_filtered)


        # Crea sequenze RAW (non scalate) per questo video
        for i in range(len(ear_values_video_filtered) - seq_length + 1):
            sequence_raw = ear_values_video_filtered[i : i + seq_length]
            # La label per la sequenza [t-N+1...t] è lo stato del frame t
            label_str = labels_video_filtered[i + seq_length - 1] # Label del frame *finale* della sequenza
            all_sequences_raw.append(sequence_raw)
            all_labels_str.append(label_str)

        total_sequences_created += max(0, len(ear_values_video_filtered) - seq_length + 1) # Conta sequenze per video

    if not all_sequences_raw:
        print(f"[ERRORE] Nessuna sequenza (raw) di lunghezza {seq_length} creata dai dati RAW (totale {total_ear_values_processed} punti validi).")
        return None, None

    print(f"[INFO] Totale punti EAR validi e con label conosciuta processati: {total_ear_values_processed}")
    print(f"[INFO] Totale sequenze RAW create: {total_sequences_created}")

    # Scala le sequenze RAW usando lo scaler fittato
    scaled_sequences = []
    for seq_raw in all_sequences_raw:
        reshaped_seq = np.array(seq_raw).reshape(-1, 1) # (seq_length, 1)
        scaled_seq = scaler.transform(reshaped_seq) # (seq_length, 1)
        scaled_sequences.append(scaled_seq)

    X = np.array(scaled_sequences) # Shape: (num_sequences, seq_length, 1)

    try:
        y = label_encoder.transform(all_labels_str) # Shape: (num_sequences,) con indici di classe
    except ValueError as e:
         print(f"[ERRORE] Errore durante la codifica delle label delle sequenze: {e}")
         print(f"Label non codificabili: {set(all_labels_str) - set(label_encoder.classes_)}")
         return None, None

    print(f"\n[INFO] Dati sequenziali creati (scalati e codificati) con SEQ_LENGTH={seq_length}:")
    print(f"[INFO] Shape Sequenze: {X.shape}, Shape Labels: {y.shape}")

    return X, y

# --- LSTMs (PyTorch) ---
class EyeStateLSTM_128(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateLSTM_128, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

        self.name = "lstm_1_layer_128"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateLSTM_150(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateLSTM_150, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size + 32, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size + 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + 32, num_classes)

        self.name = "lstm_1_layer_150"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateLSTM_182(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateLSTM_182, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size + 64, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size + 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + 64, num_classes)

        self.name = "lstm_1_layer_182"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
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
    
class EyeStateLSTM_64(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateLSTM_64, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 64, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size - 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 64, num_classes)

        self.name = "lstm_1_layer_64"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateLSTM_32(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateLSTM_32, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 96, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size - 96)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 96, num_classes)

        self.name = "lstm_1_layer_32"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

class EyeStateDoubleLSTM_128(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateDoubleLSTM_128, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

        self.name = "lstm_2_layers_128"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateDoubleLSTM_150(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateDoubleLSTM_150, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size + 32, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size + 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + 32, num_classes)

        self.name = "lstm_2_layers_150"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateDoubleLSTM_182(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateDoubleLSTM_182, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size + 64, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size + 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + 64, num_classes)

        self.name = "lstm_2_layers_182"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateDoubleLSTM_96(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateDoubleLSTM_96, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 32, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size - 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 32, num_classes)

        self.name = "lstm_2_layers_96"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateDoubleLSTM_64(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateDoubleLSTM_64, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 64, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size - 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 64, num_classes)

        self.name = "lstm_2_layers_64"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateDoubleLSTM_32(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateDoubleLSTM_32, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 96, batch_first=True, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size - 96)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 96, num_classes)

        self.name = "lstm_2_layers_32"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out

class EyeStateTripleLSTM_128(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateTripleLSTM_128, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, num_classes)

        self.name = "lstm_3_layers_128"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateTripleLSTM_150(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateTripleLSTM_150, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size + 32, batch_first=True, num_layers=3)
        self.layer_norm = nn.LayerNorm(hidden_size + 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + 32, num_classes)

        self.name = "lstm_3_layers_150"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateTripleLSTM_182(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateTripleLSTM_182, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size + 64, batch_first=True, num_layers=3)
        self.layer_norm = nn.LayerNorm(hidden_size + 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size + 64, num_classes)

        self.name = "lstm_3_layers_182"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateTripleLSTM_96(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateTripleLSTM_96, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 32, batch_first=True, num_layers=3)
        self.layer_norm = nn.LayerNorm(hidden_size - 32)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 32, num_classes)

        self.name = "lstm_3_layers_96"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateTripleLSTM_64(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateTripleLSTM_64, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 64, batch_first=True, num_layers=3)
        self.layer_norm = nn.LayerNorm(hidden_size - 64)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 64, num_classes)

        self.name = "lstm_3_layers_64"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
class EyeStateTripleLSTM_32(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EyeStateTripleLSTM_32, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size - 96, batch_first=True, num_layers=3)
        self.layer_norm = nn.LayerNorm(hidden_size - 96)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size - 96, num_classes)

        self.name = "lstm_3_layers_32"

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out
    
# --- Funzione di Training ---
def train_model_pytorch(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    print("\n[INFO] Inizio training modello (PyTorch)...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10

    if len(train_loader.dataset) == 0:
        print("[ERRORE] Dataset di training vuoto. Impossibile addestrare.")
        return model

    val_available = len(val_loader.dataset) > 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # epoch_loss = running_loss / len(train_loader.dataset)
        # epoch_acc = 100 * train_correct / train_total

        if val_available:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * sequences.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = 100 * val_correct / val_total

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_acc:.5f}%")

            if epoch_val_loss < best_val_loss:
                torch.save(model.state_dict(), MODEL_FILENAME)
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Miglior val loss: {best_val_loss:.4f}, miglior val acc: {epoch_val_acc:.5f}%")
                print("Early stopping attivato.")
                break
        else:
            print("[WARN] Dataset di validation vuoto. Saltando valutazione e early stopping in training.")
            pass

    print("[INFO] Addestramento completato.")

    # Carica i pesi migliori alla fine dell'training se sono stati salvati
    if os.path.exists(MODEL_FILENAME):
        print(f"Caricamento pesi del modello migliore da {MODEL_FILENAME}")
        try:
            model.load_state_dict(torch.load(MODEL_FILENAME, map_location=device))
        except Exception as e:
            print(f"[ERRORE] Errore nel ricaricare il modello migliore salvato: {e}")
            pass
    else:
        print(f"[WARN] File modello {MODEL_FILENAME} non trovato. Modello non addestrato o salvato.")

    return model

# --- Funzione di Inferenza (PyTorch) usando dati RAW pre-estratti ---
def run_inference_on_video_data(raw_ear_data, ground_truth, model, scaler, label_encoder, seq_length, device):
    """
    Esegui l'inferenza su dati RAW (EAR e GT) pre-estratti per un singolo video.
    Scala, crea sequenze con SEQ_LENGTH attuale e passa al modello.
    Ritorna le predizioni {frame_index: label_string}.
    """
    if raw_ear_data is None or ground_truth is None:
         print("  [WARN] Dati RAW mancanti per inferenza.")
         return {}

    model.eval()
    model.to(device)

    predictions = {}

    common_frames = sorted(list(set(raw_ear_data.keys()) & set(ground_truth.keys())))
    valid_frames_for_seq = [
        f for f in common_frames
        if not np.isnan(raw_ear_data.get(f)) and ground_truth.get(f) in label_encoder.classes_
    ]
    valid_frames_for_seq.sort()

    ear_values_raw_filtered = [raw_ear_data.get(f) for f in valid_frames_for_seq]

    if not ear_values_raw_filtered:
        print("  [WARN] Nessun valore EAR valido e con label conosciuta nei dati RAW. Saltando inferenza per questo video.")
        return predictions

    if len(ear_values_raw_filtered) < 1:
         print("  [WARN] Meno di 1 punto EAR valido per la scalatura. Saltando inferenza.")
         return predictions

    try:
        scaled_ear_values = scaler.transform(np.array(ear_values_raw_filtered).reshape(-1, 1)) # Shape (num_valid_frames, 1)
    except Exception as e:
         print(f"  [ERRORE] Errore durante la scalatura dei dati RAW per inferenza: {e}. Saltando.")
         traceback.print_exc()
         return predictions

    sequences = []
    sequence_frame_indices = []

    if len(scaled_ear_values) < seq_length:
        print(f"  [WARN] Meno punti EAR validi ({len(scaled_ear_values)}) della SEQ_LENGTH ({seq_length}). Nessuna sequenza creata. Saltando inferenza.")
        return predictions

    for i in range(len(scaled_ear_values) - seq_length + 1):
        sequence = scaled_ear_values[i : i + seq_length] # Shape (seq_length, 1)
        sequences.append(sequence)
        original_frame_index = valid_frames_for_seq[i + seq_length - 1]
        sequence_frame_indices.append(original_frame_index)

    if not sequences:
        print(f"  [WARN] Nessuna sequenza di lunghezza {seq_length} creata. Saltando inferenza.")
        return predictions

    input_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(device) # Shape (num_sequences, seq_length, 1)

    inference_start_time = time.time()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_indices_tensor = torch.max(outputs.data, 1)
        predicted_indices = predicted_indices_tensor.cpu().numpy()
    inference_end_time = time.time()
    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
    print(f"  [INFO] Tempo inferenza rete ({len(sequences)} sequenze): {inference_duration_ms:.5f} ms, {inference_duration_ms/len(sequences):.5f} ms/seq")

    predicted_labels_str = label_encoder.inverse_transform(predicted_indices)

    for i in range(len(sequence_frame_indices)):
        predictions[sequence_frame_indices[i]] = predicted_labels_str[i]

    return predictions

# --- Funzione Valutazione (accetta GT dict) ---
def evaluate_predictions(predictions, ground_truth, label_encoder):
    df_pred = pd.DataFrame(list(predictions.items()), columns=['frame', 'eye_state_pred']).set_index('frame')
    df_gt = pd.DataFrame(list(ground_truth.items()), columns=['frame', 'eye_state_gt']).set_index('frame')

    df_merged = df_pred.join(df_gt, how='inner')

    valid_labels_str = list(label_encoder.classes_)

    df_merged_valid = df_merged[df_merged['eye_state_pred'].isin(valid_labels_str) & df_merged['eye_state_gt'].isin(valid_labels_str)].copy()

    if df_merged_valid.empty:
            print("  [WARN] Nessun frame corrispondente tra predizioni valide e ground truth valido per la valutazione.")
            return

    y_true = df_merged_valid["eye_state_gt"]
    y_pred = df_merged_valid["eye_state_pred"]

    print("\n  --- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=valid_labels_str, zero_division=0))

    print("\n  --- Confusion Matrix ---")
    try:
            cm = confusion_matrix(y_true, y_pred, labels=valid_labels_str)
            header_len = max(len(l) for l in valid_labels_str) if valid_labels_str else 5
            print(" " * (header_len + 2) + " ".join([f"{l:<5}" for l in valid_labels_str]) + " (Predicted)")
            print("-" * (len(valid_labels_str) * 6 + header_len + 4))
            for i, label_true in enumerate(valid_labels_str):
                row_str = " ".join([f"{cm[i][j]:<5}" if j < len(cm[i]) else f"{0:<5}" for j in range(len(valid_labels_str))])
                print(f"{label_true:<{header_len}} | " + row_str)
            print("(Actual)")
    except Exception as e:
            print(f"  Errore nella creazione/stampa della matrice di confusione: {e}")

# --- Main Execution (PyTorch) ---
if __name__ == "__main__":
    # --- Parametri Addestramento ---
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 24
    NUM_EPOCHS = 100

    print("="*30 + " FASE 1: PREPARAZIONE/CARICAMENTO DATI RAW " + "="*30)

    train_video_paths = []
    test_video_paths = []

    print("[INFO] Marker dati RAW non trovato o caricamento liste fallito. Estrazione RAW da tutti i video MP4...")

    if len(VIDEO_PATHS) != len(JSON_PATHS):
        print("[ERRORE] Il numero di video e file JSON non corrisponde!")
        sys.exit(1)

    # --- Split dei VIDEO in training POOL e TEST set finale ---
    video_indices = list(range(len(VIDEO_PATHS)))
    train_pool_video_indices, test_video_indices = train_test_split(
        video_indices, test_size=TEST_SIZE_VIDEOS, random_state=RANDOM_STATE
    )

    if not train_pool_video_indices:
        print("[ERRORE] Nessun video assegnato al pool di training/validation. Aumenta il numero totale di video o riduci TEST_SIZE_VIDEOS.")
        sys.exit(1)

    train_video_paths = [VIDEO_PATHS[i] for i in train_pool_video_indices] # Popola train_video_paths
    test_video_paths = [VIDEO_PATHS[i] for i in test_video_indices]        # Popola test_video_paths

    print(f"\n[INFO] Divisione dei video completata:")
    print(f"[INFO] Video per Training/Validation Pool: {len(train_video_paths)}")
    print(f"[INFO] Video per Test Finale: {len(test_video_paths)}")

    # Salva le liste dei percorsi dei video
    try:
        with open(TRAIN_VIDEO_PATHS_FILE, 'wb') as f: pickle.dump(train_video_paths, f)
        with open(TEST_VIDEO_PATHS_FILE, 'wb') as f: pickle.dump(test_video_paths, f)
        print(f"[INFO] Liste video (Train Pool/Test) salvate in: {OUTPUT_DIR}/")
    except Exception as e:
        print(f"[ERRORE] Errore nel salvare le liste video: {e}")
        traceback.print_exc()


    # --- Estrai e salva i dati RAW per TUTTI i video (Train Pool e Test) ---
    print("\n[INFO] Estrazione e salvataggio dati RAW per i video del TRAINING/VALIDATION Pool...")
    train_pool_json_paths = [p.replace(VIDEO_EXT, JSON_EXT) for p in train_video_paths]
    for video_path, json_path in zip(train_video_paths, train_pool_json_paths):
        process_or_load_raw_data_for_video(video_path, json_path, TRAIN_RAW_DATA_DIR)

    print("\n[INFO] Estrazione e salvataggio dati RAW per i video del TEST set...")
    test_json_paths_aligned = [p.replace(VIDEO_EXT, JSON_EXT) for p in test_video_paths]
    for video_path, json_path in zip(test_video_paths, test_json_paths_aligned):
        process_or_load_raw_data_for_video(video_path, json_path, TEST_RAW_DATA_DIR)

    # --- FASE 2: PREPARAZIONE DATI SEQUENZIALI SCALATI (TRAINING e VALIDATION) ---
    print("\n" + "="*30 + " FASE 2: PREPARAZIONE DATI SEQUENZIALI SCALATI (PyTorch) " + "="*30)

    # --- Suddividi i video del TRAINING POOL in Training effettivo e Validation ---
    if not train_video_paths:
        print("[ERRORE] Liste video di training pool non disponibili. Impossibile procedere con FASE 2.")
        sys.exit(1)

    print(f"[INFO] Suddivisione dei {len(train_video_paths)} video del Training Pool in Training effettivo e Validation...")
    actual_train_video_paths, validation_video_paths = train_test_split(
        train_video_paths, test_size=VALIDATION_SPLIT_VIDEOS, random_state=RANDOM_STATE # Random state per riproducibilità dello split T/V video
    )

    print(f"[INFO] Video per Training effettivo: {len(actual_train_video_paths)}")
    print(f"[INFO] Video per Validation: {len(validation_video_paths)}")

    if not actual_train_video_paths:
        print("[ERRORE] Nessun video assegnato al set di training effettivo. Impossibile addestrare.")
        sys.exit(1)

    # --- Carica i dati RAW solo per i video di TRAINING effettivo ---
    actual_train_raw_ear_dict_list, actual_train_raw_gt_dict_list = load_raw_data_for_video_list(
        actual_train_video_paths, TRAIN_RAW_DATA_DIR
    )
    if not actual_train_raw_ear_dict_list:
        print("[ERRORE] Nessun dato RAW valido caricato per il set di training effettivo. Impossibile procedere.")
        sys.exit(1)


    # --- Aggrega i dati RAW del TRAINING effettivo per fittare Scaler ed Encoder ---
    print("[INFO] Aggregazione dati RAW dal set di Training effettivo per fittare Scaler/Encoder...")
    all_raw_ear_values_for_scaler_fit = []
    all_raw_gt_labels_for_encoder_fit = []

    for raw_ear_data, ground_truth in zip(actual_train_raw_ear_dict_list, actual_train_raw_gt_dict_list):
        if raw_ear_data is None or ground_truth is None: continue

        # Raccogli i valori grezzi e label per fittare scaler/encoder
        all_raw_ear_values_for_scaler_fit.extend([ear for ear in raw_ear_data.values() if not np.isnan(ear)])
        ear_frames = {f for f, ear in raw_ear_data.items() if not np.isnan(ear)}
        all_raw_gt_labels_for_encoder_fit.extend([label for f, label in ground_truth.items() if f in ear_frames])

    if not all_raw_ear_values_for_scaler_fit:
        print("[ERRORE] Nessun valore EAR valido aggregato dal set di Training effettivo per fittare Scaler/Encoder.")
        sys.exit(1)

    # --- Fitta e salva Scaler ed Encoder (sui dati RAW aggregati del TRAINING effettivo) ---
    scaler = StandardScaler()
    scaler.fit(np.array(all_raw_ear_values_for_scaler_fit).reshape(-1, 1))
    print(f"[INFO] Scaler fittato su {len(all_raw_ear_values_for_scaler_fit)} valori EAR grezzi dal set di Training effettivo.")
    try:
        with open(SCALER_FILENAME, 'wb') as f: pickle.dump(scaler, f)
        print(f"[INFO] Scaler salvato in: {SCALER_FILENAME}")
    except Exception as e:
        print(f"[ERRORE] Errore nel salvare lo scaler: {e}")
        traceback.print_exc()

    label_encoder = LabelEncoder()
    expected_labels = ['open', 'close', 'opening', 'closing']
    try:
        unique_actual_labels = sorted(list(set(all_raw_gt_labels_for_encoder_fit)))
        all_labels_to_fit = sorted(list(set(expected_labels + unique_actual_labels)))
        if not all_labels_to_fit:
            print("[ERRORE] Nessuna label disponibile (né attesa né trovata) per fittare l'encoder.")
            sys.exit(1)
        label_encoder.fit(all_labels_to_fit)
        print(f"[INFO] LabelEncoder fittato con classi: {label_encoder.classes_}")
        try:
            with open(ENCODER_FILENAME, 'wb') as f: pickle.dump(label_encoder, f)
            print(f"[INFO] LabelEncoder salvato in: {ENCODER_FILENAME}")
        except Exception as e:
            print(f"[ERRORE] Errore nel salvare il LabelEncoder: {e}")
            traceback.print_exc()

    except ValueError as e:
        print(f"[ERRORE] Errore durante il fit del LabelEncoder: {e}")
        sys.exit(1)

    X_train, y_train = create_sequences_from_raw_data(
        actual_train_raw_ear_dict_list, actual_train_raw_gt_dict_list, scaler, label_encoder, SEQ_LENGTH
    )

    if X_train is None or len(X_train) == 0:
        print("[ERRORE] Creazione sequenze training effettive fallita o set vuoto. Uscita.")
        sys.exit(1)

    X_val, y_val = None, None
    if validation_video_paths:
        validation_raw_ear_dict_list, validation_raw_gt_dict_list = load_raw_data_for_video_list(
            validation_video_paths, TRAIN_RAW_DATA_DIR
        )
        if validation_raw_ear_dict_list:
            X_val, y_val = create_sequences_from_raw_data(
                validation_raw_ear_dict_list, validation_raw_gt_dict_list, scaler, label_encoder, SEQ_LENGTH
            )
            if X_val is None:
                print("[WARN] Creazione sequenze validation fallita. Procedo senza set di validation.")
                X_val, y_val = None, None
        else:
            print("[WARN] Nessun dato RAW valido caricato per il set di validation. Procedo senza set di validation.")

    # --- Crea PyTorch Datasets e DataLoaders ---
    train_dataset = EarDataset(X_train, y_train)
    val_dataset = EarDataset(X_val, y_val) if X_val is not None else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset else None

    num_classes = len(label_encoder.classes_)

    print("\n" + "="*30 + " FASE 3: COSTRUZIONE E TRAINING/CARICAMENTO MODELLO (PyTorch) " + "="*30)

    # --- Scegli il tipo di modello e crea l'istanza ---
    available_models = {
        "lstm_1_layer_32": EyeStateLSTM_32,
        "lstm_1_layer_64": EyeStateLSTM_64,
        "lstm_1_layer_96": EyeStateLSTM_96,
        "lstm_1_layer_128": EyeStateLSTM_128,
        "lstm_1_layer_150": EyeStateLSTM_150,
        "lstm_1_layer_182": EyeStateLSTM_182,
        "lstm_2_layers_32": EyeStateDoubleLSTM_32,
        "lstm_2_layers_64": EyeStateDoubleLSTM_64,
        "lstm_2_layers_96": EyeStateDoubleLSTM_96,
        "lstm_2_layers_128": EyeStateDoubleLSTM_128,
        "lstm_2_layers_150": EyeStateDoubleLSTM_150,
        "lstm_2_layers_182": EyeStateDoubleLSTM_182,
        "lstm_3_layers_32": EyeStateTripleLSTM_32,
        "lstm_3_layers_64": EyeStateTripleLSTM_64,
        "lstm_3_layers_96": EyeStateTripleLSTM_96,
        "lstm_3_layers_128": EyeStateTripleLSTM_128,
        "lstm_3_layers_150": EyeStateTripleLSTM_150,
        "lstm_3_layers_182": EyeStateTripleLSTM_182,
    }

    print("\n[INFO] Modelli disponibili:")
    for i, model_name in enumerate(available_models.keys()):
        print(f"{i+1}. {model_name}")

    MODEL_TYPE = None
    while MODEL_TYPE not in available_models:
        try:
            user_input = input("Scegli il modello (inserisci il numero corrispondente): ")
            MODEL_TYPE = list(available_models.keys())[int(user_input) - 1]
        except (ValueError, IndexError):
            print("[ERRORE] Input non valido. Inserisci un numero tra 1 e", len(available_models))

    if MODEL_TYPE not in available_models:
        print(f"[ERRORE] Tipo di modello '{MODEL_TYPE}' non definito.")
        sys.exit(1)

    model_class = available_models[MODEL_TYPE]

    try:
        if "lstm" in MODEL_TYPE or "gru" in MODEL_TYPE:
             model = model_class(NUM_FEATURES, HIDDEN_SIZE_1, num_classes).to(device)
        else:
             print(f"[ERRORE] Logica per inizializzazione modello '{MODEL_TYPE}' non implementata.")
             sys.exit(1)

    except TypeError as e:
         print(f"[ERRORE] Errore nell'inizializzazione del modello '{MODEL_TYPE}': {e}. Controlla i parametri __init__.")
         sys.exit(1)

    print(f"[INFO] Usando modello: {type(model).__name__} (Nome per file: {model.name})")

    # --- Imposta il nome del file modello basato sull'attributo 'name' ---
    MODEL_FILENAME = os.path.join(OUTPUT_DIR, model.name + ".pth")

    # Check if model needs training or loading
    train_needed = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if train_needed:
        if len(train_loader.dataset) == 0:
            print("[ERRORE] Dataset di training vuoto. Impossibile addestrare.")
        elif val_loader is None and VALIDATION_SPLIT_VIDEOS > 0:
            print("[WARN] Validation split video > 0 ma nessun dato di validation valido creato. Procedo senza validation.")
            # Pass None for val_loader to train_model_pytorch
            model = train_model_pytorch(model, train_loader, None, criterion, optimizer, NUM_EPOCHS, device)
        else:
            print("[INFO] Inizio training del modello.")
            model = train_model_pytorch(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)

        if model is None:
            print("[ERRORE] Addestramento modello fallito. Uscita.")
            sys.exit(1)
    else:
        model.eval()
        print("[INFO] Saltando la fase di training (modello caricato).")

    # --- FASE 4: INFERENZA E VALUTAZIONE (TEST SET) ---
    print("\n" + "="*30 + " FASE 4: INFERENZA E VALUTAZIONE (TEST SET) " + "="*30)

    if not test_video_paths:
        print("[INFO] Nessun video di test definito. Saltando la fase di valutazione finale.")
    elif scaler is None or label_encoder is None:
        print("[ERRORE] Scaler o Encoder non disponibili. Impossibile eseguire l'inferenza sui dati di test.")
    elif model is None:
        print("[ERRORE] Modello non disponibile (non addestrato o caricato). Impossibile eseguire l'inferenza.")
    else:
        print(f"[INFO] Avvio valutazione sui {len(test_video_paths)} video del TEST set (usando dati RAW pre-estratti).")

        test_json_paths_aligned = [p.replace(VIDEO_EXT, JSON_EXT) for p in test_video_paths]

        if len(test_video_paths) != len(test_json_paths_aligned):
            print("[ERRORE] Disallineamento percorsi video/json di test. Impossibile valutare.")
        else:
            all_test_y_true = []
            all_test_y_pred = []
            # --------------------------------------------------------------------------

            for video_path, json_path in zip(test_video_paths, test_json_paths_aligned):
                print(f"\n--- Elaborazione Video Test: {os.path.basename(video_path)} ---") # Cambiato output per chiarezza
                raw_ear_data, ground_truth = process_or_load_raw_data_for_video(video_path, json_path, TEST_RAW_DATA_DIR)
                
                if raw_ear_data is not None and ground_truth is not None:
                    predictions = run_inference_on_video_data(
                        raw_ear_data, ground_truth, model, scaler, label_encoder, SEQ_LENGTH, device # Passa l'attuale SEQ_LENGTH
                    )
                    if predictions:
                        df_pred = pd.DataFrame(list(predictions.items()), columns=['frame', 'eye_state_pred']).set_index('frame')
                        df_gt = pd.DataFrame(list(ground_truth.items()), columns=['frame', 'eye_state_gt']).set_index('frame')

                        df_merged = df_pred.join(df_gt, how='inner')

                        valid_labels_str = list(label_encoder.classes_)

                        df_merged_valid = df_merged[df_merged['eye_state_pred'].isin(valid_labels_str) & df_merged['eye_state_gt'].isin(valid_labels_str)].copy()

                        if df_merged_valid.empty:
                            print("  [WARN] Nessun frame corrispondente tra predizioni valide e ground truth valido per questo video.")
                        else:
                            all_test_y_true.extend(df_merged_valid["eye_state_gt"].tolist())
                            all_test_y_pred.extend(df_merged_valid["eye_state_pred"].tolist())
                            print(f"  [INFO] Aggiunti {len(df_merged_valid)} frame validi per la valutazione complessiva da questo video.")

                    else:
                        print(f"  [WARN] Nessuna predizione generata per {os.path.basename(video_path)}. Saltando aggiunta alla valutazione complessiva.")
                else:
                    print(f"[WARN] Dati RAW per il video {os.path.basename(video_path)} non disponibili o caricamento fallito. Saltando elaborazione.")

            print("\n" + "="*30 + " RISULTATI COMPLESSIVI TEST SET " + "="*30)

            if not all_test_y_true:
                print("[INFO] Nessun dato valido raccolto dai video di test per la valutazione complessiva.")
            else:
                eval_labels = list(label_encoder.classes_)

                print("\n  --- Classification Report Complessivo ---")
                print(classification_report(all_test_y_true, all_test_y_pred, labels=eval_labels, zero_division=0))

                print("\n  --- Confusion Matrix Complessiva ---")
                try:
                    cm = confusion_matrix(all_test_y_true, all_test_y_pred, labels=eval_labels)
                    header_len = max(len(l) for l in eval_labels) if eval_labels else 5
                    print(" " * (header_len + 2) + " ".join([f"{l:<5}" for l in eval_labels]) + " (Predicted)")
                    print("-" * (len(eval_labels) * 6 + header_len + 4))
                    for i, label_true in enumerate(eval_labels):
                        row_str = " ".join([f"{cm[i][j]:<5}" if j < len(cm[i]) else f"{0:<5}" for j in range(len(eval_labels))])
                        print(f"{label_true:<{header_len}} | " + row_str)
                    print("(Actual)")
                except Exception as e:
                    print(f"  Errore nella creazione/stampa della matrice di confusione complessiva: {e}")

    print("="*75)
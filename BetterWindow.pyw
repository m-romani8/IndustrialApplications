import tkinter as tk
from tkinter import font
import time
import socket
import threading

# Imposta dimensioni della finestra
WIDTH, HEIGHT = 400, 300

def update_temperature(change):
    """Aggiorna la temperatura e avvia l'animazione"""
    global temperature
    temperature += change
    animate_change("Heating" if change > 0 else "Cooling")

def animate_change(action):
    """Animazione per il cambiamento della temperatura"""
    vector = [action, action+".", action+"..", action+"..."]
    start = time.time()
    end = 0
    i = 0
    bg_color = 'red' if action == "Heating" else 'blue'
    
    while end - start < 1:
        root.configure(bg=bg_color)
        end = time.time()
        temp_label.config(text=str(temperature)+"°C", bg=bg_color)
        status_label.config(text=vector[i % 4], bg=bg_color)
        root.update()
        i += 1
        time.sleep(0.15)

    root.configure(bg='green')
    status_label.config(text="", bg="green")
    temp_label.config(bg="green")

def receive_alert():
    """Riceve i messaggi dal Drowsiness Detector senza bloccare Tkinter"""
    while True:
        try:
            msg = client.recv(1024).decode()
            if msg == "DROWSY":
                print("🚨 Ricevuto segnale di allarme! Aumento temperatura!")
                root.after(0, update_temperature, -1)  # Diminuisce la temperatura di 1°C
                time.sleep(0.2)
        except:
            break  # Se la connessione si chiude, esce dal loop

# Crea la finestra principale
root = tk.Tk()
root.title("Car Temperature")
root.geometry(f"{WIDTH}x{HEIGHT}")
root.configure(bg='green')

# Imposta il font
temp_font = font.Font(family="Helvetica", size=80)
status_font = font.Font(family="Helvetica", size=50)

# Variabile per la temperatura
temperature = 18

# Crea e posiziona gli elementi
temp_label = tk.Label(root, text=str(temperature)+"°C", font=temp_font, fg="white", bg="green")
temp_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

status_label = tk.Label(root, text="", font=status_font, fg="white", bg="green")
status_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

# Bind dei tasti per test manuale
root.bind("0", lambda event: update_temperature(1))
root.bind("1", lambda event: update_temperature(-1))

# Connessione al drowsiness detector
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 12345))

# Avvia un thread per ricevere i dati senza bloccare l'interfaccia
threading.Thread(target=receive_alert, daemon=True).start()

# Avvia il loop principale di Tkinter
root.mainloop()

# Chiude la connessione quando Tkinter si chiude
client.close()

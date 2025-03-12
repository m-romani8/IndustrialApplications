import tkinter as tk
from tkinter import font
import time

# Imposta dimensioni della finestra
WIDTH, HEIGHT = 400, 300

#Hi! this is a test

def update_temperature(change):
    global temperature
    temperature += change
    animate_change("Heating" if change > 0 else "Cooling")

def animate_change(action):
    vector = [action, action+".", action+"..", action+"..."]
    start = time.time()
    end = 0
    i = 0
    bg_color = 'red' if action == "Heating" else 'blue'
    while end - start < 1:
        root.configure(bg=bg_color)
        end = time.time()
        temp_label.config(text=str(temperature)+"°C",bg=bg_color)
        status_label.config(text=vector[i % 4],bg=bg_color)
        root.update()
        i += 1
        time.sleep(0.15)

    root.configure(bg='green')
    status_label.config(text="",bg="green")
    temp_label.config(bg="green")
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
temp_label = tk.Label(root, text=str(temperature)+"°C", font=temp_font, fg="white",bg="green")
temp_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

status_label = tk.Label(root, text="", font=status_font, fg="white",bg="green")
status_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

# Bind dei tasti
root.bind("0", lambda event: update_temperature(1))
root.bind("1", lambda event: update_temperature(-1))

# Avvia il loop principale
root.mainloop()

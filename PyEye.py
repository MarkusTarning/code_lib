# -*- coding: utf-8 -*-
"""
__PyEye__(Object identification with speech support)
Information regarding the program:
The program uses the available camera to predict the objects 
that are in the camera's vicinity.The program uses: YOLOv10n for prediction.

Created by: Markus Tärning 2025

@author: markus.tarning@student.nbi-handelsakademin.se

"""

import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
import pyttsx3

# Ladda YOLOv10n-modellen
model = YOLO("yolov10n.pt") # yolov8n.pt

# Initiera talsyntes
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Globala variabler
running = False  # Styr kameran
interval = 0.2  # Tidsintervall mellan prediktioner
confidence_threshold = 0.1  # Minsta sannolikhet för prediktion
speak_enabled = False  # Styr talstödet
no_repeat = False  # Styr om objekt ska återupprepas
identified_objects = set()  # Set för identifierade objekt

# Starta kameran
cap = cv2.VideoCapture(0)

# Skapa huvudfönster
root = tk.Tk()
root.title("PyEye")
root.geometry("800x600")

# Frame för GUI-komponenter
control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

# Skapa en canvas för att visa videoflödet
canvas = tk.Label(root)
canvas.pack()

# Funktion för att uppdatera videoflödet
def update_frame():
    global running, identified_objects
    last_spoken = ""

    while running:
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            results = model(frame)
            detected_objects = set()

            for result in results:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = result.names[int(box.cls[0])]
                        
                        if not no_repeat or label not in identified_objects:
                            detected_objects.add(label)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            identified_objects.update(detected_objects)

            if speak_enabled and detected_objects:
                spoken_text = ", ".join(detected_objects)
                if spoken_text != last_spoken:
                    engine.say(spoken_text)
                    engine.runAndWait()
                    last_spoken = spoken_text

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.configure(image=imgtk)

            elapsed_time = time.time() - start_time
            time.sleep(max(0, interval - elapsed_time))

# Starta kameran
def start_camera():
    global running
    if not running:
        running = True
        threading.Thread(target=update_frame, daemon=True).start()

# Pausa kameran
def pause_camera():
    global running
    running = False

# Justera tidsintervall
def update_interval(value):
    global interval
    interval = float(value)

# Justera sannolikhetsgräns
def update_confidence(value):
    global confidence_threshold
    confidence_threshold = float(value)

# Aktivera/inaktivera talstöd
def toggle_speech():
    global speak_enabled
    speak_enabled = speech_var.get()

# Aktivera/inaktivera av upprepade objekt
def toggle_repeat():
    global no_repeat
    no_repeat = no_repeat_var.get()

# Sparar objektlista till fil
def save_object_list():
    with open("identified_objects.txt", "w") as file:
        file.write("\n".join(identified_objects))
    print("Lista sparad till identified_objects.txt")

# Knappar
btn_start = ttk.Button(control_frame, text="Starta Kamera", command=start_camera)
btn_start.grid(row=0, column=0, padx=5, pady=5)

btn_pause = ttk.Button(control_frame, text="Pausa Kamera", command=pause_camera)
btn_pause.grid(row=0, column=1, padx=5, pady=5)

btn_save = ttk.Button(control_frame, text="Spara Objektlista", command=save_object_list)
btn_save.grid(row=0, column=2, padx=5, pady=5)

# Reglage för tidsintervall
tk.Label(control_frame, text="Tidsintervall (sek)").grid(row=1, column=0, padx=5)
scale_interval = tk.Scale(control_frame, from_=0.2, to=5, resolution=0.2, orient="horizontal", command=update_interval)
scale_interval.set(interval)
scale_interval.grid(row=1, column=1, columnspan=2, sticky="we", padx=5)

# Reglage för sannolikhetsgräns
tk.Label(control_frame, text="Sannolikhetströskel").grid(row=2, column=0, padx=5)
scale_confidence = tk.Scale(control_frame, from_=0.1, to=1.0, resolution=0.02, orient="horizontal", command=update_confidence)
scale_confidence.set(confidence_threshold)
scale_confidence.grid(row=2, column=1, columnspan=2, sticky="we", padx=5)

# Checkbox för talstöd
speech_var = tk.BooleanVar()
speech_checkbox = ttk.Checkbutton(control_frame, text="Aktivera talstöd", variable=speech_var, command=toggle_speech)
speech_checkbox.grid(row=3, column=0, pady=5)

# Checkbox för att inte upprepa objekt
no_repeat_var = tk.BooleanVar()
repeat_checkbox = ttk.Checkbutton(control_frame, text="Undvik upprepade objekt", variable=no_repeat_var, command=toggle_repeat)
repeat_checkbox.grid(row=3, column=1, columnspan=2, pady=5, padx=20)


# Tkinter GUI
root.mainloop()

# Stänger kameran när programmet avslutas
cap.release()
cv2.destroyAllWindows()
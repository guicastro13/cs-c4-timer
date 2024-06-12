import numpy as np
import cv2
import tensorflow as tf
from mss import mss
import tkinter as tk
import time

x, y = 920, 5
width, height = 80, 35

# Função para capturar a tela principal
def capture_screen():
    with mss() as sct:
        monitor = sct.monitors[1]  # Captura da tela principal
        frame = np.array(sct.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
# Carregar o modelo treinado
model = tf.keras.models.load_model('c4_detection_model.h5')

# Função para detectar o ícone na região de interesse
def detect_icon(frame):
    roi = frame[y:y+height, x:x+width]
    roi_resized = cv2.resize(roi, (50, 50))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 50, 50, 3))
    prediction = model.predict(roi_reshaped)
    return prediction > 0.5

class CountdownWindow:
    def __init__(self, countdown_time=41):
        # Configuração inicial da janela
        self.window = tk.Tk()
        self.window.overrideredirect(True)  # Remove a barra de título
        self.window.attributes('-topmost', True)  # Mantém a janela no topo
        self.window.geometry("+50+50")  # Posição da janela na tela
        self.window.wm_attributes('-transparentcolor', self.window.cget('bg'))  # Torna o fundo transparente

        # Label para mostrar o contador
        self.label = tk.Label(self.window, text=str(countdown_time), font=("Arial", 50))
        self.label.pack()

        # Variável para a contagem regressiva
        self.countdown_time = countdown_time

    def update_countdown(self):
        if self.countdown_time > 0:
            print("Contador: ", self.countdown_time)
            self.countdown_time -= 1
            self.label.config(text=str(self.countdown_time))
            self.window.after(1000, self.update_countdown)
        else:
            print("Contador finalizado e mudado para 41 segundos.")
            self.countdown_time = 41
            self.label.config(text=str(" "))
            self.window.quit()
  
    def start(self):
        self.update_countdown()
        self.window.mainloop()

    def get_current_countdown(self):
        return self.countdown_time

# Função para criar e controlar a janela e o contador
def create_and_start_countdown():
    countdown_window = CountdownWindow()
    countdown_window.start()
    return countdown_window


activation_count = 0

while True:
    frame = capture_screen()

    if not detect_icon(frame):
        print("C4!")
        activation_count += 1
        if activation_count == 5:
            print("O ícone foi detectado e o contador foi iniciado.")
            activation_count = 0
            countdown_window = create_and_start_countdown()

cv2.destroyAllWindows()
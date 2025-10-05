import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf
from mss import mss
import tkinter as tk


X, Y = 920, 5
WIDTH, HEIGHT = 80, 35
ROI_MONITOR = {"left": X, "top": Y, "width": WIDTH, "height": HEIGHT}
TARGET_SIZE = (50, 50)
THRESHOLD = 0.5
SLEEP_BETWEEN_FRAMES = 0.01
NORMALIZATION_FACTOR = 1.0 / 255.0


@dataclass
class DetectionResources:
    interpreter: Callable[[tf.Tensor], tf.Tensor]
    grabber: mss
    input_buffer: np.ndarray


class IconDetector:
    def __init__(self, model_path: str = "c4_detection_model.h5") -> None:
        self._resources = self._build_resources(model_path)

    @staticmethod
    def _build_resources(model_path: str) -> DetectionResources:
        model = tf.keras.models.load_model(model_path)

        @tf.function
        def infer(batch: tf.Tensor) -> tf.Tensor:
            return model(batch, training=False)

        grabber = mss()
        input_buffer = np.zeros((1, TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float32)
        return DetectionResources(interpreter=infer, grabber=grabber, input_buffer=input_buffer)

    def detect_icon(self) -> bool:
        raw = self._resources.grabber.grab(ROI_MONITOR)
        frame = np.frombuffer(raw.rgb, dtype=np.uint8)
        frame = frame.reshape((raw.height, raw.width, 3))
        resized = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        self._resources.input_buffer[0] = resized * NORMALIZATION_FACTOR
        prediction = self._resources.interpreter(self._resources.input_buffer)
        score = float(prediction.numpy()[0])
        return score > THRESHOLD


class CountdownWindow:
    def __init__(self, countdown_time: int = 41) -> None:
        self.window = tk.Tk()
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.geometry("+50+50")
        self.window.wm_attributes('-transparentcolor', self.window.cget('bg'))

        self.label = tk.Label(self.window, text=str(countdown_time), font=("Arial", 50))
        self.label.pack()

        self.countdown_time = countdown_time

    def update_countdown(self) -> None:
        if self.countdown_time > 0:
            print("Contador:", self.countdown_time)
            self.countdown_time -= 1
            self.label.config(text=str(self.countdown_time))
            self.window.after(1000, self.update_countdown)
        else:
            print("Contador finalizado e mudado para 41 segundos.")
            self.countdown_time = 41
            self.label.config(text=" ")
            self.window.quit()

    def start(self) -> None:
        self.update_countdown()
        self.window.mainloop()

    def get_current_countdown(self) -> int:
        return self.countdown_time


def create_and_start_countdown() -> CountdownWindow:
    countdown_window = CountdownWindow()
    countdown_window.start()
    return countdown_window


def main() -> None:
    detector = IconDetector()
    activation_count = 0

    try:
        while True:
            has_icon = detector.detect_icon()
            if not has_icon:
                print("C4!")
                activation_count += 1
                if activation_count == 5:
                    print("O icone foi detectado e o contador foi iniciado.")
                    activation_count = 0
                    create_and_start_countdown()
            else:
                activation_count = 0

            time.sleep(SLEEP_BETWEEN_FRAMES)
    except KeyboardInterrupt:
        print("Encerrando deteccao.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

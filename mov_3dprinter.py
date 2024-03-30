import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Inicialización y configuración de la comunicación serial
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Espera para estabilizar la conexión

# Funciones de utilidad
def enviar_comando_gcode(comando):
    print(f"Enviando comando: {comando}")  # Depuración
    ser.write(f"{comando}\n".encode())
    ser.flush()

# Preparación inicial de la impresora
def inicializar_impresora():
    enviar_comando_gcode("M104 S200")  # Precalentar hotend a 200°C
    enviar_comando_gcode("M140 S60")   # Precalentar cama a 60°C
    enviar_comando_gcode("G28")        # Home all axes
    enviar_comando_gcode("G1 Z0.2 F600")  # Elevar en el eje Z 0.2 mm

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Inicializar la impresora al inicio
inicializar_impresora()

# Variables de estado
estado_extrusion = False  # Controla si actualmente se debe extruir o no
ultimo_cambio_extrusion = time.time()  # Controla el último cambio para evitar activaciones accidentales
delay_extrusion = 1.0  # Tiempo mínimo en segundos entre cambios de estado de extrusión

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Determinar si la mano es derecha o izquierda
            handedness = results.multi_handedness[0]
            label = handedness.classification[0].label

            # Obtener la posición del pulgar y del índice
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calcular la distancia entre el pulgar y el índice
            distance_thumb_index = np.linalg.norm(np.array((thumb_tip.x, thumb_tip.y)) - np.array((index_tip.x, index_tip.y)))

            if label == 'Right':
                tiempo_actual = time.time()
                # Cambiar el estado de extrusión si se junta y separa el pulgar con el índice
                if distance_thumb_index < 0.05 and tiempo_actual - ultimo_cambio_extrusion > delay_extrusion:
                    estado_extrusion = not estado_extrusion
                    ultimo_cambio_extrusion = tiempo_actual

                x_pos = np.interp((thumb_tip.x + index_tip.x) / 2, [0, 1], [15, 200])
                y_pos = np.interp((thumb_tip.y + index_tip.y) / 2, [0, 1], [200, 15])

                # Si el estado de extrusión está activo, extruir mientras se mueve
                if estado_extrusion:
                    enviar_comando_gcode(f"G91")  # Modo relativo
                    enviar_comando_gcode(f"G1 X{x_pos:.2f} Y{y_pos:.2f} E0.1 F1000")
                    enviar_comando_gcode("G90")  # Volver a modo absoluto
                else:
                    enviar_comando_gcode(f"G1 X{x_pos:.2f} Y{y_pos:.2f} F3000")  # Solo mover sin extruir

            # Código para la mano izquierda y manejo de incremento en Z aquí...

            mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', image_rgb)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()

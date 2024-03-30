import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Inicialización de la comunicación serial con la impresora 3D
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Espera para estabilizar la conexión

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

ultimo_comando_z = 0  # Variable para rastrear el tiempo desde el último comando Z

def enviar_comando_gcode(comando):
    global ultimo_comando_z
    ahora = time.time()
    if "Z" in comando and ahora - ultimo_comando_z < 0.5:
        print("Esperando para enviar otro comando Z...")
        return
    print(f"Enviando comando: {comando}")
    ser.write(f"{comando}\n".encode())
    ser.flush()
    if "Z" in comando:
        ultimo_comando_z = ahora


def inicializar_impresora():
    enviar_comando_gcode("M104 S200")  # Precalentar hotend a 200°C
    enviar_comando_gcode("M140 S60")   # Precalentar cama a 60°C
    enviar_comando_gcode("G28")        # Home all axes
    enviar_comando_gcode("G1 Z0.2 F600")  # Elevar en el eje Z 0.2 mm

# Inicialización de la impresora 3D
inicializar_impresora()

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # Procesamiento de la imagen con MediaPipe
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dibujo del recuadro que simula la cama caliente
    cv2.rectangle(image_rgb, (50, 50), (frame_width - 50, frame_height - 50), (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            handedness = results.multi_handedness[0]
            label = handedness.classification[0].label

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            # Control de movimientos en Z para la mano izquierda
            if label == 'Left':
                distance_thumb_index = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y]))
                distance_thumb_ring = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([ring_finger_tip.x, ring_finger_tip.y]))
                if distance_thumb_index < 0.05:  # Pulgar e índice juntos
                    enviar_comando_gcode("G91")  # Modo relativo
                    enviar_comando_gcode("G1 Z0.2 F600")  # Subir 0.2mm en Z
                    enviar_comando_gcode("G90")  # Modo absoluto
                elif distance_thumb_ring < 0.05:  # Pulgar y anular juntos
                    enviar_comando_gcode("G91")
                    enviar_comando_gcode("G1 Z-0.2 F600")  # Bajar 0.2mm en Z
                    enviar_comando_gcode("G90")

            # Control de movimientos en X e Y para la mano derecha
            if label == 'Right':
                distance_thumb_index = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y]))
                if distance_thumb_index < 0.05:  # Se juntan pulgar e índice
                    x_cama = np.interp(thumb_tip.x, [0, 1], [0, 200])  # Ajuste según dimensiones de cama
                    y_cama = np.interp(thumb_tip.y, [0, 1], [200, 0])
                    enviar_comando_gcode(f"G1 X{x_cama:.2f} Y{y_cama:.2f} F3000")

            mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', image_rgb)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()

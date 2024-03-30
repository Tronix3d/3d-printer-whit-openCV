import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# Establecer conexión serial con la impresora 3D
ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # Espera para que la conexión se establezca

def enviar_comando_gcode(comando):
    print(f"Enviando comando: {comando}")
    ser.write(f"{comando}\n".encode())
    ser.flush()

# Inicializa Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_width, screen_height = 640, 480  # Asume una resolución de cámara de 640x480 para calcular la proporción

# Variables globales para almacenar el punto de referencia inicial y la extrusión
x_ref, y_ref = None, None
extrusion = 0.0

# Estado para detectar cambio de gesto de puño cerrado
puño_izquierdo_cerrado_anteriormente = False
separación_anterior = False  # Estado anterior de separación de pulgar e índice

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    puño_izquierdo_cerrado_actualmente = False  # Resetear estado para este frame
    separación_actual = False  # Estado actual de separación de pulgar e índice

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            classification = results.multi_handedness[hand_idx]
            label = classification.classification[0].label  # 'Right' o 'Left'

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance_thumb_index = np.linalg.norm(np.array((thumb_tip.x, thumb_tip.y)) - np.array((index_tip.x, index_tip.y)))
            
            # Control de ejes X e Y con extrusión proporcional (Mano Derecha)
            if label == 'Right':
                # Aquí se incluye la lógica detallada previamente para el control de los ejes X e Y
                
                cx_current = (thumb_tip.x + index_tip.x) / 2
                cy_current = (thumb_tip.y + index_tip.y) / 2
                
                if x_ref is None or y_ref is None:
                    x_ref, y_ref = cx_current, cy_current
                else:
                    dx = (cx_current - x_ref) * 220  # Mapeo proporcional
                    dy = (cy_current - y_ref) * 220
                    desplazamiento = np.sqrt(dx**2 + dy**2)
                    extrusion += desplazamiento * 0.1  # Ajuste de extrusión proporcional
                    
                    enviar_comando_gcode(f"G1 X{dx:.2f} Y{dy:.2f} E{extrusion:.2f} F1000")
                    
                    x_ref, y_ref = cx_current, cy_current  # Actualizar punto de referencia
                    
            # Control del eje Z (Mano Izquierda)
            if label == 'Left':
                if distance_thumb_index < 0.05:
                    separación_actual = True  # Detectar juntar pulgar e índice
                
                all_fingers_closed = True
                for fingertip in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
                    if hand_landmarks.landmark[fingertip].y > hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y:
                        all_fingers_closed = False
                        break
                if all_fingers_closed:
                    puño_izquierdo_cerrado_actualmente = True

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    # Comprobar cambio de estado y enviar comando de Home si es necesario
    if puño_izquierdo_cerrado_actualmente and not puño_izquierdo_cerrado_anteriormente:
        enviar_comando_gcode("G28")  # Home all axes
        
    if separación_actual and not separación_anterior:
        enviar_comando_gcode("G1 Z0.2 F600")  # Mover Z 0.2mm hacia arriba cada vez que se juntan y separan pulgar e índice
        
    puño_izquierdo_cerrado_anteriormente = puño_izquierdo_cerrado_actualmente
    separación_anterior = separación_actual

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()

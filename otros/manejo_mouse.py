import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(pt1, pt2, frame_height, frame_width):
    x1, y1 = int(pt1[0] * frame_width), int(pt1[1] * frame_height)
    x2, y2 = int(pt2[0] * frame_width), int(pt2[1] * frame_height)
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), (x1 + x2) // 2, (y1 + y2) // 2

cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

zoom_threshold = 40  # Umbral para el gesto de zoom in que genera un click

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)  # Voltear la imagen de la cámara para sincronización con el movimiento del mouse
    frame_height, frame_width, _ = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            distance, cx, cy = calculate_distance((thumb_tip.x, thumb_tip.y), 
                                                  (index_finger_tip.x, index_finger_tip.y),
                                                  frame_height, frame_width)
            
            # Convertir la posición de la mano en coordenadas de la pantalla
            screen_x = np.interp(cx, (0, frame_width), (0, screen_width))
            screen_y = np.interp(cy, (0, frame_height), (0, screen_height))
            
            # Mover el cursor del mouse
            pyautogui.moveTo(screen_x, screen_y)
            
            if distance < zoom_threshold:
                # Realizar un click si se detecta un gesto de zoom in
                pyautogui.click()
                pyautogui.sleep(0.1)  # Pequeña pausa para evitar múltiples clicks
                
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

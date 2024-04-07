import cv2
import mediapipe as mp

# Inicializar Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convertir la imagen de BGR a RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen y detectar las manos
    results = hands.process(image)
    
    # Convertir la imagen de nuevo a BGR para mostrarla
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Dibuja las marcas de la mano y las conexiones
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Mediapipe clasifica la mano como izquierda o derecha, que se encuentra en los resultados
            classification = results.multi_handedness[hand_idx]
            label = classification.classification[0].label  # Será 'Derecha' o 'Izquierda'
            
            if label == "Right":
                label = "Derecha"
            else:
                label = "Izquierda"
   
            text = f'Mano {label}'
            
            # Obtener la posición de la muñeca para mostrar la clasificación
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            org = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
            
            cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

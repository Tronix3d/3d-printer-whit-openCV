import cv2
import numpy as np

# Cargar el clasificador preentrenado para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Dibujar rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Superposición de formas y texto
        # Esto es solo un ejemplo básico, necesitarás personalizarlo
        cv2.putText(frame, "Nombre: Tony Stark", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Profesión: Ingeniero", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Ejemplo de cómo podrías dibujar formas alrededor de los ojos
        # Estos valores están hardcodeados, deberías ajustarlos según la posición y tamaño del rostro
        cv2.circle(frame, (x + int(w*0.3), y + int(h*0.4)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (x + int(w*0.7), y + int(h*0.4)), 10, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

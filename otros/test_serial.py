import serial
import time

# Configura el puerto serial. Ajusta 'COM3' y 115200 a tu configuración específica.
puerto_serial = 'COM3'
baudios = 115200

try:
    # Intenta establecer la conexión serial
    with serial.Serial(puerto_serial, baudios, timeout=1) as ser:
        time.sleep(2)  # Espera para que la conexión se estabilice
        
        # Enviar un comando al dispositivo. Por ejemplo, 'M115' es un comando común G-code para obtener información del firmware en impresoras 3D.
        comando = "M115\n"
        ser.write(comando.encode())

        # Esperar brevemente para que el dispositivo responda
        time.sleep(1)

        # Leer la respuesta del dispositivo
        respuesta = ser.read_all().decode()

        # Imprimir la respuesta
        print(f"Respuesta recibida: {respuesta}")
        
        ser.write("G28".encode())

except serial.SerialException as e:
    print(f"Error al abrir el puerto serial: {e}")

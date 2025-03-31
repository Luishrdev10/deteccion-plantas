import cv2
from ultralytics import YOLO
import numpy as np

"""
#prueba con imagen 
# Cargar el modelo entrenado
model = YOLO('C:/Users/luish/Proyectos Visual studio code/Yolo/deteccion-plantas/src/runs/train/exp1/weights/best.pt') # llama al modelo entrenado de train.py llamado best.pt

if __name__ == "__main__":
    # Validar el modelo (evitar multiprocessing issue en Windows)
    metrics = model.val(workers=0)
    print(metrics) 

    # Verificar las clases detectadas
    print(model.names) 

    # Cargar imagen de prueba
    image_path = "C:/Users/luish/Proyectos Visual studio code/Yolo/deteccion-plantas/datasets/test/images/image.png" # aqui es la direccion de la imagen 
    image = cv2.imread(image_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
    else:
        # Realizar predicción
        results = model(image, imgsz=640,conf=0.1) # imgsz 

        # Mostrar la imagen con las detecciones
        cv2.imshow("Detección", results[0].show())  # mostrara el resultado de la imagen 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

"""
"""
 # probando con camara en directo 
# Cargar el modelo entrenado 

model = YOLO('C:/Users/luish/Proyectos Visual studio code/Yolo/deteccion-plantas/src/runs/train/exp1/weights/best.pt')

# Iniciar la captura de video desde la webcam (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Realizar predicción en el frame
    results = model(frame, conf=0.5)  # Ajusta conf para mejor precisión

    # Mostrar la imagen con las detecciones
    
    cv2.imshow("Detección en Tiempo Real", results[0].plot())

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
"""
"""
model = YOLO('C:/Users/luish/Proyectos Visual studio code/Yolo/deteccion-plantas/src/runs/train/exp1/weights/best.pt')

# Iniciar la captura de video desde la webcam (0 = cámara por defecto)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar un frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Realizar predicción en el frame
    results = model(frame, conf=0.5)

    for result in results:
        # extraer las coordenadas de la caja del objeto detectado
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf.cpu().numpy().item())
            label = model.names[int(box.cls.cpu().numpy().item())]

            # Dibujar la caja del objeto detectado
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

           
          
    # deteccion en tiempo real
    cv2.imshow("Detección en Tiempo Real", results[0].plot())

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
"""




model = YOLO('C:/Users/luish/Proyectos Visual studio code/Yolo/deteccion-plantas/src/runs/train/exp1/weights/best.pt')

# cap = cv2.VideoCapture(0)

# Conectar a DroidCam   esto es en la teminal si lo conectas por usb : pasos=>1 .\adb kill-server. 2 .\adb start-server 3 .\adb devices 
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

total_detecciones_unicas = 0  # Contador global de objetos nuevos
tracked_centroids = []       # Lista de centroides del frame anterior
umbral_distancia = 50        # Distancia mínima (en píxeles) para considerar que es un objeto nuevo

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    results = model(frame,(640,480) ,conf=0.6)
    centroides_actuales = []  # Centroides de este frame
    detecciones_frame = 0

    for result in results:
        for box in result.boxes:
            # Extraer coordenadas y calcular centroide
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf.cpu().numpy().item())
            label = model.names[int(box.cls.cpu().numpy().item())]

            # Dibujar la caja y la etiqueta en el frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            cv2.putText(frame, f'{label} ({confidence:.2f})', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Calcular el centroide de la caja
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            centroides_actuales.append((cx, cy))
            detecciones_frame += 1

    # Comparar centroides actuales con los del frame anterior para determinar objetos nuevos
    for centro in centroides_actuales:
        # Si no hay ningún centroide previo, se considera nuevo
        if not tracked_centroids:
            total_detecciones_unicas += 1
        else:
            # Calcular la distancia mínima respecto a cada centroide anterior
            distancias = [np.linalg.norm(np.array(centro) - np.array(c_prev)) for c_prev in tracked_centroids]
            if min(distancias) > umbral_distancia:
                total_detecciones_unicas += 1

    # Actualizar los centroides para el siguiente frame
    tracked_centroids = centroides_actuales.copy()

    # Mostrar el conteo en el frame
    cv2.putText(frame, f'Detecciones en frame: {detecciones_frame}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Total detecciones unicas: {total_detecciones_unicas}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Detección en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





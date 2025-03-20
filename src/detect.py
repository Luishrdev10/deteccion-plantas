import cv2
from ultralytics import YOLO


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

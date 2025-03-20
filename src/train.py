from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Cargar el modelo YOLO11n pre-entrenado
    model = YOLO("yolo11n.pt")

    # Entrenar el modelo con opciones de salida
    results = model.train(
        data="C:/Users/luish/Proyectos Visual studio code/Yolo/deteccion-plantas/config/data.yaml", # ruta del archivo yaml
        epochs=1,                   #frecuencia que practica el modelo 
        batch=2,                    # entre mas grande sea el batch puede acelerar el entrenamiento si tienes suficiente memoria
        workers=0,                  # Número de subprocesos de trabajo para cargar los datos
        imgsz=640,                  # Tamaño de la imagen en píxeles
        device="cuda:0",           # Usar GPU   => (device="cpu") si no tienes GPU 
        project="runs/train",      # Directorio de resultados
        name="exp1",               # Nombre del experimento
        exist_ok=True,             #  sobrescribir experimento en la misma carpeta
        save=True,                 # Guarda los resultados
        plots=True,                # Genera gráficos y otros archivos de salida
        verbose=True               #  Muestra información detallada sobre el proceso de entrenamiento
    )
    torch.cuda.empty_cache() #Libera la memoria de la GPU 

    print("Entrenamiento finalizado. Revisa la carpeta 'runs/train/exp1' para ver los resultados.")



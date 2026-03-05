import cv2
import os
import numpy as np
import time

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS AUTOMÁTICAS (BLINDADO)
# =============================================================================
# Detecta la carpeta donde está guardado este archivo (C:\Reconocimiento EmocionesB3)
base_path = os.path.dirname(os.path.abspath(__file__))
dataPath = os.path.join(base_path, 'Data')

def obtenerModelo(method, facesData, labels):
    emotion_recognizer = None
    
    # Seleccionamos el método
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor
    print(f"\n--- Entrenando ({method}) ... ---")
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time() - inicio
    print(f"Tiempo de entrenamiento ({method}): {tiempoEntrenamiento:.2f} segundos")

    # Guardar el modelo en la MISMA carpeta del proyecto
    nombre_archivo = "modelo" + method + ".xml"
    ruta_modelo = os.path.join(base_path, nombre_archivo)
    
    emotion_recognizer.write(ruta_modelo)
    print(f"✅ Modelo guardado exitosamente en: {ruta_modelo}")

# Verificación de seguridad
if not os.path.exists(dataPath):
    print(f"❌ ERROR: No encuentro la carpeta de datos en: {dataPath}")
    print("Asegúrate de haber capturado rostros primero.")
    exit()

emotionsList = os.listdir(dataPath)
emotionsList.sort() # <--- IMPORTANTE: Mantiene el orden alfabético

print('Lista de emociones detectadas:', emotionsList)

labels = []
facesData = []
label = 0

print("Leyendo imágenes...")
for nameDir in emotionsList:
    emotionsPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(emotionsPath):
        # Cargar imagen en escala de grises
        img_path = os.path.join(emotionsPath, fileName)
        facesData.append(cv2.imread(img_path, 0))
        labels.append(label)
    
    label = label + 1

# =============================================================================
# 2. ENTRENAMIENTO (Optimizado)
# =============================================================================
# Solo entrenamos LBPH porque es el que usa tu App Principal.
# Si activas los otros, tardará mucho más y ocupará más espacio innecesariamente.

# obtenerModelo('EigenFaces', facesData, labels)   # <-- Desactivado por lento
# obtenerModelo('FisherFaces', facesData, labels)  # <-- Desactivado por lento
obtenerModelo('LBPH', facesData, labels)           # <-- ¡ESTE ES EL IMPORTANTE!

print("\n✨ Entrenamiento finalizado.")
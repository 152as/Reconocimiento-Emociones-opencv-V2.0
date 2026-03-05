# Reconocimiento de Emociones en Tiempo Real (V2.0)

Este proyecto desarrolla un sistema de visión artificial capaz de detectar y clasificar emociones humanas a través de una cámara web utilizando **Python** y la librería **OpenCV**.

## 🚀 Descripción
El sistema utiliza algoritmos de reconocimiento facial para identificar rostros y posteriormente clasificar la emoción detectada basándose en modelos previamente entrenados (EigenFaces, FisherFaces y LBPH).

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python
* **Librerías:** OpenCV (cv2), NumPy
* **Herramientas:** Haar Cascades para la detección frontal de rostros.

## 📁 Estructura del Proyecto
* `capturandoRostros2.py`: Script para recolectar las muestras de rostros y etiquetas.
* `entrenando2.py`: Script para entrenar los modelos de reconocimiento.
* `reconocimientoEmociones2.py`: Script principal para la ejecución en tiempo real.
* `haarcascade_frontalface_default.xml`: Clasificador pre-entrenado para detección facial.

## 🔧 Instalación y Uso
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/152as/Reconocimiento-Emociones-opencv-V2.0.git](https://github.com/152as/Reconocimiento-Emociones-opencv-V2.0.git)
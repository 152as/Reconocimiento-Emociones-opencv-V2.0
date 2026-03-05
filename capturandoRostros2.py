import cv2
import os
import imutils

# =============================================================================
# 1. SELECCIÓN DE EMOCIÓN (Mantén este bloque igual)
# =============================================================================
# CAMBIA ESTO SEGÚN LA EMOCIÓN QUE QUIERAS CAPTURAR    Tristeza   Felicidad  Enojo      Sorpresa        

emotionName = 'Sorpresa'             

# =============================================================================
# 2. CONFIGURACIÓN DE RUTAS INTELIGENTES
# =============================================================================
# Detecta automáticamente la carpeta donde está este archivo .py
# Esto evita errores por el nombre de usuario o acentos en la ruta
base_path = os.path.dirname(os.path.abspath(__file__))

# Carpeta donde se guardarán las fotos
dataPath = os.path.join(base_path, 'Data')
emotionsPath = os.path.join(dataPath, emotionName)

# Crear carpetas si no existen de forma segura
os.makedirs(emotionsPath, exist_ok=True)

# BUSCAR EL XML EN TU CARPETA (Evita buscar en System32 o AppData)
xml_local = os.path.join(base_path, 'haarcascade_frontalface_default.xml')

# Verificación de seguridad antes de iniciar
if not os.path.exists(xml_local):
    print(f"\n❌ ERROR CRÍTICO: No se encuentra el archivo '{xml_local}'")
    print("Asegúrate de copiar el archivo haarcascade_frontalface_default.xml en esta carpeta.")
    exit()

# =============================================================================
# 3. LÓGICA DE CAPTURA
# =============================================================================
faceClassif = cv2.CascadeClassifier(xml_local)

if faceClassif.empty():
    print("❌ ERROR: No se pudo cargar el clasificador. El archivo XML podría estar dañado.")
    exit()

# Intentar abrir la cámara (CAP_DSHOW ayuda en laptops con Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

print(f"\n✅ Todo listo. Capturando: {emotionName}")
print("Enfoca tu rostro a la cámara...")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("❌ Error: No se puede acceder a la cámara.")
        break
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Guardar imagen con ruta protegida
        file_path = os.path.join(emotionsPath, f'rostro_{count}.jpg')
        cv2.imwrite(file_path, rostro)
        count += 1
        
    cv2.imshow('Capturando Rostros - (ESC para salir)', frame)

    k = cv2.waitKey(1)
    # Detenerse al presionar ESC o al llegar a 200 fotos
    if k == 27 or count >= 200:
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Finalizado: Se guardaron {count} imágenes en {emotionsPath}")
import cv2
import os
import numpy as np
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
import customtkinter as ctk 
from openai import OpenAI

# =============================================================================
# CONFIGURACIÓN 
# =============================================================================

# 1. TU API KEY
client = OpenAI(api_key="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")

# 2. RUTAS BLINDADAS (Portabilidad Total)
# Detecta automáticamente dónde está este archivo (C:\Reconocimiento EmocionesB3)
base_path = os.path.dirname(os.path.abspath(__file__))

dataPath = os.path.join(base_path, 'Data')
emojisPath = os.path.join(base_path, 'Emojis') 
modeloPath = os.path.join(base_path, 'modeloLBPH.xml')
carpetaInformes = os.path.join(base_path, 'Informes finales')

# RUTA DEL XML LOCAL (La clave para arreglar el error de Emérita)
xml_local = os.path.join(base_path, 'haarcascade_frontalface_default.xml')

if not os.path.exists(carpetaInformes):
    os.makedirs(carpetaInformes)

ctk.deactivate_automatic_dpi_awareness()

# =============================================================================
# LÓGICA DE LA APLICACIÓN
# =============================================================================

class EmotionApp:
    def __init__(self):
        self.nombre_usuario = ""
        self.running = True 
        self.video_activo = True 
        self.cap = None # Inicializamos explícitamente
        
        # Verificación de seguridad para el XML
        if not os.path.exists(xml_local):
            print(f"❌ ERROR CRÍTICO: Falta el archivo {xml_local}")
            print("Por favor copia el haarcascade_frontalface_default.xml en esta carpeta.")
        
        self.emojis_cache = {}
        self.cargar_emojis()
        
        self.pedir_nombre() 

    def cargar_emojis(self):
        if not os.path.exists(dataPath): return
        lista_emociones = os.listdir(dataPath)
        extensiones = ['.png', '.jpg', '.jpeg']
        
        for emocion in lista_emociones:
            for ext in extensiones:
                ruta_img = os.path.join(emojisPath, emocion + ext)
                if os.path.exists(ruta_img):
                    try:
                        pil_img = Image.open(ruta_img)
                        self.emojis_cache[emocion] = ctk.CTkImage(light_image=pil_img, 
                                                                  dark_image=pil_img, 
                                                                  size=(140, 140))
                        break
                    except:
                        pass

    def pedir_nombre(self):
        ctk.set_appearance_mode("Dark")
        self.dialog = ctk.CTk()
        self.dialog.title("Registro")
        self.dialog.geometry("300x200")
        
        ctk.CTkLabel(self.dialog, text="Ingrese Nombre del Paciente:", font=("Arial", 14)).pack(pady=20)
        self.entry_nombre = ctk.CTkEntry(self.dialog)
        self.entry_nombre.pack(pady=10)
        
        ctk.CTkButton(self.dialog, text="Iniciar Sesión", command=self.validar_nombre).pack(pady=10)
        self.dialog.mainloop()

    def validar_nombre(self):
        self.nombre_usuario = self.entry_nombre.get()
        if self.nombre_usuario:
            self.dialog.destroy()
            self.iniciar_app_principal()

    def iniciar_app_principal(self):
        self.root = ctk.CTk()
        self.root.title(f"Sesión Terapéutica - Paciente: {self.nombre_usuario}")
        self.root.geometry("1200x800") 
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Cargar Modelo LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(modeloPath):
            self.recognizer.read(modeloPath)
        else:
            print("ADVERTENCIA: No se encontró el modelo entrenado.")
            
        self.labels = os.listdir(dataPath)
        self.labels.sort() 
        
        # Cargar Detector de Rostros desde TU CARPETA (Evita errores de sistema)
        self.faceClassif = cv2.CascadeClassifier(xml_local)
        
        # Historial GPT
        self.historial_chat = [
            {"role": "system", "content": f"Eres un psicólogo profesional y empático. Estás atendiendo a {self.nombre_usuario}. Tu objetivo es analizar su estado emocional y conversar brevemente para dar un diagnóstico constructivo."}
        ]

        self.setup_ui()
        self.start_video()
        self.root.mainloop()

    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=1) 
        self.root.grid_columnconfigure(1, weight=2) 

        # --- COLUMNA IZQUIERDA: VIDEO ---
        self.frame_video = ctk.CTkFrame(self.root)
        self.frame_video.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_video = ctk.CTkLabel(self.frame_video, text="")
        self.lbl_video.pack(pady=20, expand=True)
        
        self.lbl_estado = ctk.CTkLabel(self.frame_video, text="Estado: Esperando análisis...", font=("Arial", 20, "bold"), text_color="cyan")
        self.lbl_estado.pack(pady=20)

        # Botón Reactivar
        self.btn_reactivar = ctk.CTkButton(self.frame_video, text="🔄 Reactivar Cámara", fg_color="gray", command=self.reactivar_camara)
        self.btn_reactivar.pack(pady=10)

        # --- COLUMNA DERECHA: CHAT Y EMOJI ---
        self.frame_chat = ctk.CTkFrame(self.root, fg_color="#181818") 
        self.frame_chat.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # 1. EMOJI 
        self.lbl_emoji_img = ctk.CTkLabel(self.frame_chat, text="") 
        self.lbl_emoji_img.pack(pady=10)

        # 2. CHAT 
        self.scroll_chat = ctk.CTkScrollableFrame(self.frame_chat, fg_color="transparent")
        self.scroll_chat.pack(expand=True, fill="both", padx=10, pady=5)
        
        # 3. INPUTS
        self.frame_input = ctk.CTkFrame(self.frame_chat, fg_color="transparent")
        self.frame_input.pack(fill="x", padx=10, pady=10)

        self.progress = ctk.CTkProgressBar(self.frame_input)
        self.progress.set(0)
        self.progress.pack(fill="x", pady=5)
        
        self.btn_analizar = ctk.CTkButton(self.frame_input, text="📷 CAPTURAR Y ANALIZAR", fg_color="#1f538d", height=40, command=self.iniciar_analisis_facial)
        self.btn_analizar.pack(fill="x", pady=5)

        self.entry_mensaje = ctk.CTkEntry(self.frame_input, placeholder_text="Escriba su mensaje aquí...", height=40)
        self.entry_mensaje.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.entry_mensaje.bind("<Return>", lambda e: self.enviar_mensaje_usuario())

        self.btn_enviar = ctk.CTkButton(self.frame_input, text="Enviar ➤", width=80, height=40, fg_color="#00a884", command=self.enviar_mensaje_usuario)
        self.btn_enviar.pack(side="right")

        self.btn_finalizar = ctk.CTkButton(self.root, text="🔴 FINALIZAR SESIÓN Y GUARDAR INFORME", fg_color="#8B0000", height=40, command=self.generar_informe_final)
        self.btn_finalizar.grid(row=1, column=0, columnspan=2, pady=10, padx=20, sticky="ew")

    def start_video(self):
        # Aseguramos limpieza antes de iniciar
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.update_frame()

    def reactivar_camara(self):
        if not self.video_activo:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.video_activo = True
            self.lbl_estado.configure(text="Estado: Cámara activa", text_color="cyan")
            self.update_frame() 

    def update_frame(self):
        # Si la cámara no existe o no está activa, no hacemos nada
        if not self.running or not self.video_activo or self.cap is None: 
            return

        try:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceClassif.detectMultiScale(gray, 1.3, 5)
                
                emotion_detected = "Neutro" 

                for (x, y, w, h) in faces:
                    rostro = cv2.resize(gray[y:y+h, x:x+w], (150, 150))
                    label, conf = self.recognizer.predict(rostro)
                    
                    if conf < 85: 
                        emotion_detected = self.labels[label]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, emotion_detected, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                self.current_emotion = emotion_detected

                if self.root.winfo_exists():
                    img = ctk.CTkImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), size=(480, 360))
                    self.lbl_video.configure(image=img)
                    
                    # Emoji
                    if self.current_emotion in self.emojis_cache:
                        self.lbl_emoji_img.configure(image=self.emojis_cache[self.current_emotion])
                    else:
                        self.lbl_emoji_img.configure(image=None)
            
            # Solo nos llamamos a nosotros mismos si el video SIGUE activo
            if self.running and self.video_activo:
                self.root.after(10, self.update_frame)
        except Exception as e:
            print(f"Error cámara: {e}")

    def agregar_burbuja_chat(self, remitente, texto, es_usuario):
        if not self.root.winfo_exists(): return

        color_fondo = "#005c4b" if es_usuario else "#202c33"
        alineacion = "e" if es_usuario else "w"
        
        contenedor = ctk.CTkFrame(self.scroll_chat, fg_color="transparent")
        contenedor.pack(fill="x", pady=5)
        
        burbuja = ctk.CTkLabel(
            contenedor, 
            text=f"{texto}", 
            fg_color=color_fondo,
            text_color="white",
            corner_radius=15,
            justify="left",
            wraplength=400,
            padx=15, pady=10,
            font=("Arial", 12)
        )
        burbuja.pack(anchor=alineacion, padx=10)
        
        nombre = ctk.CTkLabel(contenedor, text=remitente, font=("Arial", 9), text_color="gray")
        nombre.pack(anchor=alineacion, padx=15)

        self.scroll_chat._parent_canvas.yview_moveto(1.0)

    def iniciar_analisis_facial(self):
        # Detenemos lógica de video
        self.video_activo = False 
        self.lbl_estado.configure(text=f"CAPTURA: {self.current_emotion}", text_color="#00FF00")
        
        # Limpieza de cámara
        if self.cap is not None:
            self.cap.release()
        self.cap = None 

        self.btn_analizar.configure(state="disabled")
        threading.Thread(target=self.proceso_carga).start()

    def proceso_carga(self):
        for i in range(101):
            if not self.running: return 
            time.sleep(0.01)
            try:
                self.progress.set(i / 100)
            except:
                pass
        
        if self.running:
            mensaje_sistema = f"He analizado tu expresión facial y detecto: {self.current_emotion}."
            self.root.after(0, lambda: self.agregar_burbuja_chat("Sistema", mensaje_sistema, False))
            
            self.historial_chat.append({"role": "user", "content": f"[El sistema detectó visualmente: {self.current_emotion}]. ¿Qué opinas?"})
            
            self.obtener_respuesta_gpt()
            try:
                self.btn_analizar.configure(state="normal")
            except:
                pass

    def enviar_mensaje_usuario(self):
        texto = self.entry_mensaje.get()
        if not texto: return
        
        self.agregar_burbuja_chat(self.nombre_usuario, texto, True)
        self.entry_mensaje.delete(0, "end")
        
        self.historial_chat.append({"role": "user", "content": texto})
        threading.Thread(target=self.obtener_respuesta_gpt).start()

    def obtener_respuesta_gpt(self):
        try:
            resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=self.historial_chat)
            respuesta = resp.choices[0].message.content
            self.historial_chat.append({"role": "assistant", "content": respuesta})
            
            self.root.after(0, lambda: self.agregar_burbuja_chat("Psicólogo IA", respuesta, False))
        except Exception as e:
            self.root.after(0, lambda: self.agregar_burbuja_chat("Error", str(e), False))

    def generar_informe_final(self):
        self.agregar_burbuja_chat("Sistema", "Generando informe médico... espera un momento.", False)
        prompt_final = "Genera un informe final breve y profesional con: Diagnóstico emocional y Recomendaciones."
        self.historial_chat.append({"role": "user", "content": prompt_final})
        
        def tarea_informe():
            try:
                resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=self.historial_chat)
                informe_ia = resp.choices[0].message.content
                fecha = datetime.now().strftime("%Y-%m-%d_%H-%M")
                nombre_archivo = f"Informe_{self.nombre_usuario}_{fecha}.txt"
                ruta_final = os.path.join(carpetaInformes, nombre_archivo)
                
                with open(ruta_final, "w", encoding="utf-8") as f:
                    f.write(f"PACIENTE: {self.nombre_usuario}\nFECHA: {fecha}\n\n{informe_ia}")

                self.root.after(0, lambda: self.agregar_burbuja_chat("Sistema", f"✅ Informe guardado: {nombre_archivo}", False))
                time.sleep(3)
                self.on_close() 
            except Exception as e:
                print(f"Error: {e}")
        
        threading.Thread(target=tarea_informe).start()

    def on_close(self):
        self.running = False
        self.video_activo = False
        if self.cap is not None:
            self.cap.release()
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

if __name__ == "__main__":
    EmotionApp()
import cv2
import SeguimientoManos as sm
from ultralytics import YOLO
from gtts import gTTS
import pygame
import os
import time
import tempfile
import threading

#lectura de la camara
cap=cv2.VideoCapture(0)
#cambiar la resolucion
cap.set(3, 1280)
cap.set(4,720)

#leer nuestro modelo
model=YOLO('last.pt')

#declarar detector
detector=sm.detectormanos(Confdeteccion=0.9)

# Inicializar pygame para reproducir audio
pygame.mixer.init()

# Variables para control de TTS
ultima_letra = ""
ultimo_tiempo_tts = 0
intervalo_tts = 2  # Segundos entre reproducciones de la misma letra

def reproducir_tts(texto):
    """Función para convertir texto a voz y reproducirlo"""
    try:
        # Crear objeto gTTS
        tts = gTTS(text=texto, lang='es', slow=False)
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)
        
        # Reproducir el archivo
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Esperar a que termine la reproducción
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        # Limpiar archivo temporal
        os.unlink(temp_filename)
        
    except Exception as e:
        print(f"Error en TTS: {e}")

while True:
    #realizar la lectura de la cap
    ret,frame=cap.read()
    #extra informacion de la mano
    frame=detector.encontrarmanos(frame,dibujar=False) #!!!PUNTOS
    #posicion de una sola mano
    lista1, bbox, mano=detector.encontrarposicion(frame,ManoNum=0,dibujarPuntos=False, dibujarBox=False, color=[0,255,0])
    #si hay mano
    if mano==1:
        #Extraer la informacion del cuadro
        xmin,ymin,xmax,ymax= bbox

        # Asignamos margen
        xmin -= 40
        ymin -= 40
        xmax += 40
        ymax += 40

        # Asegurar que el recorte está dentro de la imagen
        alto, ancho, _ = frame.shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(ancho, xmax)
        ymax = min(alto, ymax)

        # Realizar recorte
        recorte = frame[ymin:ymax, xmin:xmax]

        if recorte.size != 0:
            recorte = cv2.resize(recorte, (640, 640), interpolation=cv2.INTER_CUBIC)
            resultados = model.predict(recorte, conf=0.4)  # Reducir confianza para más detecciones
            if len(resultados) != 0:
                for result in resultados:
                    # Obtener la clase detectada
                    if len(result.boxes) > 0:
                        clase_id = int(result.boxes[0].cls[0])
                        clase_nombre = model.names[clase_id]
                        confianza = float(result.boxes[0].conf[0])
                        
                        # Mostrar en pantalla principal
                        cv2.putText(frame, f"Letra: {clase_nombre} ({confianza:.2f})", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        
                        # Mostrar estado del TTS
                        if pygame.mixer.music.get_busy():
                            cv2.putText(frame, "Reproduciendo audio...", 
                                      (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Control de TTS - reproducir solo si es una nueva letra o ha pasado el intervalo
                        tiempo_actual = time.time()
                        if (clase_nombre != ultima_letra or 
                            tiempo_actual - ultimo_tiempo_tts > intervalo_tts):
                            
                            # Reproducir TTS en un hilo separado para no bloquear la detección
                            hilo_tts = threading.Thread(target=reproducir_tts, args=(f"Letra {clase_nombre}",))
                            hilo_tts.daemon = True
                            hilo_tts.start()
                            
                            # Actualizar variables de control
                            ultima_letra = clase_nombre
                            ultimo_tiempo_tts = tiempo_actual
                        
                        # Mostrar recorte con detección
                        anotaciones = result.plot()
                        cv2.imshow("RECORTE", anotaciones)
                        print(f"Detectado: {clase_nombre} con confianza {confianza:.2f}")
            else:
                # Mostrar recorte sin detección
                cv2.imshow("RECORTE", recorte)

    # Mostrar instrucciones
    cv2.putText(frame, "ESC: Salir | TTS activado", 
              (50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    #Mostrar FPS
    cv2.imshow("Lenguaje Vocales",frame)
    #leer nuestro teclado
    t=cv2.waitKey(1)
    if t==27:
        break
cap.release()
cv2.destroyAllWindows()
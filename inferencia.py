import cv2
import SeguimientoManos as sm
from ultralytics import YOLO

#lectura de la camara
cap=cv2.VideoCapture(0)
#cambiar la resolucion
cap.set(3, 1280)
cap.set(4,720)

#leer nuestro modelo
model=YOLO('last.pt')

#declarar detector
detector=sm.detectormanos(Confdeteccion=0.9)

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
            resultados = model.predict(recorte, conf=0.55)
            if len(resultados) != 0:
                for result in resultados:
                    masks = result.masks
                    coordenadas = masks
                    anotaciones = resultados[0].plot()
                cv2.imshow("RECORTE", anotaciones)

    #Mostrar FPS
    cv2.imshow("Lenguaje Vocales",frame)
    #leer nuestro teclado
    t=cv2.waitKey(1)
    if t==27:
        break
cap.release()
cv2.destroyAllWindows()
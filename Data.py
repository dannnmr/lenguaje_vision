#importar librerias
import cv2
import os

#import la clase
import SeguimientoManos as sm

# Creacion de la carpeta
nombre= 'Letra_U'
direccion='C:/Users/Asus TUF/Desktop/LenguajeVocales/data'
carpeta=direccion +'/'+ nombre
# si no esta creada la carpeta
if not os.path.exists(carpeta):
    print("carpeta creada: ", carpeta)
    #creamos la carpeta
    os.makedirs(carpeta)

#lectura de la camara
cap=cv2.VideoCapture(0)
#cambiar la resolucion
cap.set(3, 1280)
cap.set(4,720)

#Declaramos contador
cont=0
#declarar detector
detector=sm.detectormanos(Confdeteccion=0.9)

while True:
    #realizar la lectura de la cap
    ret,frame=cap.read()
    #extra informacion de la mano
    frame=detector.encontrarmanos(frame,dibujar=True) #!!!PUNTOS
    #posicion de una sola mano
    lista1, bbox, mano=detector.encontrarposicion(frame,ManoNum=0,dibujarPuntos=False, dibujarBox=False, color=[0,255,0])
    #si hay mano
    if mano==1:
        #Extraer la informacion del cuadro
        xmin,ymin,xmax,ymax= bbox

        #Asignamos margen
        xmin = xmin-40
        ymin = ymin-40
        xmax = xmax+40
        ymax = ymax+40

        #Realizar recorte de nuestra mano
        recorte = frame[ymin:ymax, xmin:xmax]

        #Redimensionamiento
        #recorte=cv2.resize(recorte,(640,640), interpolation = cv2.INTER_CUBIC)

        #almacenar nuestras imagenes
        cv2.imwrite(carpeta+"/U_{}.jpg".format(cont),recorte)

        #Aumentamos contador
        cont=cont+1
        cv2.imshow("RECORTE",recorte)

    #Mostrar FPS
    cv2.imshow("Lenguaje Vocales",frame)
    #leer nuestro teclado
    t=cv2.waitKey(1)
    if t==27 or cont==100:
        break
cap.release()
cv2.destroyAllWindows()


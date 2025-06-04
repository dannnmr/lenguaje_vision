#importar librerias
import math
import cv2
import mediapipe as mp
import time

#se crea la clase
class detectormanos():
    #se inicializa los parametros de la deteccion
    def __init__(self, mode=False, maxManos=2, model_complexity=1, Confdeteccion =0.5, ConfSegui=0.5):
        self.mode = mode #creamos el objeto y el tendra su propia variable
        self.maxManos = maxManos #lo mismo se hara con todas las objetos
        self.compl=model_complexity
        self.Confdeteccion=Confdeteccion
        self.ConfSegui=ConfSegui

        #Se crea los objetos que detectaron las manos y las dibujaran
        self.mpmanos=mp.solutions.hands
        self.manos=self.mpmanos.Hands(self.mode,self.maxManos,self.compl,self.Confdeteccion,self.ConfSegui)
        self.dibujo=mp.solutions.drawing_utils
        self.tip = [4,8,12,16,20]

        #Funcion para encontrar las manos
    def encontrarmanos(self, frame, dibujar=True ):
        imgcolor=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.resultados=self.manos.process(imgcolor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame,mano,self.mpmanos.HAND_CONNECTIONS) #dibujamos las conexiones de las manos
        return frame

    #funcion para encontrar la posicion
    def encontrarposicion(self,frame,ManoNum=0, dibujarPuntos=True,dibujarBox=True,color=[]):
        xlista=[]
        ylista=[]
        bbox=[]
        player=0
        self.lista=[]
        if self.resultados.multi_hand_landmarks:
            miMano=self.resultados.multi_hand_landmarks[ManoNum]
            prueba=self.resultados.multi_hand_landmarks
            player=len(prueba)
                #print(player)
            for id, lm in enumerate(miMano.landmark):
                alto,ancho,c=frame.shape # extraemos las dimensiones de los fps
                cx,cy =int(lm.x*ancho),int(lm.y*alto) #convertimos la informacion en pixeles
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id,cx,cy])
                if dibujarPuntos:
                    cv2.circle(frame,(cx,cy), 3, (0,0,0),cv2.FILLED) #dibujamos un circulo

            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax
            if dibujarBox:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), color, 2)
        return self.lista,bbox,player

        #funcion para detectar y dibujar los dedso arriba
    def dedosarriba(self):
        dedos=[]
        if self.lista[self.tip[0]][1]<self.lista[self.tip[0]-1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

#!/bin/bash

echo "Instalando dependencias para el sistema de reconocimiento de vocales con TTS..."

# Instalar dependencias de Python
pip install -r requirements.txt

echo "¡Instalación completada!"
echo ""
echo "Para ejecutar el programa:"
echo "python inferencia.py"
echo ""
echo "Controles:"
echo "- ESC: Salir del programa"
echo "- El sistema reproducirá automáticamente el nombre de cada vocal detectada"
echo "- Hay un intervalo de 2 segundos entre reproducciones de la misma letra"

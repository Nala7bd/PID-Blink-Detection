REQUISITOS:
Para ejecutar esta aplicación se necesitará de la última versión de Python 3, 
junto a las siguientes librerias, en su última versión disponible:

- Numpy.
- OpenCV

-----------------------------------
FUNCIONAMIENTO:

Para ejecutar la aplicación habrá que abrir una terminal de python en la ubicación del archivo .py y escribir 
el siguiente comando:

python blink_detection.py videoEntrada videoSalida 

Donde videoEntrada se refiere a la ruta del archivo correspondiente al video que contabilizirá 
los guiños y videoSalida se refiere a la ruta del archivo correspondiente al video generado a partir 
primero con la región de la cara y los ojos marcada. 

IMPORTANTE: los videos de salida deben de tener la extensión .avi

Ejemplos:

python blink_detection.py inputs/video.mp4 outputs/videoSalida.avi


---------------------------------
FUNCIONAMIENTO AVANZADO:
 
Partiendo de la linea de comando explicada en la parte anterior, existe una serie de parámetros 
opcionales con los que experimentar la aplicación que se introducirán después de indicar la ruta
del video de salida. Los parámetros opcionales son:

-blink_frames: nº entero que indica cuantos frames tendrán que detectarse con los ojos cerrados para 
contarlos como guiños. Detalles en la memoria

-likeness_factor: nº decimal que indica el grado de similitud. Detalles en la memoria


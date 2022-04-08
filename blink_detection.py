#IMPORTS:
import sys
import cv2 as cv
import numpy as np


##############################################################


faceClassifier = cv.CascadeClassifier("data/haarcascade_frontalface_default.xml")

#DEFINICIÓN DE LAS FUNCIONES


#Función para aplicar un filtro de mediana a cada canal rgb, con el objetivo de eliminar el ruido
# -Parámetros de entrada: img= la imagen donde queremos detectar caras.
#                         k = tamaño de la máscara mediana que se va aplicar a cada canal k*k. k tiene que ser impar. 
#                             Por defecto k=3
# -Parámetros de salida: 
#                         imágen resultante de aplicar los filtros en cada canal rgb
#
def rgbSmoothing(img,k=3):
    
    b,g,r=cv.split(img)
    
    b= cv.medianBlur(b,k)
    g= cv.medianBlur(g,k)
    r= cv.medianBlur(r,k)

    return cv.merge([b,g,r])

##############################################################

#Función para aplicar una ecualización del histograma al canal y YCrCb de la imágen
# -Parámetros de entrada: img= la imagen a la que queremos ajustar la iluminación
#                         histogram = variable Booleana que indica si quieres mostrar por pantalla los histogramas pre y post 
#                                     ecualización.
# -Parámetros de salida: 
#                         imágen resultante de aplicar la ecualización sobre el canal Y
#
def lightnessEqualization(img):
    
    y,cr,cb=cv.split(cv.cvtColor(img,cv.COLOR_BGR2YCR_CB))
    
    y_= cv.equalizeHist(y)
    
    
    
        
    return cv.cvtColor(cv.merge([y_,cr,cb]), cv.COLOR_YCR_CB2BGR)

##############################################################

#Función para detectar las caras de una imágen, 
# -Parámetros de entrada: img= la imagen donde queremos detectar caras.
#                         scale= valor del factor escala utilizado por el CascadeClassifier. Su valor por defecto es el 1.1
#                         neighbors= nº de vecinos mínimos que utilizará el CascadeClassifier. Su valor por defecto es 3.
#                         minSize  = Tamaño mínimo que tendra la zona que consideraremos cara
#                         maxSize  = Tamaño máximo que tendra la zona que consideraremos cara
#
# -Parámetros de salida: face= imagen únicamente formada por la cara detectada.
#                        image= imagen sobre la que detectamos la cara, con la región de la cara marcada con un rectángulo.
#                        coordenadas = lista de coordenadas de la zona de la cara
def detectFaces(img,scale=1.1,neighbors=3,minSize=None,maxSize=None):
    
    image=img.copy()
    

    faces_locations = faceClassifier.detectMultiScale(image,scaleFactor=scale,minNeighbors=neighbors,minSize=minSize,maxSize=maxSize)
    
    if  len(faces_locations)==0:
        raise SystemError("No se ha podido detectar una cara, revise el video de entrada o los parámetros opcionales establecidos.")
    
    
    area = 0
    coordenadas=list()
    for x,y,w,h in faces_locations:
        
        ##IMPORTANTE: para cv.rectangle las coordenadas son x,y ; para hacer slicing con numpy las coordenadas son y,x.
        nueva_area = h * w
        if area < nueva_area:
            area = nueva_area
            coordenadas=[x,y,w,h]
            face=img[y:y+h,x:x+w,:]     

    cv.rectangle(image,(coordenadas[0],coordenadas[1]),(coordenadas[0]+coordenadas[2],coordenadas[1]+coordenadas[3]),(255,0,0),2)
    return face,image,coordenadas


##############################################################

#Función para reescalar la imágen a otro tamaño distinto.
# -Parámetros de entrada: img= la imagen que queremos reescalar.
#                         scale_percent= porcentaje en el que reescalamos la imágen, 100 es el valor base.
# -Parámetros de salida: 
#                         imágen resultante del reescalado
#

def rescale(img,scale_percent=60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    
    dim = (width, height)
    
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)

##############################################################


## FUNCION PARA OBTENER UN RECORTE DEL AREA DERECHA DONDE ESTA EL OJO

## ENTRADA: tiene una img de la cara (tras detectFaces) como entrada 

## DEVUELVE: un recorte de la parte derecha donde se encuentra el ojo

def recortaOjoDerecho(img):
    x,y,z = img.shape
    # Mediante EXPERIMENTACION se ha determinado que es posible realizar un recorte mas exacto por ello x = x//10 e y = y//10 (recorte mas apurado por arriba y por el lado derecho)
    res = img[x//10:x//2, y//10:y//2]
    # Si realizamos algun cambio en estos parametros es necesario realizarlos en sacaContornos: CAMBIO_COORDENADAS
    return res

##############################################################

## FUNCION PARA OBTENER UN RECORTE DEL AREA IZQUIERDA DONDE ESTA EL OJO

## ENTRADA: tiene una img de la cara (tras detectFaces) como entrada 

## DEVUELVE: un recorte de la parte izquierda donde se encuentra el ojo
def recortaOjoIzquierdo(img):
    x,y,z = img.shape
    # Mediante EXPERIMENTACION se ha determinado que es posible realizar un recorte mas exacto por ello x = x//10 e y = y- y//8 (recorte mas apurado por arriba y por el lado derecho)
    res = img[x//10:x//2, y//2:y- y//8]
    # Si realizamos algun cambio en estos parametros es necesario realizarlos en sacaContornos: CAMBIO_COORDENADAS
    return res


##############################################################

## FUNCION PARA SACAR LOS CONTORNOS 

## ENTRADA: se le pasan como parametros la imagen a la que sacarle un contorno        ==> img
##          imagen de la cara a recortar                                              ==> img_cara 
##          umbrales del cotorno a elegir                                             ==> area_bajo, area_alto
##          erosion x veces aplicada y dilatacion                                     ==> erosion = 0,dilatacion = 0
##          es izq     (BOOL)                                                         ==> izq= False                         


## DEVUELVE: imagen_recortada_rectangulo, [puntos_interes] ==> 
##  puntos_interes == [(cx,cy),(X,Y,W,H)] (puntos de interes son los puntos donde tiene el centro el objeto encontrado
##           y las cordenadas de corte del rectangulo respecto la fotografia recortada NO ABSOLUTAS)


def sacaContornos(img,img_cara,erosion = 0,dilatacion = 0,izq = False):
  
    # Sacamos el shape de la imagen a pintar para ajustar las coordenadas de recortaOjoDerecho e Izquierdo y mas operaciones
    A,L,Z = img_cara.shape 

    # Hacemos una copia de la imagen original para que no le afecte el procesado
    copia_img = img.copy()

    # Damos un rango de color posible parala umbralizacion en las 3 capas de color EXPERIMENTACION
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([80, 80, 80])

    # Damos un rango dinamico en funcion del shape de la imagen que se ha de pintar (se usara para el area de los contornos) EXPERIMENTACION
    area_bajo = int(A/2)
    area_alto = 5000

    # Imagen resultante de la umbralizacion
    img_binaria = cv.inRange(copia_img, lower_color,upper_color)

    # Aplicacion de dilataciones y erosiones con un nucleo pequeño para realizar retoques (en funcion de v.Entrada erosion y dilatacion)
    # DUDA esta bien hecho la erosion(deciamos que estaba al reves erosion = a dilatacion y al reve)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=np.uint8)
    img_binaria_modificada = cv.erode(img_binaria, kernel, iterations= erosion)
    img_binaria_modificada = cv.dilate(img_binaria_modificada, kernel, iterations= dilatacion)

    # Damos la vuelta al resultado de la umbralizacion para el correcto funcionamiento de cv.findContours
    ### DUDA : creo que al final no es necesario hacer esto
    img_binaria_final = 255-img_binaria_modificada

    # Obtenemos la lista de contornos y  (DUDA quitar? no se usa img_hierarchy) => la jerarquia      
    lista_contours, img_hierarchy = cv.findContours(img_binaria_final,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # Creamos una lista que estara compuesta con los centros (cx,cy) de todos los umbrales para una posterior comprobacion
    lista_centros_puntos = []

    # Recorremos la lista de contornos donde se buscara rellenar la lista anterior con los centros de estos contornos
    for c in lista_contours:
    
        # Comprobamos que el area de dichos contornos se encuentre dentro del rango establecido al comienzo de la funcion
        if cv.contourArea(c) >= area_bajo and cv.contourArea(c)<=area_alto:

            # Sacamos los centros de los contornos
            M = cv.moments(c)
            cx = int(M['m10']/M['m00']) ## Puntos para el calculo de la Verificacion
            cy = int(M['m01']/M['m00']) ## Puntos para el calculo de la Verificacion
            
            # Sacamos las coordenadas necesarias de cada contorno 
            x, y, w, h = cv.boundingRect(c)
            
            # Comprobamos si realiazmos el tratamiento en el ojo iquierdo o derecho mediante izq v.Entrada
            if izq == True:
                
                # Ajustamos las coordenadas al recorteOjoIzquierdo (mirar esta funcion en caso de duda)
                # CAMBIO_COORDENADAS
                xr = x + A//2
                yr = y + L//8
                cx_r = cx+ A//2 #######
                cy_r = cy + L // 8
                lista_centros_puntos.append(((cx_r,cy),(xr,yr,w,h)))

            else:
                
                # Ajustamos las coordenadas al recorteOjoDerecho (mirar esta funcion en caso de duda)
                # CAMBIO_COORDENADAS
                xr = x + A//10
                yr = y + L//10
                cx_r = cx+ A//10 ####### 
                cy_r = cy + L // 10

                lista_centros_puntos.append(((cx_r,cy_r),(xr,yr,w,h)))

    # Sacamos el centro con la mayor coordenada y (vertical en la imagen) 
    punto_final = ()
    menor_centro = 0
    for i in lista_centros_puntos:
        centro_y = i[0][1]
        if centro_y >= menor_centro:
            menor_centro = centro_y
            punto_final = i

    # Daremos el Formato de salida que queremos con el centro (cx_r, max(cy)) y sus coordenadas (x,y,w,h)
    lista_coordenadas_final = []
    lista_coordenadas_final.append((punto_final[0][0], punto_final[0][1]))
    lista_coordenadas_final.append(punto_final[1])

    # Recorte de la imagen original de la zona del ojo encontrada
    result = img_cara[punto_final[1][1]:punto_final[1][1]+punto_final[1][3],punto_final[1][0]:punto_final[1][0]+punto_final[1][2]]

    return result,lista_coordenadas_final


##############################################################

#Función que dada una cara devolverá la imagen del ojo, así como sus coordenadas.

#Parámetros de entrada: detecting face -> cara detectada a la que se la hará la función para sacar el ojo.
#                       derecho -> atributo booleano que determinará si el ojo a sacar es el derecho o el izquierdo

def saca_ojo(detecting_face,derecho):
  
   #si tenemos que sacar el ojo derecho, haremos un pequeño recorte para sacar la zona y posteriormente llamamos a la función para calcularlo
    if derecho:
        detecting_ojo = recortaOjoDerecho(detecting_face)
        ojo,coordenadas = sacaContornos(detecting_ojo,detecting_face,erosion=0,dilatacion=2,izq= False)
     
       
   
    #en caso contrario hacemos lo mismo, pero haciendo recorte sobre el ojo izquierdo
    else: 
        detecting_ojo = recortaOjoIzquierdo(detecting_face)
        ojo,coordenadas = sacaContornos(detecting_ojo,detecting_face,erosion=0,dilatacion=2,izq= True)
       
    return ojo,coordenadas,detecting_ojo


##############################################################


## Funcion que dada una imagen, las coordenadas de los ojos izquierdos y derechos anteriores y la imagen de la zona de la cara anterios
#devuelve las imágenes de los dos nuevos ojos recortados asi como la imagen de la cara con ambos cuadrados dibujados.

#parámetros de entrada : new_image -> imagen del nuevo frame a procesar
#                        coordenadas_iniciales_izquierdo -> coordenadas de la posición relativa del ojo izquierdo
#                        coordenadas_iniciales_derecho -> coordenadas de la posición relativa del ojo derecho
#                        zona_cara -> imagen inicial de la zona de la cara anterior


def muestra_ojos(new_image,coordenadas_iniciales_izquierdo,coordenadas_iniciales_derecho,zona_cara):

    #sacamos el shape de la imagen de la cara anterior
    sh = np.shape(zona_cara)
    
    #calculamos la posición de la cara en el nuevo frame
    ni_img_face,n_img_entera,ccara= detectFaces(new_image,scale=1.1)

    #realizamos un reescalado para que ambas imágenes de la cara tengan el mismo tamaño
    escalado_nueva_cara = cv.resize(ni_img_face,(sh[0],sh[1]), interpolation=cv.INTER_CUBIC)
   

    #guardamos las coordenadas del ojo izquierdo y sacamos la posición relativa del mismo segun la nueva imagen de la cara
    ojo_x_izquierdo = coordenadas_iniciales_izquierdo[1][0]
    ojo_y_izquierdo = coordenadas_iniciales_izquierdo[1][1]
    ojo_w_izquierdo = coordenadas_iniciales_izquierdo[1][2]    
    ojo_h_izquierdo = coordenadas_iniciales_izquierdo[1][3]

    new_eye_izquierdo = escalado_nueva_cara[ojo_y_izquierdo:ojo_y_izquierdo+ojo_h_izquierdo,ojo_x_izquierdo:ojo_x_izquierdo+ojo_w_izquierdo]

    #guardamos las coordenadas del ojo derecho y sacamos la posición relativa del mismo segun la nueva imagen de la cara
    ojo_x_derecho = coordenadas_iniciales_derecho[1][0]
    ojo_y_derecho = coordenadas_iniciales_derecho[1][1]
    ojo_w_derecho = coordenadas_iniciales_derecho[1][2]    
    ojo_h_derecho = coordenadas_iniciales_derecho[1][3]
    new_eye_derecho = escalado_nueva_cara[ojo_y_derecho:ojo_y_derecho+ojo_h_derecho,ojo_x_derecho:ojo_x_derecho+ojo_w_derecho]
    
    #pintamos los rectángulos de la zona de los ojos en la imagen
    cv.rectangle(escalado_nueva_cara, (ojo_x_izquierdo, ojo_y_izquierdo), (ojo_x_izquierdo+ ojo_w_izquierdo, ojo_y_izquierdo + ojo_h_izquierdo), (0, 0, 255), 2)
    cv.rectangle(escalado_nueva_cara, (ojo_x_derecho, ojo_y_derecho), (ojo_x_derecho+ ojo_w_derecho, ojo_y_derecho + ojo_h_derecho), (0, 0, 255), 2)
    



    #sacamos la dimension original para poder pintar sobre la imagen original los resultados obtenidos
    dimension_original= ni_img_face.shape
    img_face_rectangulos = cv.resize(escalado_nueva_cara,(dimension_original[0],dimension_original[1]), interpolation=cv.INTER_CUBIC)
    

    
    #añadimos a la imagen original la nueva obtenida, de forma que obtenemos la cara original con los cuadros pintados
    n_img_entera[ccara[1]:ccara[1]+ccara[3],ccara[0]:ccara[0]+ccara[2],:] = img_face_rectangulos
    

    return  new_eye_izquierdo,new_eye_derecho,n_img_entera #class_eye(new_eye,ojo_comprobador)


##############################################################



#funcion que dada dos imágenes de un ojo y un rango determinado, predice si se encuentra la suficiente diferencia entre ambas
#para determinar si hay un guiño

#parametros de entrada: img_clas -> imagen del ojo que se usará para comprobar sus diferencias
#                       img_prueba -> imagen de la cual se determinará si tiene suficiente similitud
#                       rango -> valor que se usara para comprobar el guiño
#                       get_value -> parametro booleano que determinara si se devuelve el valor o el booleano

#parametros de salida  value -> relación de píxeles blancos iguales entre ambas imágenes
#                      clasificación -> valor booleano de si se detecta guiño o no

def class_eye(img_clas,img_prueba,rango,get_value):
    
    #iniciamos los parámetros para hacer una umbralización teniendo en cuenta los valores más oscuros
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([80, 80, 80])
    
    #realizamos la umbraliación con estos colores
    img_binaria_prueba = cv.inRange(img_prueba, lower_color,upper_color)
    img_binaria_clasificador = cv.inRange(img_clas, lower_color,upper_color)
    
    #sacamos el shape de la imagen clasificadora
    (f,c) = np.shape(img_binaria_clasificador)
    
    #iniciamos unos contadores que determinarán la cantidad de píxeles blancos en la clasificadora y la cantidad de píxeles blancos iguales en ambas
    eq_amount = 0
    tl_amount = 0
    
     
    #realizamos la cuenta de estos valores
    for i in range(f):
        for j in range(c):
            if img_binaria_clasificador[i][j] == 255:
                tl_amount= tl_amount + 1
                
                if img_binaria_prueba[i][j] == 255:
                    eq_amount = eq_amount + 1

                    
    if get_value:
        #devolvemos el valor de la relación
        return eq_amount/tl_amount
        
    else:
        #devolvemos el valor booleano de si este valor pertenece al rango
        return eq_amount/tl_amount < rango
    
    
############################################################## FUNCIÓN PRINCIPAL #####################################################

# Función para contabilizar la cantidad de guiños que se producen en un video.

# --Parametros de entrada: Como entrada se tendrá en cuenta la ruta del vídeo que queremos procesar y la ruta donde queramos que se guarde el nuevo video
    
# --Prametros de salida: Escribirá un nuevo video en memoria según la ruta indicada, y la cantidad de guiños para cada ojo.

def blinks_in_video(video_entrada,video_salida,blinkframes=10, likeness_factor = 0.8):
    
    #se lee el video y se guarda la cantidad de frames que se usan
    cap = cv.VideoCapture(video_entrada)
    success,image = cap.read()
    
 
    frames_img_list = []

    #iteramos sobre los frames para guardar cada uno de ellos en la lista frames_img_list
    while success:
        
        frames_img_list.append(image)
        success,image = cap.read()

    #liberamos el video
    cv.destroyAllWindows()
    cap.release()
        
    #guardamos el primer frame del video, del cual sacaremos los diferentes datos para comparar
    frame_principal = frames_img_list[0]
    
    #hacemos un preprocesado de la imagen principal
    frame_principal_preprocesado = rescale(rgbSmoothing(frame_principal))

    #guardamos la cara detectada y le hacemos el preprocesado para obtener los ojos
    zona_cara,imagen_pintada_inicial,_ = detectFaces(frame_principal_preprocesado,scale=1.1)
    zona_cara_eq = lightnessEqualization(zona_cara)
  
    #sacamos las coordendas de los ojos así como su imagen inicial
    ojo_inicial_derecho,coordenadas_iniciales_derecho,recorte_inicial_derecho = saca_ojo(zona_cara_eq,derecho=True)
  
    ojo_inicial_izquierdo,coordenadas_iniciales_izquierdo,recorte_inicial_izquierdo = saca_ojo(zona_cara_eq,derecho=False)
    
    #sacamos los valores de los segundos ojos para poder obtener su rango para el valor de clasificación
    segundo_ojo_izq,segundo_ojo_der,primer_frame_nuevo = muestra_ojos(rescale(rgbSmoothing(frames_img_list[1])),coordenadas_iniciales_izquierdo,coordenadas_iniciales_derecho,zona_cara)
    val_clas_izq = class_eye(ojo_inicial_izquierdo,segundo_ojo_izq,0,True) *2/3
    val_clas_der = class_eye(ojo_inicial_derecho,segundo_ojo_der,0,True) *2/3

  
    
    #Inicializamos una lista que contará los resultados de las clasificaciones según los frames, así como los contadores de guiños.
    l_blink_izq = [False]
    l_blink_der = [False]
    cnt_derecho = 0
    cnt_izquierdo = 0

    #sacamos las dimensiones de la imagen para determinar el tamaño de los frames del video
    height,width,layers = np.shape(primer_frame_nuevo)
 
    
    #creamos un nuevo video con 24fps y el tamaño del frame inicial
    new_video = cv.VideoWriter(video_salida,0,24,(width,height))
    new_video.write(primer_frame_nuevo)
    
    #iteramos sobre los diferentes frames para comprobar si ha habido un guiño entre ellos, ademas devolvemos los datos necesarios
    #para poder pintar sobre la imagen la zona de la cara y de los ojos, y lo escribimos sobre el video
    for i in range(1,len(frames_img_list)):
        #sacamos los ojos y la imagen de la cara
        ojo_izquierdo,ojo_derecho,cuadrado_entero = muestra_ojos(rescale(rgbSmoothing(frames_img_list[i])),coordenadas_iniciales_izquierdo,coordenadas_iniciales_derecho,zona_cara)
        
        #calculamos si ha habido guiño en ambos ojos y guardamos el valor en las listas
        l_blink_izq.append(class_eye(ojo_inicial_izquierdo,ojo_izquierdo,val_clas_izq,False))
        l_blink_der.append(class_eye(ojo_inicial_derecho,ojo_derecho,val_clas_der,False))
        
        #escribimos el nuevo frame con los rectángulos en el video
        new_video.write(cuadrado_entero)
        
    array_izquierdo = np.array(l_blink_izq)
    array_derecho = np.array(l_blink_der)
    

    
    
    
    #calculamos si entre los últimos 20 frames ha habido 10 con clasificación de guiño, en ese caso aumentamos el contador y reseteamos
    #los valores necesarios para que el mismo no se cuente múltiples veces
    # for i in range(20,len(frames_img_list)):
        
    #     if(np.sum(array_izquierdo[i-20:i])>10):
    #         array_izquierdo[i-20:i] = False
    #         cnt_izquierdo += 1
    #         j=i
    #         while np.sum(array_izquierdo[j:j+10])>=8:
    #             j=j+1
    #         array_izquierdo[i:j+10] = False
               
                
    #     if(np.sum(array_derecho[i-20:i])>10):
    #         array_derecho[i-20:i] = False
    #         cnt_derecho += 1
    #         j=i
    #         while np.sum(array_derecho[j:j+10])>=8:
    #             j=j+1
    #         array_derecho[i:j+10] = False
            
            
    margen_frames= 2*blinkframes
    for i in range(margen_frames,len(frames_img_list)):
        
        if(np.sum(array_izquierdo[i-margen_frames:i])>blinkframes):
            array_izquierdo[i-margen_frames:i] = False
            cnt_izquierdo += 1
            j=i
            while np.sum(array_izquierdo[j:j+10])>=8:
                j=j+1
            array_izquierdo[i:j+10] = False
               
                
        if(np.sum(array_derecho[i-margen_frames:i])>blinkframes):
            array_derecho[i-margen_frames:i] = False
            cnt_derecho += 1
            j=i
            while np.sum(array_derecho[j:j+10])>=8:
                j=j+1
            array_derecho[i:j+10] = False
            
    
    cv.destroyAllWindows()
    new_video.release()
    
    return (cnt_derecho,cnt_izquierdo)
        
        
#######
    
    

#FUNCIÓN MAIN

def main():
    args = sys.argv
    
    
    

    if len(args) >5 or len(args)<3:
        raise SystemError("La aplicación toma como mínimo 2 parámetros de entrada y como máximo 3.")
        
        
    video_entrada = args[1]
    video_salida= args[2]
    print("Procesando video....")
    if len(args)==3:
        cnt_derecho,cnt_izquierdo=blinks_in_video(video_entrada, video_salida)
        
    else:
        frames= int(args[3])
        if len(args)==4:
            cnt_derecho,cnt_izquierdo=blinks_in_video(video_entrada, video_salida,frames)
        else:
            likenes_factor = args[4]
            cnt_derecho,cnt_izquierdo=blinks_in_video(video_entrada, video_salida,frames,likenes_factor)

        
    print("Se han detectado {} guiños del ojo izquierdo y {} del ojo derecho".format(cnt_izquierdo,cnt_derecho))
    
    

if __name__ == "__main__":
    main()
    
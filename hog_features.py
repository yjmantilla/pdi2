#--------------------------------------------------------------------------
#------- Extraccion De Caracteristicas HOG ----------------------------------------------
#------- Para Reconocimiento de Placas-------------------------------------------
#------- Por: Yorguin José Mantilla Ramos    yorguinj.mantilla@udea.edu.co --------------
#-------      Estudiante de Ingeniería Electrónica -----------------
#-------      CC 1127617499 , Tel +13053591904,  Wpp +573115154452 -------------------
#------- Curso Básico de Procesamiento de Imágenes y Visión Artificial-----
#------- Abril de 2023--------------------------------------------------
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#--Importacion de Librerias  -----------------------------------------------
#--------------------------------------------------------------------------
# Ploteo
import matplotlib.pyplot as plt


# Procesamiento de Imagenes
from skimage.feature import hog
from skimage import data, exposure,io,color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import area_opening
from skimage.measure import label
from skimage.transform import resize

# Interaccion con Sistema Operativo
import os
import glob

# Basicas
import numpy as np
import scipy

# Paralelizacion del procesamiento
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil


#--------------------------------------------------------------------------
#-- Definicion de locaciones de caracteres de forma manual  -----------------------------------------------
#--------------------------------------------------------------------------

locations={
0:(10,30,60,130),
1:(65,30,115,130),
2:(120,30,170,130),
3:(205,30,255,130),
4:(270,30,320,130),
5:(325,30,375,130),
}

# Funcion para equalizar el histograma
def stretchlim(img,method='adaptive'):
    # Enfoque estadistico por percentiles
    if method == 'percentile':
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        img2 = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Enfoque 
    elif method == 'equalization':
        # Metodo Clásico
        img2 = exposure.equalize_hist(img)
    elif method == 'adaptive':
        # Metodo Adaptativo
        img2 =exposure.equalize_adapthist(img, clip_limit=0.03)
    return img2

# Funcion auxiliar para binarizar
def binarize(x,l):
    return (x > l)*1

# Funcion high-level para obtener las letras de una imagen de placa
def get_letters(img_path,shortcut=False,locations=locations):

    # Extraer caracteres de la placa para crear los labels (targets)
    # del dataset para la metodologia de machine learning
    filename=os.path.basename(img_path).replace('.jpg','')
    assert len(filename)==6
    chars = filename

    # Lectura de la imagen

    image= io.imread(img_path) # Lectura
    image = color.rgb2gray(image) # Conversion a grises
    image=stretchlim(image) # Equalización

    image=1-image # Inversion de los valores
    l=threshold_otsu(image) # Umbralizacion automatica mediante metodo Otsu
    image = binarize(image,l) # Binarizacion mediante el umbral encontrado por Otsu

    if not shortcut: # Vía honesta de acondicionamiento de las letras
        image = clear_border(image) # Limpieza de los bordes

        image = area_opening(image) # Operador de apertura para crear la region por cada letra

    #plt.imshow(image)
    #plt.show()

    # Etiquetado de regiones, cortamos a la seccion que contiene el texto del medio (los caracteres de la placa en sí)
    labeled_image = label(image[30:130,:],connectivity=2,background=1)

    # Obtener la cantidad de etiquetas y cuantos pixeles tiene cada una, para un sanity check
    labels,counts = np.unique(labeled_image,return_counts=True)

    # asumimos que las letras son las primeras 7 (Background y las 6 letras)
    labels = labels[:7]
    counts = counts[:7]

    # Listas auxiliares para guardar informacion letra por letra
    letters = [] # Las imagenes de cada letra en sí
    features = [] # Los valores Hogs
    hogs = [] # Las imagenes de visualizacion del hog
    figs = [] # La figura que compara la letra con la imagen hog en un mismo plot
    
    # Listas para guardar coordenadas del bounding box de cada letra
    minjs = []
    maxjs = []
    minis = []
    maxis = []
    bounds = []

    # atajo de segmentacion de las letras (localizacion manual)
    if shortcut:
        labels = [0,1,2,3,4,5,6]
    for l,la in enumerate(labels[1:]):# Skipeamos el background
        if shortcut:
            # Segmentacion manual
            ys = np.arange(locations[l][0],locations[l][2]) # Obtenemos los bounding box y sus correspondientes coordenadas x,y
            xs = np.arange(locations[l][1],locations[l][3])

            # Ubicamos dichos limites desde el punto de vista de los indices del array

            # Obtenemos los indices de las 4 esquinas que encierran el bounding box
            idxs = np.array(list(product(xs,ys)))
            idxs =(idxs[:,0],idxs[:,1])
            labeled_image=image 
        else:
            idxs=np.where(labeled_image==la)
        #Identificacion de las 4 esquinas x.y
        mini,maxi = min(idxs[0]),max(idxs[0])
        minj,maxj = min(idxs[1]),max(idxs[1])

        minjs.append(minj)
        maxjs.append(maxj)
        minis.append(mini)
        maxis.append(maxi)

        # Definicion del bounding box
        bounding_box = [minj,mini,maxj,maxi]
        bounds.append(bounding_box)

        # Estandarizacion del tamaño de la imgagen caracter
        standard_size =(100, 50)
        letter = resize(labeled_image[mini:maxi,minj:maxj],standard_size ,anti_aliasing=False) #labeled_image[]>0?
        # plt.imshow(letter)
        # plt.show()

        letters.append(letter)

        # Aplicacion del HOG
        fd, hog_image = hog(letter, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

        features.append(fd)
        hogs.append(hog_image)

        # Ploteo de los resultados
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(letter, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Equalizamos para que las caracteristicas del gradiente sea vean mejor

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        # Ploteo
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        figs.append(fig)

    # Asociamos Etiquetas con las letras del nombre
    idxs=scipy.stats.rankdata(minjs).astype(int)-1
    chars = np.array([x for x in chars],dtype=object)[idxs]
    #[x.show() for x in figs]
    #TODO: assert order of chars or info in the other lists
    return chars,letters,features,hogs,bounds,figs

# Funcion para ejecutar la extraccion de caracteristicas sobre cada ruta de una imagen
def pipeline(xpath,verbose=False):
    # obtenemos los caracteres de la placa
    placa = os.path.basename(xpath).replace('.jpg','')

    # Generamos los archivos para todas las letras dentro de  la imagen
    files = [os.path.join(single_char_path,placa+f'-{char}-{i}_both.jpg') for i,char in enumerate(placa)]

    # Verificamos si ya hemos procesado la imagen
    done = [os.path.isfile(x) for x in files]
    done = all(done)

    # Caso donde ya procesamos la imagen
    if done:
        if verbose:
            print(f'{placa} files existed... skipping!')

    # Si aun no, ejecutamos el algoritmo
    if not done:

        # Obtenemos las letras
        a=get_letters(xpath,shortcut=True)
        if verbose:
            print(xpath,end=' ',flush=True)

        # Ploteo y guardado de las imagenes para monitear el calculo de los hogs
        iii=0
        for char,letter,feature,_hog,bound,fig in zip(*a):
            if verbose:
                print(char,end='',flush=True)

            # Guardado de las imagenes de monitoreo
            outpath = os.path.join(single_char_path,placa+f'-{char}-{iii}_both.jpg')
            iii+=1
            if not os.path.isfile(outpath):

                # Ploteo de las imagenes para monitoreo
                fig.savefig(outpath)
                fig,ax=plt.subplots(1)
                ims=ax.imshow(_hog)
                fig.savefig(outpath.replace('_both','_hog'))
                fig,ax=plt.subplots(1)
                ims=ax.imshow(letter)
                fig.savefig(outpath.replace('_both','_char'))
                plt.close('all')
                np.save(outpath.replace('_both','_hog').replace('.jpg','.npy'),_hog)
                np.save(outpath.replace('_both','_char').replace('.jpg','.npy'),letter)
                np.save(outpath.replace('_both','_feature').replace('.jpg','.npy'),feature)
            else:
                if verbose:
                    print(f'{outpath} already existed...skipping!')
        if verbose:
            print('')

if __name__=='__main__':

    # Ejecucion del pipeline sobre todas las imagenes

    # Obtenemos todas las imagenes
    image_paths = [x.replace('\\','/') for x in glob.glob('placas/dataset/*.jpg')]

    # Directorio de salida del algoritmo
    single_char_path = "Y:/code/pdi2/placas/single-char-dataset2"
    os.makedirs(single_char_path,exist_ok=True)

    # Paralelizacion
    njobs=len(psutil.Process().cpu_affinity())
    # test: pipeline(image_paths[1])
    result = Parallel(n_jobs=njobs)(delayed(pipeline)(xpath) for xpath in tqdm(image_paths))

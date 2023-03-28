import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure,io,color
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import area_opening
from skimage.measure import label
from skimage.transform import resize
import os
import numpy as np
import glob
import scipy
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil


locations={0:(10,30,60,130),
1:(65,30,115,130),
2:(120,30,170,130),
3:(205,30,255,130),
4:(270,30,320,130),
5:(325,30,375,130),
}
def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def stretchlim(img,method='adaptive'):

    # Contrast stretching
    if method == 'percentile':
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        img2 = exposure.rescale_intensity(img, in_range=(p2, p98))
    elif method == 'equalization':
        # Equalization
        img2 = exposure.equalize_hist(img)
    elif method == 'adaptive':
        # Adaptive Equalization
        img2 =exposure.equalize_adapthist(img, clip_limit=0.03)
    return img2

def binarize(x,l):
    return (x > l)*1


def get_letters(img_path,shortcut=False,locations=locations):

    filename=os.path.basename(img_path).replace('.jpg','')
    assert len(filename)==6
    chars = filename

    image= io.imread(img_path)
    image = color.rgb2gray(image)
    image=stretchlim(image)

    image=1-image
    l=threshold_otsu(image)
    image = binarize(image,l)

    if not shortcut:
        image = clear_border(image)

        image = area_opening(image)[30:130,:]

    #plt.imshow(image)
    #plt.show()

    #labeling
    labeled_image = label(image,connectivity=2)
    labels,counts = np.unique(labeled_image,return_counts=True)

    # keep first 7 (background+6 letters)

    labels = labels[:7]
    counts = counts[:7]
    letters = []
    features = []
    hogs = []
    figs = []
    minjs = []
    maxjs = []
    minis = []
    maxis = []
    bounds = []
    if shortcut:
        labels = [0,1,2,3,4,5,6]
    for l,la in enumerate(labels[1:]):#skip background
        if shortcut:
            ys = np.arange(locations[l][0],locations[l][2])
            xs = np.arange(locations[l][1],locations[l][3])
            idxs = np.array(list(product(xs,ys)))
            idxs =(idxs[:,0],idxs[:,1])
            labeled_image=image
        else:
            idxs=np.where(labeled_image==la)
        #identify border x,y
        mini,maxi = min(idxs[0]),max(idxs[0])
        minj,maxj = min(idxs[1]),max(idxs[1])

        minjs.append(minj)
        maxjs.append(maxj)
        minis.append(mini)
        maxis.append(maxi)
        bounding_box = [minj,mini,maxj,maxi]
        bounds.append(bounding_box)

        # Assume label 0 is background
        standard_size =(100, 50)
        letter = resize(labeled_image[mini:maxi,minj:maxj],standard_size ,anti_aliasing=False) #labeled_image[]>0?
        # plt.imshow(letter)
        # plt.show()

        letters.append(letter)

        fd, hog_image = hog(letter, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

        features.append(fd)
        hogs.append(hog_image)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(letter, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        figs.append(fig)
    idxs=scipy.stats.rankdata(minjs).astype(int)-1
    chars = np.array([x for x in chars],dtype=object)[idxs]
    #[x.show() for x in figs]
    #TODO: assert order of chars or info in the other lists
    return chars,letters,features,hogs,bounds,figs

def pipeline(xpath,verbose=False):
    placa = os.path.basename(xpath).replace('.jpg','')
    files = [os.path.join(single_char_path,placa+f'-{char}-{i}_both.jpg') for i,char in enumerate(placa)]
    done = [os.path.isfile(x) for x in files]
    done = all(done)
    if done:
        if verbose:
            print(f'{placa} files existed... skipping!')
    if not done:
        a=get_letters(xpath,shortcut=True)
        if verbose:
            print(xpath,end=' ',flush=True)
        iii=0
        for char,letter,feature,_hog,bound,fig in zip(*a):
            if verbose:
                print(char,end='',flush=True)

            outpath = os.path.join(single_char_path,placa+f'-{char}-{iii}_both.jpg')
            iii+=1
            if not os.path.isfile(outpath):
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
    image_paths = [x.replace('\\','/') for x in glob.glob('placas/dataset/*.jpg')]
    single_char_path = "Y:/code/pdi2/placas/single-char-dataset"
    os.makedirs(single_char_path,exist_ok=True)
    njobs=len(psutil.Process().cpu_affinity())
    result = Parallel(n_jobs=njobs)(delayed(pipeline)(xpath) for xpath in tqdm(image_paths))

from scipy.stats import skew,kurtosis
import numpy
import matplotlib.pyplot as plt
import numpy as np

class Indices():
    def __init__(self,src):
        pass
    def __get_stats__(self):
        index_data=np.ma.masked_invalid(self.index_array).compressed()
        data_stats={}
        data_stats["Max"]=index_data.max()
        data_stats["Mean"]=index_data.mean()
        data_stats["Median"]=np.median(index_data)
        data_stats["Min"]=index_data.min()
        data_stats['skew']=skew(index_data)
        data_stats['kurtosis']=kurtosis(index_data)
        data_stats['skew x kurtosis']=data_stats['skew']*data_stats['kurtosis']
        return data_stats
        
    def get_histogram(self,ax=None):

        array_flat = self.index_array.flatten()

        # Plot the histogram
        if ax!=None:
          ax.hist(array_flat, bins=50, color='green', alpha=0.8)
        else:
          plt.hist(array_flat, bins=50, color='green', alpha=0.8) 

        ax.set_xlabel('NDVI')
        ax.set_ylabel('Frequency')
        ax.set_title('NDVI Histogram')
        

    def get_heatmap(self,ax=None,cmap="RdYlGn"):
       
       if ax!=None:
         img=ax.imshow(self.index_array, cmap=cmap)
       else:
         plt.imshow(self.index_array, cmap=cmap)
        
       plt.colorbar(img,ax=ax)

    def get_thres_region(self,ax,threshold=0.7):

       red_band = self.src.read(1)
       # Highlight stressed regions on the original image
       stressed_regions = np.where(self.index_array<threshold, 1, 0)
       highlighted_image = red_band.copy()  # Use the red band as a base for highlighting
       highlighted_image[stressed_regions == 1] = 255  # Set stressed regions to maximum intensity

       # Plot the original image with stressed regions highlighted
       #cmap = plt.cm.colors.ListedColormap(['white', 'red'])
       img=ax.imshow(highlighted_image, cmap='gray')
       plt.colorbar(img,ax=ax)


class ndvi(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_ndvi__()

    def __calculate_ndvi__(self):
        red_band = np.nan_to_num(self.src.read(self.band_identifier['Red']))
        nir_band = np.nan_to_num(self.src.read(self.band_identifier['NIR']))
        ndvi = (nir_band.astype(float)-red_band.astype(float))/(nir_band.astype(float)+red_band.astype(float))
        return ndvi    

class vari(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_vari__()

    def __calculate_vari__(self):
        red_band = self.src.read(self.band_identifier['Red'])
        green_band = self.src.read(self.band_identifier['Green'])
        vari = (green_band.astype(float)-red_band.astype(float))/(green_band.astype(float)+red_band.astype(float))
        return vari

class gli(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_gli__()

    def __calculate_gli__(self):
        red_band = self.src.read(self.band_identifier['Red'])
        green_band = self.src.read(self.band_identifier['Green'])
        gli = (2*green_band.astype(float)-red_band.astype(float))/(2*green_band.astype(float)+red_band.astype(float))
        return gli

class nir(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_gli__()

    def __calculate_gli__(self):
        nir_band = self.src.read(self.band_identifer['NIR'])
        return nir_band

class bai(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_bai__()

    def __calculate_bai__(self):
        red_band = np.nan_to_num(self.src.read(self.band_identifier['Red']))
        nir_band = np.nan_to_num(self.src.read(self.band_identifier['NIR']))
        bai= 1/((0.06-(nir_band.astype(float)/10000)**2)+(1+(red_band.astype(float)/10000)**2))
        return bai

class bsi(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_bsi__()

    def __calculate_bsi__(self):
        red_band = np.nan_to_num(self.src.read(self.band_identifier['Red']).astype(float))
        nir_band = np.nan_to_num(self.src.read(self.band_identifier['NIR']).astype(float))
        green_band = np.nan_to_num(self.src.read(self.band_identifier['Green']).astype(float))
        blue_band = self.src.read(self.band_identifier['Blue']).astype(float)
        bsi= ((((nir_band+green_band)-(red_band+blue_band))/(nir_band+green_band+red_band+blue_band))*100)+100
        return bsi

class char(Indices):

    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_char__()

    def __calculate_char__(self):
        red_band = np.nan_to_num(self.src.read(self.band_identifier['Red']).astype(float))
        blue_band = np.nan_to_num(self.src.read(self.band_identifier['Blue']).astype(float))
        green_band = np.nan_to_num(self.src.read(self.band_identifier['Green']).astype(float))

        char= (blue_band+green_band+red_band)+np.maximum(np.abs(blue_band-green_band),np.abs(blue_band-red_band),np.abs(red_band-green_band))*15
        return char

class sr(Indices):
    def __init__(self, src,band_identifier):
        self.src=src
        self.band_identifier=band_identifier
        self.index_array=self.__calculate_sr__()

    def __calculate_sr__(self):
        nir_band = np.nan_to_num(self.src.read(self.band_identifier['NIR']).astype(float))
        red_band = np.nan_to_num(self.src.read(self.band_identifier['Red']).astype(float))
        sr=nir_band/red_band
        return sr


    


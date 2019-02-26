# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:21:11 2019

__author__ = 'David Whitney'
__email__ = 'david.whitney@mpfi.org'
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import scipy.ndimage as snd
from scipy.io import loadmat,savemat

# Setup binned HSV colormap
LUT_colors = [
    [    0.,    1.0000,        0.],
    [0.1569,    1.0000,        0.],
    [0.5490,    1.0000,        0.],
    [0.8235,    1.0000,        0.],
    [1.0000,    1.0000,        0.],
    [1.0000,    0.8235,        0.],
    [1.0000,    0.5882,        0.],
    [1.0000,    0.3529,        0.],
    [1.0000,        0.,        0.],
    [1.0000,        0.,    0.4784],
    [0.7059,        0.,    1.0000],
    [0.3922,        0.,    1.0000],
    [    0.,        0.,    1.0000],
    [    0.,    0.3137,    1.0000],
    [    0.,    0.5490,    0.5882],
    [    0.,    0.7843,    0.1961]]
hsv_LUT = mpl.colors.LinearSegmentedColormap.from_list(
          'hsv_paper', LUT_colors, N=16)

class OPM():
    # Class for the orientation preference map
    def __init__(self,folder_path=str('')):
        self.folder_path = folder_path
        if len(folder_path)>0:
            self.load_data(folder_path)
            self.generate_OPM()
            self.spatial_filter_OPM()
                
    def load_data(self,folder_path=str('')):
        """load mask and trial-averaged, single-condition response maps to unidirectionally drifting gratings"""
        assert type(folder_path) in [str,unicode], 'folder_path must be a string'
        if len(folder_path)>0:
            file_path = folder_path+'/opm_data.mat'
            assert os.path.isfile(file_path),'opm_data.mat not found at specified path: {}'.format(file_path)
            self.data = loadmat(file_path) # Loads a dictionary containing the imaging data
            self.mask = self.data['mask'].astype('bool')
    
    def generate_OPM(self, normalizeData=False):
        """generate complex-field orientation map by computing the vector sum of the single-condition maps"""
        scms = np.copy(self.data['scms'])
        [h,w,n_conds] = scms.shape
        angle_list   = np.linspace(0,2*np.pi,n_conds+1)
        phaseVector  = np.exp(1j*2*angle_list[:-1])
        if normalizeData: # cocktail blank normalizes array
            scms = scms / np.repeat(np.sum(scms,axis=2).reshape([h,w,1]),n_conds,axis=2)
        self.opm = np.sum(scms*np.tile(phaseVector,[h,w,1]),axis=2)
    
    def spatial_filter_OPM(self,sigma=15):
        """Spatial high-pass filter the orientation map at the specified cutoff (sigma)"""
        mask = self.mask & np.isfinite(self.opm)
        opm  = self.opm
        opm[~mask] = 0.0
        diffMaps = [np.real(opm),np.imag(opm)]
        for i in range(len(diffMaps)):
            diffMaps[i] = diffMaps[i] - 1.*snd.gaussian_filter(diffMaps[i],sigma,mode='constant',cval=0) \
                                          /snd.gaussian_filter(mask.astype(diffMaps[i].dtype),sigma,mode='constant',cval=0)
        self.opm = diffMaps[0]+1j*diffMaps[1]
        self.opm[~mask] = np.nan

    def get_OPM(self, isPolarMap=True, clip_val=0.1):
        """Return hsv angle map or an hsv polar map from the complex-field orientation preference map.
        For both angle and polar maps, hue denotes the preferred orientation. Magnitude in the polar map 
        indicates orientation selectivity."""
        h,w = self.opm.shape
        normalized_opm = np.mod(np.angle(self.opm),2*np.pi)/(2*np.pi)
        hsv_map = hsv_LUT(normalized_opm)
        if isPolarMap:
            mag_map = np.clip(abs(self.opm)/clip_val,0,1)
            hsv_map[:,:,:3] = hsv_map[:,:,:3]*np.repeat(mag_map.reshape(h,w,1),3,axis=2)
        mask = np.repeat(np.copy(self.mask).astype(hsv_map.dtype).reshape([h,w,1]),hsv_map.shape[2],axis=2)
        return hsv_map*mask;  

    def addOPMContours(self,ax=None,angles=[0,45]):
        """Add white contours to the current axis (ax) delineating the zero-crossings
        of the orientation preference map for the specified angles. Default is [0,45]."""
        if(type(ax)==type(None)):
            (fig,ax)=plt.subplots()
        [xx,yy]=np.meshgrid(np.arange(0,self.opm.shape[1]), \
                            np.arange(0,self.opm.shape[0]))
        for angle in angles:
            ax.contour(xx,yy,np.real(self.opm/np.exp(1j*2*np.pi*angle/180.)),0,colors='white')
        
if __name__=='__main__':
    make_dataset = True
    if make_dataset:
        ref_folder_path = r'G:\Python\spontaneousData\significanceMaps\F16550_significanceData.mat'
        dataDict = loadmat(ref_folder_path)
        data = dict()
        scms = np.transpose(dataDict['scms_unfiltered'][:,:,:,1],axes=[1,0,2])
        mask = dataDict['ROI'].astype('bool').T
        savemat('G:\Python\spontaneousData\pythonCode\IpythonNotebook\opm_data.mat',{'scms':scms,'mask':mask,'stim_directions':np.linspace(0,360,9)[:-1],'spatial_resolution':4000/153.})

    # Load our sample dataset
    folder_path = os.path.dirname(__file__)
    test_obj = OPM(folder_path)

    # Show the orientation preference map with zero-crossings contours
    (fig,ax)=plt.subplots()
    fig.suptitle('Orientation preference map')
    im=ax.imshow(test_obj.get_OPM(),cmap=hsv_LUT,vmin=0,vmax=180)
    test_obj.addOPMContours(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="10%", pad=0.10)
    cbar = fig.colorbar(im,cax=cax,ticks=np.arange(0,181,45), orientation='horizontal')

    # Show the trial-averaged single condition maps
    stim_directions = np.squeeze(test_obj.data['stim_directions'])
    n_conds = len(stim_directions)
    [fig,axes]=plt.subplots(nrows=2,ncols=np.ceil(n_conds/2.).astype('int16'),figsize=(8,4))
    fig.suptitle('Single-condition maps')
    for i, ax in enumerate(axes.flatten()):
        # Plot single-condition map
        im=ax.imshow(test_obj.data['scms'][:,:,i],cmap='gray',vmin=0,vmax=1.0)
        ax.set_title('{}$^\circ$'.format(stim_directions[i]))
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.10)
        fig.colorbar(im,cax=cax, ticks=np.arange(0,1.01,0.5), orientation='vertical')
    fig.tight_layout()
    plt.show(block=True)
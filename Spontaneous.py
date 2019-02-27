# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:04:05 2019

@author: whitneyd
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import scipy.ndimage as snd
from scipy.io import loadmat, savemat
import zipfile
import warnings
FILTER_CUTOFF = np.round(195*153/4000.) # This should be 7-8 pixels

def sub2ind(SIZ, rows, cols):
    """ returns the linear index equivalent to the row and column subscripts in
    the arrays I and J for a matrix of size SIZ. """
    return rows * SIZ[1] + cols

def ind2sub(siz, ind):
    """ returns the arrays ROWS and COLS containing the equivalent row and
    column subscripts corresponding to the index matrix IND for a matrix
    of size SIZ. """
    rows = (ind.astype('int') / siz[1])
    cols = (ind.astype('int') % siz[1])
    return (rows, cols)

def downsampleimg_stack(img_stack, downsample_factor, full_downsampling=True):
    "spatially downsample image by a whole number."""
    if len(img_stack.shape)==2:
        img_stack = img_stack[None, :, :]  # We add an extra dimension to accommodate a single image
    if full_downsampling:
        (n, x, y) = img_stack.shape
        x = x - np.mod(x, downsample_factor)
        y = y - np.mod(y, downsample_factor)
        new_stack = np.zeros((n, x / downsample_factor, y / downsample_factor), dtype=img_stack.dtype)
        for i in range(downsample_factor):
            for j in range(downsample_factor):
                new_stack += img_stack[:, i:x:downsample_factor, j:y:downsample_factor] / (downsample_factor ** 2)
    else:
        new_stack = img_stack[:, ::downsample_factor, ::downsample_factor]  # Downscale images images fast
    return np.squeeze(new_stack)
    
class Spontaneous():
    # Class for containing extracted spontaneous activity patterns, and assembling correlation patterns
    def __init__(self, folder_path=str('')):
        self.folder_path = folder_path
        if len(folder_path) > 0:
            self.load_data(folder_path)

    def load_data(self, folder_path=str('')):
        """load spontaneous time-series data, including an imaging stack containing the active event frames"""
        assert type(folder_path) in [str,unicode], 'folder_path must be a string'
        if len(folder_path) > 0:
            file_path = folder_path + '/spont_data.mat'
            assert os.path.isfile(file_path), 'spont_data.mat not found at specified path: {}'.format(file_path)
            self.data = loadmat(file_path)  # Loads a dictionary containing the imaging data
            self.mask = self.data['mask'].astype('bool')

            tseries_path = folder_path + '/spont_time_series.zip'
            if(os.path.isfile(tseries_path)):
                with zipfile.ZipFile(tseries_path, 'r') as zip:
                    self.data['spont_time_series_3Hz'] = zip.read('spont_time_series.npy')
            else:
                warningMessage = 'Warning: Could not find spont_time_series.zip at specified path: {}/n. Time-series data will not be included.'.format(tseries_path)
                warnings.warn(warningMessage)

    def spatial_filter_events(self,sigma=FILTER_CUTOFF):
        """Spatial high-pass filter the active event frames at the specified cutoff (sigma)"""
        mask = self.mask
        event_frames = self.data['spont_event_frames']
        nFrames,h,w = event_frames.shape
        for i in range(nFrames):
            event_frame = event_frames[i,:,:]
            event_frame[~mask] = 0.0
            event_frame = event_frame - 1.*snd.gaussian_filter(event_frame,sigma,mode='constant',cval=0) \
                                          /snd.gaussian_filter(mask.astype(event_frame.dtype),sigma,mode='constant',cval=0)
            event_frame[~mask] = np.nan
            event_frames[i,:,:] = event_frame
        self.data['spont_event_frames'] = event_frames

    def get_correlation_table(self):
        """ Compute the pairwise correlation of every pixel for the active event frames."""
        mask = self.mask
        active_frames = self.data['spont_event_frames']
        nFrames, h, w = active_frames.shape
        active_frames = np.reshape(active_frames,[nFrames,h*w])[:,mask.flatten()].T
        active_frames = active_frames - np.tile(np.nanmean(active_frames, axis=1), [active_frames.shape[1], 1]).T # Z-Score
        active_frames = active_frames / np.tile(np.nanstd(active_frames, axis=1), [active_frames.shape[1], 1]).T
        self.corrTable = np.dot(active_frames, active_frames.T) / active_frames.shape[1]
        return self.corrTable

    def get_correlation_pattern(self,seedPtLoc=[0,0]):
        """ Return a correlation pattern for the specified seed point location (X,Y)"""
        if ~hasattr(self, 'corrTable'):
            self.get_correlation_table()
        mask = self.mask
        labeledMask = mask.astype('int32').flatten()
        labeledMask[mask.flatten()] = np.arange(0,np.sum(mask))
        labeledMask = np.reshape(labeledMask,mask.shape)
        seedID = labeledMask[seedPtLoc[1],seedPtLoc[0]]
        assert seedID!=0, "Not a valid seed location. Must be within mask"
        corrPattern = 0*mask.flatten().astype('float32')
        corrPattern[mask.flatten()] = self.corrTable[:,seedID]
        corrPattern = np.reshape(corrPattern,mask.shape)
        return corrPattern
    
    def get_colormap(self):
        """ Return colormap where red indicates positive correlation, white is
        no correlation, and blue indicates anticorrelation """
        LUT_colors =  [mpl.cm.RdBu(n) for n in range(255)]
        return mpl.colors.LinearSegmentedColormap.from_list('BuRd', LUT_colors[::-1], N=255)
            
if __name__=='__main__':
    # save file
    make_dataset = False
    if make_dataset:
        save_directory = 'G:/Python/spontaneousData/pythonCode'
        ferret_name    = '1655'
        expt_date      = '2014-04-22'
        time_series_ID    = 1
        downsample_factor = 2
        full_downsampling = True
       
        # Get spatially-downsampled time series and mask
        DF_by_F0 = np.load('{}/F{}/{}/tseries_{}/DF_by_F0.npy'.format(save_directory,ferret_name,expt_date,time_series_ID))
        mask     = np.load('{}/F{}/{}/tseries_{}/ROI.npy'.format(save_directory,ferret_name,expt_date,time_series_ID))
        DF_by_F0 = np.reshape(DF_by_F0,[mask.shape[0],mask.shape[1],DF_by_F0.shape[1]])
        DF_by_F0 = np.transpose(DF_by_F0,axes=[2,1,0])
        DF_by_F0 = downsampleimg_stack(DF_by_F0,downsample_factor,full_downsampling)
        mask     = downsampleimg_stack(mask.astype('float32'), downsample_factor, full_downsampling)
        mask     = (mask.T * np.min(np.isfinite(DF_by_F0), axis=0))==1

        # Get frame IDs and create an imaging stack containing all of the active spontaneous frames
        event_frames = np.load('{}/F{}/{}/tseries_{}/best_frames_new.npy'.format(save_directory,ferret_name,expt_date,time_series_ID))
        large_frames = np.load('{}/F{}/{}/tseries_{}/per_frame_data/frame_is_big.npy'.format(save_directory,ferret_name,expt_date,time_series_ID))
        event_frame_ID = event_frames[large_frames[event_frames]]
        spont_event_frames = DF_by_F0[event_frame_ID,:,:].copy()

        # Temporally downsample time series
        [n_frames,w,h] = DF_by_F0.shape
        DF_by_F0_15Hz  = DF_by_F0.reshape([n_frames,w*h])[:,mask.flatten()]
        DF_by_F0_3Hz   = np.zeros([DF_by_F0_15Hz.shape[0]/5,DF_by_F0_15Hz.shape[1]],dtype=DF_by_F0_15Hz.dtype)
        for i in range(5):
            DF_by_F0_3Hz = DF_by_F0_3Hz+DF_by_F0_15Hz[i::5,:]/5
        mean_response_trace_15Hz = np.nanmean(DF_by_F0_15Hz,axis=1)
        mean_response_trace_3Hz  = np.nanmean(DF_by_F0_3Hz ,axis=1)
        t_15Hz = np.arange(0,n_frames/1)/15.
        t_3Hz  = np.arange(0,n_frames/5)/3.
        event_frame_ID_15Hz = event_frame_ID
        event_frame_ID_3Hz  = np.round(event_frame_ID/5.)

        # Save files
        savemat('G:\Python\spontaneousData\pythonCode\IpythonNotebook\spont_data.mat',
                {'spont_event_frames':spont_event_frames,                \
                 'mask':mask,                                            \
                 'spont_event_frameIDs_15Hz':event_frame_ID_15Hz,        \
                 'spont_event_frameIDs_3Hz':event_frame_ID_3Hz,          \
                 'mean_time_series_trace_15Hz':mean_response_trace_15Hz, \
                 'mean_time_series_trace_3Hz':mean_response_trace_3Hz,   \
                 'frame_timestamps_15Hz':t_15Hz,                         \
                 'frame_timestamps_3Hz':t_3Hz})
        time_series_file_path = 'G:\Python\spontaneousData\pythonCode\IpythonNotebook\spont_time_series'
        np.save(time_series_file_path+'.npy',DF_by_F0_3Hz)
        zip = zipfile.ZipFile(time_series_file_path+'.zip', mode='w')
        zip.write(time_series_file_path+'.npy',os.path.basename(time_series_file_path+'.npy'))
        os.remove(time_series_file_path+'.npy')

    # Load our sample dataset and perform some processing
    folder_path = os.path.dirname(__file__)
    test_obj = Spontaneous(folder_path)
    test_obj.get_correlation_table()
    test_obj.spatial_filter_events()

    # Show the average fluorescence changes of the time-series for a 15s window and some active event frames
    time_window = [15*100,15*115]
    event_frame_IDs  = test_obj.data['spont_event_frameIDs_15Hz'].flatten()
    time_stamps      = test_obj.data['frame_timestamps_15Hz'].flatten()
    mean_fluor_trace = test_obj.data['mean_time_series_trace_15Hz'].flatten()
    event_loc = 0 * time_stamps
    event_loc[event_frame_IDs] = 1+np.arange(len(event_frame_IDs))
    unique_events = np.unique(event_loc[time_window[0]:time_window[1]])[1:].astype('int16')
    n_uniq_events = len(unique_events)

    fig, axes = plt.subplots(nrows=2,ncols=n_uniq_events,figsize=(12,4))
    axes = axes.flatten()
    ax = plt.subplot2grid((2, n_uniq_events), (0, 0), colspan=n_uniq_events)
    ax.plot(time_stamps, mean_fluor_trace, 'g') # plot the fluorescence time series for a cell
    [ax.scatter(time_stamps[i],0,marker = '^', color = 'black') for i in range(len(time_stamps)) if event_loc[i]>0]
    ax.set_ylabel('Mean $\Delta$F/F')
    ax.set_xlabel('Time (s)')
    ax.set_xlim([time_window[0]/15.,time_window[1]/15.])
    for index in range(n_uniq_events):
        # Show the active frame
        ax = axes[index+n_uniq_events]
        im = ax.imshow(test_obj.data['spont_event_frames'][unique_events[index],:,:], cmap='gray', vmin=0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.10)
        fig.colorbar(im, cax=cax, orientation='vertical')
    fig.tight_layout()

    # Show a correlation pattern at given seed point
    seedPt = [55,55]
    corrPattern = test_obj.get_correlation_pattern(seedPt)
    fig,ax = plt.subplots()
    im=ax.imshow(corrPattern,cmap=test_obj.get_colormap(),vmin=-1,vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.10)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title("Seed location: [{X},{Y}]".format(X=seedPt[1],Y=seedPt[0]))
    ax.plot(seedPt[1],seedPt[0],'og')
    plt.show(block=True)
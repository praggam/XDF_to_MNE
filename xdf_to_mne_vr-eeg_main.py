# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:40:22 2021
Convert XDF File(s) to MNE FIF File(s)
@author: Philipp Raggam
"""

#%% Import libraries

import os
import glob
import pyxdf
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy


#%% Define parameters

# Subject directory
sub_str = 'Pilot_2'
# Paradigm directory
param_dir = 'resting_state'
# Recording directory
fpath = 'D:' + os.sep + 'VR-EEG' + os.sep + sub_str + os.sep + param_dir

# LSL stream names
stream_neurone, stream_markers, stream_hand_pos = 'NeuroneStream', 'VR_GAME', 'Hand_POS'
stream_names = [stream_neurone, stream_markers, stream_hand_pos]

# Number of EEG channels
n_chans_eeg = 64
# Sampling rate
fs = 1000
# Channel names EEG
chan_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8', 'Fz','Cz','Pz','Iz','FC1','FC2','CP1','CP2','FC5','FC6',
              'CP5','CP6','FT9','FT10','TP9','TP10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4','CP3','CP4','PO3','PO4','F5','F6','C5','C6','P5',
              'P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz','CPz','POz','Oz']

# Channel name event markers
chan_names.append('STIM')

# Hand position markers
right_x, right_y, right_z,left_x, left_y, left_z = 81, 82, 83, 91, 92, 93
hand_pos_list = [right_x, right_y, right_z, left_x, left_y, left_z]
# Number of hand pos channels
n_chans_hand_pos = len(hand_pos_list)
# Channel names hand pos
chan_names_hand_pos = ['HPRx','HPRy','HPRz','HPLx','HPLy','HPLz']
# Append channels
for chan in chan_names_hand_pos: chan_names.append(chan)


#%% EEG functions

def plot_eeg_stream(eeg_times,fs,fname,run_idx):    
    
    n_samples = len(eeg_times)
    # Show eeg stream consistency
    fig_eeg_times = plt.figure(figsize=(15,15))
    plt.scatter(np.arange(0,n_samples)/fs, eeg_times-np.min(eeg_times))
    plt.title('EEG sampling consistency - ' + sub_str + '_run_' + str(run_idx+1))
    plt.xlabel('time (s)'), plt.ylabel('time stamps')
    grid_step = 20 if n_samples/fs > 400 else 10
    plt.xticks(np.arange(0,n_samples/fs,step=grid_step)), plt.yticks(np.arange(0,eeg_times[-1]-np.min(eeg_times), step=grid_step))
    plt.xlim(0, n_samples/fs + 1), plt.ylim(0, eeg_times[-1]-np.min(eeg_times)+1)
    plt.grid(), plt.tight_layout(), plt.show()
    
    # Save figure
    fig_eeg_times.savefig(fname + '_eeg_sampling_consistency.png')
    # Close figure
    plt.close()


# %% Event markers functions

def generate_markers_channel(markers_array, markers_times, eeg_times):
    
    n_samples = len(eeg_times)
    n_markers = len(markers_times)
    markers_positions = np.zeros((n_markers))
    
    for m_idx in range(n_markers):
        if markers_times[m_idx] < eeg_times[-1]:
            index = np.where(eeg_times > markers_times[m_idx])[0][0]
        else:
            index = n_samples-1
            
        if index in markers_positions:
            markers_positions[m_idx] = markers_positions[m_idx-1]+1
        else:
            markers_positions[m_idx] = index
    
    markers_positions = markers_positions[np.where(markers_positions < n_samples)[0]]
    markers_data = np.zeros((1,n_samples))
    n_markers = len(markers_positions)
    
    for m_idx in range(n_markers):
        markers_data[0,int(markers_positions[m_idx])] = markers_array[m_idx]
        
    return markers_data


def plot_event_markers(markers_data, markers_array, eeg_times, fs, fname, run_idx):
    
    n_samples = len(eeg_times)
    
    # Plot event markers
    fig_markers, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, figsize=(25,15),gridspec_kw={'height_ratios': [1,19,25,25]})
    
    # plot the same data on both axes
    ax1.scatter(np.arange(0,n_samples)/fs , markers_data), ax2.scatter(np.arange(0,n_samples)/fs , markers_data)
    ax3.scatter(np.arange(0,n_samples)/fs , markers_data), ax4.scatter(np.arange(0,n_samples)/fs , markers_data)
    
    # zoom-in / limit the view to different portions of the data
    ax1.set_yticks(markers_array), ax1.set_ylim(400-1,400+1), ax1.grid()
    ax2.set_yticks(markers_array), ax2.set_ylim(300-1,330+1), ax2.grid()
    ax3.set_yticks(markers_array), ax3.set_ylim(200-1, 243+1), ax3.grid()  # most of the data
    ax4.set_yticks(markers_array), ax4.set_ylim(100-1, 142+1), ax4.grid()  # outliers only
    
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False), ax2.spines.bottom.set_visible(False)
    ax3.spines.top.set_visible(False), ax3.spines.bottom.set_visible(False)
    ax4.spines.top.set_visible(False)
    
    # don't put tick labels at the
    ax1.tick_params(bottom=False), ax2.tick_params(bottom=False), ax3.tick_params(bottom=False) 
    
    # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs), ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
    ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs), ax3.plot([0, 1], [0, 0], transform=ax3.transAxes, **kwargs)
    ax4.plot([0, 1], [1, 1], transform=ax4.transAxes, **kwargs)
    
    # title and axes labeling
    plt.suptitle('Event Markers - ' + sub_str + '_run_' + str(run_idx+1))
    ax4.set_xlabel('time (s)'), ax3.set_ylabel('marker number')
    
    # tight layout
    grid_step = 20 if n_samples/fs > 400 else 10
    plt.xticks(np.arange(0,n_samples/fs,step=grid_step)), plt.xlim(0, n_samples/fs + 1)
    plt.tight_layout(), plt.show()
    
    # save and close figure
    fig_markers.subplots_adjust(hspace=0.03)  # adjust space between axes
    fig_markers.savefig(fname + '_event_markers.png')
    # Close figure
    plt.close()


# %% Hand position functions

def generate_hand_pos_channels(hand_pos_array, hand_pos_str, n_hand_pos_chan, n_samples):
    
    n_hand_pos_samp = len(hand_pos_array)//n_hand_pos_chan
    hand_pos_mat = np.zeros((n_hand_pos_chan,n_hand_pos_samp))
    
    for i in range(n_hand_pos_chan):
        pos = hand_pos_list[i]
        hand_pos_mat[i,:] = np.array([float(i.replace(str(pos),'')) for i in hand_pos_str if i.startswith(str(pos)) or i.startswith(str(-pos))])
    
    # resample hand_pos data
    hand_pos_data = scipy.signal.resample(hand_pos_mat, n_samples, axis=1)
    
    return hand_pos_data


def plot_hand_positions(n_hand_pos_chan,n_samples,fs,hand_pos_data,fname,run_idx):
    
    hand_pos_labels = ['HPRx','HPRy','HPRz','HPLx','HPLy','HPLz']
    
    # plot hand position coordinates
    fig_hand_pos = plt.figure(figsize=(20,10))
    for i in range(n_hand_pos_chan):
        label = hand_pos_labels[i]
        plt.plot(np.arange(0,n_samples)/fs, hand_pos_data[i,:], label=label)
    grid_step = 20 if n_samples/fs > 400 else 10
    plt.xticks(np.arange(0,n_samples/fs+10,step=grid_step))
    plt.yticks(np.arange(np.min(np.min(hand_pos_data))-0.5,np.max(np.max(hand_pos_data))+0.5, step = 0.5))
    plt.xlim(0,n_samples/fs+1)
    plt.ylim(np.min(np.min(hand_pos_data))-0.5,np.max(np.max(hand_pos_data))+0.5)
    plt.xlabel('time (s)'), plt.ylabel('coordinates')
    plt.title('Hand Positions - ' + sub_str + '_run_' + str(run_idx+1))
    plt.tight_layout(), plt.grid(), plt.legend(loc='lower left'), plt.show()
    # Save figure
    fig_hand_pos.savefig(fname + '_hand_positions.png')
    # Close figure
    plt.close()


#%% Stream selection

def stream_select_eeg(streams, streams_all, stream_name, stream_idx, run_idx):
    
    display_message('   Generating EEG channels...')
    n_samples = streams_all['n_samples'][run_idx]
    
    try:
        eeg_data = streams[stream_idx]['time_series'].T
        eeg_times = streams[stream_idx]['time_stamps']
        # plot eeg stream consistency
        fname = streams_all['filename'][run_idx]
        plot_eeg_stream(eeg_times,fs,fname,run_idx)
        # print message
        print('   EEG channels successfully generated.')
        
    except Exception as error:
        # print error message
        print('   Error: could not generate EEG channels!')
        print('  ', type(error).__name__, "–", error)
        # fill eeg channels with zeros
        eeg_data = np.zeros((n_chans_eeg,n_samples))
        print('   Filling channels with zeros.')
        
    # update streams
    streams_all[stream_name]['time_series'].append(eeg_data)
    
    return streams_all


def stream_select_markers(streams, streams_all, stream_name, stream_idx, run_idx):
    
    display_message('   Generating event markers channel...')
    # Generate markers channel
    eeg_times = streams_all[stream_neurone]["time_stamps"][run_idx]
    # Get samples
    n_samples = streams_all['n_samples'][run_idx]
    
    try:
        markers_array = streams[stream_idx]["time_series"][:,0]        
        markers_times = streams[stream_idx]["time_stamps"]
        markers_data = generate_markers_channel(markers_array, markers_times, eeg_times)
        # plot event markers
        fname = streams_all['filename'][run_idx]
        plot_event_markers(markers_data, markers_array, eeg_times, fs, fname, run_idx)
        print('   Event markers channel successfully generated.')
        
    except Exception as error:
        # print error message
        print('   Error: Could not generate event markers channel!')
        print('  ', type(error).__name__, "–", error)
        # fill markers channel with zeros
        markers_data = np.zeros((1,n_samples))
        print('   Filling channels with zeros.')
        
    # update streams
    streams_all[stream_name]['time_series'].append(markers_data)
    
    return streams_all


def stream_select_hand_pos(streams, streams_all, stream_name, stream_idx, run_idx):
    
    display_message('   Generating hand position channels...')
    # Create hand position stream
    n_samples = streams_all['n_samples'][run_idx]
    
    try:
        hand_pos_array = streams[stream_idx]["time_series"][:,0]
        hand_pos_str = [str(i) for i in hand_pos_array]
        hand_pos_data = generate_hand_pos_channels(hand_pos_array, hand_pos_str, n_chans_hand_pos, n_samples)
        # plot hand pos coordinates
        fname = streams_all['filename'][run_idx]
        plot_hand_positions(n_chans_hand_pos,n_samples,fs,hand_pos_data,fname,run_idx)
        print('   Hand position channels successfully created.')
    
    except Exception as error:
        print('   Error: Could not create hand position channels!')
        print('  ', type(error).__name__, "–", error)
        # fill hand pos channels with zeros
        hand_pos_data = np.zeros((n_chans_hand_pos,n_samples))
        print('   Filling channels with zeros.')
    
    # update streams    
    streams_all[stream_name]['time_series'].append(hand_pos_data)
    
    return streams_all


#%% Creating raw object

def create_raw_object(streams_all,stream_names,fs,chan_names,run_idx,n_runs,raw_all=False):
    
    display_string = 'Saving all data as one MNE raw object...' if raw_all else 'Saving data as MNE raw object...'
    display_message(display_string)
    
    try:
        # loop over runs, do not loop if run_idx < n_runs
        n_loops = n_runs if raw_all else 1
        for loop_idx in range(n_loops):
            # concatenate data channels and channel types
            data_channels, chan_types = [], []
            if raw_all: run_idx = loop_idx
            
            for stream_name in stream_names:
                if stream_name == stream_neurone:
                    data_channels = streams_all[stream_name]['time_series'][run_idx]*1e-6
                    for ct in range(n_chans_eeg): chan_types.append('eeg') 
                elif stream_name == stream_markers:
                    data_channels = np.concatenate((data_channels, streams_all[stream_name]['time_series'][run_idx]),axis=0)
                    chan_types.append('stim')
                elif stream_name == stream_hand_pos:
                    data_channels = np.concatenate((data_channels, streams_all[stream_name]['time_series'][run_idx]),axis=0)
                    for ct in range(n_chans_hand_pos): chan_types.append('misc')
            
            # Stack data from multiple runs        
            if raw_all:
                mne_filename = fpath + os.sep + sub_str + '_' + param_dir + '_all.fif'
                if loop_idx == 0: 
                    data_channels_all = data_channels 
                else:
                    data_channels_all = np.append(data_channels_all,data_channels,axis=1)
            else:
                data_channels_all = data_channels
                mne_filename = streams_all['filename'][run_idx] + '.fif'
                
        # Sampling rate
        sfreq = fs
        # Create info 
        info = mne.create_info(chan_names, sfreq, chan_types)
        # Create raw object
        raw = mne.io.RawArray(data_channels_all, info)
        # Set montage layout
        eeg_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(eeg_montage)
        # Save fif file
        raw.save(mne_filename, overwrite=True)
    
    except Exception as error:
        display_message('   Error: Could not save data as MNE raw object!')
        print('  ', type(error).__name__, "–", error)


#%% Other functions

def display_message(message_string):
    print('----------------------------------------------------------------------------------')
    print(message_string)
    

def read_xdf_files(fpath):
    # Get all xdf files
    fnames = glob.glob(fpath + os.sep + '*.xdf')
    # Get number of runs
    n_runs = len(fnames)
    # Print xdf file names
    display_message(str(n_runs) + ' run file(s) found:')
    for fname in fnames:
        print('   ' + fname)
        
    return fnames, n_runs


def load_lsl_streams(fname,streams_all):
    display_message('Loading run file ' + fname + '...')
    # Load LSL streams
    streams, header = pyxdf.load_xdf(fname)
    n_streams = len(streams)
    display_message(str(n_streams) + ' LSL stream(s) found:')
    
    # Loop through LSL streams and fill dict
    for stream_idx in range(n_streams): 
        stream_name = streams[stream_idx]['info']['name'][0]
        streams_all[stream_name]['info'].append(streams[stream_idx]['info'])
        streams_all[stream_name]['footer'].append(streams[stream_idx]['footer'])
        streams_all[stream_name]['time_stamps'].append(streams[stream_idx]['time_stamps'])
        if stream_name == stream_neurone:
            streams_all['n_samples'].append(len(streams[stream_idx]['time_stamps']))
        print('   Stream ' + str(stream_idx + 1) + ': ' + stream_name)
        
    return streams, streams_all


#%% Main function

def main():
    # Read xdf files
    fnames, n_runs = read_xdf_files(fpath)
    # Dict for all data from all streams
    streams_all = {'filename': [], 'n_samples': []}
    for stream_name in stream_names: streams_all[stream_name] = {'info': [], 'footer': [], 'time_series': [], 'time_stamps': []}

    # Loop through xdf files
    for run_idx in range(n_runs):
        fname = fnames[run_idx]
        streams_all['filename'].append(fname.replace('.xdf', ''))
        streams, streams_all = load_lsl_streams(fname,streams_all)
        
        # Loop through LSL streams
        for stream_idx in range(len(streams)):
            stream_name = streams[stream_idx]['info']['name'][0]
            display_message('Stream ' + str(stream_idx + 1) + ': ' + stream_name)
            
            # EEG stream
            if stream_name == stream_neurone:
                streams_all = stream_select_eeg(streams, streams_all, stream_name, stream_idx, run_idx)
            # Marker stream
            elif stream_name == stream_markers:
                streams_all = stream_select_markers(streams, streams_all, stream_name, stream_idx, run_idx)
            # Hand position stream
            elif stream_name == stream_hand_pos:
                streams_all = stream_select_hand_pos(streams, streams_all, stream_name, stream_idx, run_idx)
            # Stream name does not match 
            else:
                display_message('Error: Stream ' + str(stream_name) + ' does not match defined LSL streams!')
        
        # Create MNE raw object
        create_raw_object(streams_all,stream_names,fs,chan_names,run_idx,n_runs,raw_all=False)
    
    # Create MNE raw object for all runs
    if n_runs > 1:
       create_raw_object(streams_all,stream_names,fs,chan_names,run_idx,n_runs,raw_all=True)
       
    # Finished
    display_message('XDF to MNE conversion finished!')
    

#%% Run main

if __name__ == "__main__":
    main()
    

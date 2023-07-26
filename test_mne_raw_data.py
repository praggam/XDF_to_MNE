# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 16:48:28 2023

@author: Philipp Raggam
"""

#%% import packages

import mne
import os


#%% load raw object file

# Subject directory
sub_dir = 'Pilot_2'
# Paradigm directory
param_dir = 'run_files'
# Subject stromg
sub_str = 'pilot_1'
# Recording directory
fpath = 'D:' + os.sep + 'VR-EEG' + os.sep + sub_dir + os.sep + param_dir

filename = fpath + os.sep + sub_str + '_run_2.fif'

raw = mne.io.read_raw_fif(filename, preload=True)

# print raw info
print(raw.info)


#%% plot events

try:
    # Sampling frequency
    fs = raw.info['sfreq']
    
    # Get events from stim channel
    events = mne.find_events(raw, stim_channel='STIM', verbose=True)
    
    # Set event IDs
    event_id = {'task_1_start':  100,'task_1_end':  103,
                'task_2_start':  200,'task_2_end':  203,
                'task_3_start':  300,'task_3_end':  303,
                'task_1_trial_start':  101,'task_1_trial_end':  102,
                'task_2_trial_start':  201,'task_2_trial_end':  202,
                'task_3_trial_start':  301,'task_3_trial_end':  302}
    
    # Plot events
    fig_events = mne.viz.plot_events(events, sfreq=fs, event_id=event_id)
    fig_events.savefig(fpath + os.sep + sub_str + 'events.png')
except Exception as error:
    print('', type(error).__name__, "–", error)
    

#%% preprocess and plot channel signals

# Correct powerline interference: 50 Hz notch filter (and harmonic)
power_line = (50,100)
raw.load_data().notch_filter(freqs=power_line)

# Bandpass signal between 1 and 100 Hz
l_freq, h_freq = 1, 100
raw_filt = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)

# Rereference Data (CAR)
raw_filt.set_eeg_reference(ref_channels='average', ch_type='eeg')

# plot the channels
raw_filt.plot(events=events, start = 5, duration = 10, color='gray', scalings='auto')

#%% plot eeg psd

chans_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
mne.viz.plot_raw_psd(raw_filt, picks=chans_eeg, fmax = h_freq)

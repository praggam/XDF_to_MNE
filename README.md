# XDF_to_MNE

## xdf_to_mne_vr-egg_main.py
Converting xdf-file(s) recorded with LabRecorder to MNE raw object

### Flow:
1. Scanning directory with xdf-file(s)
2. Loading xdf-file(s)
3. Loading LSL streams of individual xdf-file(s)
4. Create channels from individual LSL streams
5. Generate info and create MNE raw object of individual xdf-file(s)
6. Generate info and create MNE raw object of all xdf-files combined

### Define parameters:
* Subject parameters: Define path and filenames of the recording files
* Stream names: Define the names of your LSL streams
	* stream_neurone:  stream from eeg recroding unit, do not change
	* stream_markers:  stream for event markers, change the name according to your markers stream name
	* stream_hand_pos: used for tracking hand coordinates, change string to '' if not used
* Channel names: Create a list with channel names for data from all streams
	* EEG channels:   Copy names from montage list
	* Event markers:  Channel is named 'STIM'
	* Hand positions: 6 channels for x-,y-,z-coordinates for right and left hand
* Recording parameters: Define sampling rate

### Add new LSL stream:
1. Define stream name: stream_new_stream = 'new_stream'
2. Add stream name to the list: stream_names = [stream_neurone, stream_markers, stream_hand_pos, stream_new_stream]
3. Define stream selecting function (structure can be copied from other stream selecting functions):  
	def stream_select_new_stream(streams, streams_all, stream_name, stream_idx, run_idx):  
		new_stream_data = streams[stream_idx]['time_series'][:,0]  
		'''  
		Your additional code here  
		'''  
		streams_all[stream_name]['time_series'].append(hand_pos_data)  
		return streams_all  
4. Add function to main() in LSL streams loop:
    if stream_name == stream_neurone: # EEG stream
        streams_all = stream_select_eeg(streams, streams_all, stream_name, stream_idx, run_idx)
    elif stream_name == stream_markers: # Marker stream
        streams_all = stream_select_markers(streams, streams_all, stream_name, stream_idx, run_idx)
    elif stream_name == stream_hand_pos: # Hand position stream
        streams_all = stream_select_hand_pos(streams, streams_all, stream_name, stream_idx, run_idx)
    elif stream_name == stream_new_stream: # Your ew stream
        streams_all = stream_select_new_stream(streams, streams_all, stream_name, stream_idx, run_idx)

## test_mne_raw_data.py
Loading, preprocessing and plotting data from the created raw data object

### Flow:
1. Load raw object file (.fif)
2. Plot event markers
3. Preprocess data (notch filter, bandpass filter, set CAR reference)
4. Plot channels (time series)
5. Plot PSD

### Define parameters:
* Subject parameters: Define path and filenames of the recording files



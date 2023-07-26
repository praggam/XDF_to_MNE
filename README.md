# XDF_to_MNE

## xdf_to_mne_vr-egg_main.py
Converting xdf-file(s) recorded with LabRecorder to MNE raw object

### Work Flow
1. Scanning directory with xdf-file(s)
2. Loading xdf-file(s)
3. Loading LSL streams of individual xdf-file(s)
4. Create channels from individual LSL streams
5. Generate info and create MNE raw object of individual xdf-file(s)
6. Generate info and create MNE raw object of all xdf-files combined

### Define parameters
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

## test_mne_raw_data.py

### Work Flow
1. Load raw object file (.fif)
2. Plot event markers
3. Preprocess data (notch filter, bandpass filter, set CAR reference)
4. Plot channels (time series)
5. Plot PSD

### Define parameters
* Subject parameters: Define path and filenames of the recording files



# XDF_to_MNE

## Flow:
1. Scanning directory with recording file(s)
2. Loading xdf-file(s)
3. Loading LSL streams of individual xdf-file(s)
4. Create channels from individual LSL streams
5. Generate info and create MNE raw object of individual xdf-file(s)
6. Generate info and create MNE raw object of all xdf-files combined


## Define parameters:
* Subject parameters:
	* sub_str: directory of xdf-file(s) of current subject
	* fpath:   directory of all study recordings

* LSL stream names:
	* stream_neurone:  stream from eeg recroding unit, do not change
	* stream_markers:  stream for event markers, name has to be changed (unless the VR headset is used)
	* stream_hand_pos: optional, if the VR headset is used (define as empty '[]' if not used)

* Channel names
	* chan_names: define channel names, position of chan names should match the neurone recording montage
		* 'STIM': event marker channel
		* 'HP..': hand position channels

* Hand position markers:
	* do not change 



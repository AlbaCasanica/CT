import labeling

''' Labeling '''
# copy the file to label and past it inside the " "
filepath_sensorfile = ""

# Create dataframe and find peaks, always run this first
df, peaks = labeling.detect_spikes(filepath_sensorfile)

# Extract the file with start and stop time for labels:

# plot file with labels:
labeling.plot_signal_peaks(df, peaks)

# plot closeup around peaks (whare start and stop are to be changed)
labeling.plot_closeups_around_peaks(df, peaks, 3)

## Save the .csv file with correct start and stop time to the shared folder


''' Plot with new labels (not nessessary)
    save the start_stop_time_activities file after updated time for walking and fill in the filepath for it to run the plot
    comment out the part under if you dont want to plot the result from labeling '''

filepath_correct_start_stop_labels = ""

# This also saves a new file that can just be deleted afterwards
df_correct_labels, labeled_filepath = labeling.apply_corrected_labels(filepath_sensorfile, filepath_correct_start_stop_labels)
labeling.plot_signal_peaks(df_correct_labels, peaks)
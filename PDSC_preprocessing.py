import mne
import os
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_bids import BIDSPath, read_raw_bids

bids_root = './'
preprocessed_path = os.path.join(bids_root, "derivatives", "preprocessed")
os.makedirs(preprocessed_path, exist_ok=True)

participants_fpath = os.path.join(bids_root, 'participants.tsv')
participants_df = pd.read_csv(participants_fpath, sep='\t')
subjects = participants_df['participant_id'].tolist()

bad_channels_map = {
    'sub-002': ['TP10']
}

for subject in subjects:
    subject_id = subject.split('-')[-1]
    bids_path = BIDSPath(subject=subject_id, 
                         task='Simon', 
                         suffix='eeg', 
                         extension='.set',
                         root=bids_root)
    
    try:
        raw = read_raw_bids(bids_path, verbose=False)
    except FileNotFoundError:
        continue

    raw.load_data(verbose=False)
    
    ch_to_drop = ['I1', 'I2', 'Resp']
    ch_to_drop = [ch for ch in ch_to_drop if ch in raw.ch_names]
    if ch_to_drop:
        raw.drop_channels(ch_to_drop)

    misc_chans = [ch for ch in raw.ch_names if raw.get_channel_types(ch)[0] == 'misc']
    if misc_chans:
        channel_types = {ch: 'eeg' for ch in misc_chans}
        raw.set_channel_types(channel_types)
         
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
        
    if subject in bad_channels_map:
            bads = bad_channels_map[subject]

            bads = [b for b in bads if b in raw.ch_names] 
            
            if bads:
                raw.info['bads'] = bads
                print(f"  Marked {bads} as bad.")
                
                raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)
                print(f"  Interpolated {bads}.")

    
    raw.filter(l_freq=1.0, h_freq=100, fir_design='firwin', verbose=False)
    
    raw.set_eeg_reference('average', verbose=False)

    ica = ICA(n_components=0.99, max_iter='auto', method='infomax', 
              fit_params=dict(extended=True), random_state=715)
    ica.fit(raw)

    ic_labels = label_components(raw, ica, method='iclabel')
    labels = ic_labels["labels"]
    ica.exclude = [idx for idx, label in enumerate(labels) 
                   if label not in ["brain", "other"]]
    
    ica.apply(raw)

    raw.filter(l_freq=1.0, h_freq=45.0, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=60, verbose=False)

    save_dir = os.path.join(preprocessed_path, subject)
    os.makedirs(save_dir, exist_ok=True)
    save_fname = f"{subject}_task-Simon_preprocessed.fif"
    save_fpath = os.path.join(save_dir, save_fname)
    
    raw.save(save_fpath, overwrite=True)



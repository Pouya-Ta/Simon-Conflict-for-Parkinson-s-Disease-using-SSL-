import os
import warnings
import pandas as pd
import mne

from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne_bids import BIDSPath, read_raw_bids

# Paths
bids_root = "./"
deriv_root = os.path.join(bids_root, "derivatives", "preprocessed")
os.makedirs(deriv_root, exist_ok=True)

participants_fpath = os.path.join(bids_root, "participants.tsv")
participants_df = pd.read_csv(participants_fpath, sep="\t")
subjects = participants_df["participant_id"].tolist()

# Subject-specific bad channels (we can add more subjects if needed)
bad_channels_map = {
    "sub-002": ["TP10"],
}

# Configuration
RANDOM_STATE = 715
TASK_NAME = "Simon"

# Final analysis filter:
# For ERP-focused analyses, use something like 0.1-40 Hz
# For theta/time-frequency analyses, 1-40 or 1-45 can be OK
FINAL_L_FREQ = 0.1
FINAL_H_FREQ = 40.0

# ICA-fit filter:
# 1 Hz high-pass is standard and often improves ICA stability
ICA_L_FREQ = 1.0
ICA_H_FREQ = 45.0

LINE_FREQ = 60.0

# Known non-EEG channels that may appear
NON_EEG_TO_DROP = ["I1", "I2", "Resp"]

# Optional: skip subjects if output already exists
SKIP_IF_EXISTS = False

# =========================
# Helper functions
# =========================
def safe_set_montage(raw):
    """Attach standard montage safely."""
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")


def mark_and_interpolate_bads(raw, subject, bad_channels_map):
    """Mark and interpolate known bad channels."""
    if subject not in bad_channels_map:
        return raw

    bads = [ch for ch in bad_channels_map[subject] if ch in raw.ch_names]
    if len(bads) == 0:
        return raw

    raw.info["bads"] = bads
    print(f"[{subject}] Marked bad channels: {bads}")

    # Interpolate only if there are bad channels
    raw.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
    print(f"[{subject}] Interpolated bad channels.")
    return raw


def relabel_possible_eeg_channels(raw):
    """
    Relabel only plausible scalp channels from misc -> eeg.
    Do NOT convert all misc channels blindly.
    """
    standard_montage = mne.channels.make_standard_montage("standard_1020")
    known_eeg_names = set(standard_montage.ch_names)

    mapping = {}
    for ch in raw.ch_names:
        ch_type = raw.get_channel_types(picks=[ch])[0]
        if ch_type == "misc" and ch in known_eeg_names:
            mapping[ch] = "eeg"

    if mapping:
        raw.set_channel_types(mapping)
        print(f"Relabeled misc->eeg for channels: {list(mapping.keys())}")

    return raw


def prepare_raw(subject, bids_root):
    """Load one subject from BIDS."""
    subject_id = subject.split("-")[-1]

    bids_path = BIDSPath(
        subject=subject_id,
        task=TASK_NAME,
        suffix="eeg",
        extension=".set",
        root=bids_root
    )

    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data(verbose=False)
    return raw


def remove_non_eeg_channels(raw):
    """Drop known non-EEG channels if present."""
    to_drop = [ch for ch in NON_EEG_TO_DROP if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
        print(f"Dropped non-EEG channels: {to_drop}")
    return raw


def fit_ica_on_copy(raw, subject):
    """
    Fit ICA on a dedicated filtered copy.
    Returns fitted ICA object and IC labels.
    """
    raw_ica = raw.copy()

    # Filter specifically for ICA
    raw_ica.filter(
        l_freq=ICA_L_FREQ,
        h_freq=ICA_H_FREQ,
        fir_design="firwin",
        verbose=False
    )

    # Notch line noise BEFORE ICA if needed
    raw_ica.notch_filter(freqs=[LINE_FREQ], verbose=False)

    # Average reference for ICA
    raw_ica.set_eeg_reference("average", verbose=False)

    picks_eeg = mne.pick_types(raw_ica.info, eeg=True, exclude="bads")

    ica = ICA(
        n_components=0.99,
        method="infomax",
        fit_params=dict(extended=True),
        max_iter="auto",
        random_state=RANDOM_STATE
    )

    print(f"[{subject}] Fitting ICA ...")
    ica.fit(raw_ica, picks=picks_eeg, verbose=False)

    print(f"[{subject}] Labeling ICA components with ICLabel ...")
    ic_labels = label_components(raw_ica, ica, method="iclabel")

    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    # Exclude obvious artifact classes with confidence threshold
    # You can tune this threshold later if needed
    exclude_idx = []
    artifact_labels = {"eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"}

    for idx, (label, prob_vec) in enumerate(zip(labels, probs)):
        # Get probability of predicted class
        pred_prob = prob_vec.max()
        if label in artifact_labels and pred_prob >= 0.70:
            exclude_idx.append(idx)

    ica.exclude = exclude_idx
    print(f"[{subject}] Excluding ICA components: {ica.exclude}")

    return ica, ic_labels


def preprocess_final_raw(raw, ica, subject):
    """
    Create final cleaned data for analysis.
    """
    raw_final = raw.copy()

    # Final analysis filter
    raw_final.filter(
        l_freq=FINAL_L_FREQ,
        h_freq=FINAL_H_FREQ,
        fir_design="firwin",
        verbose=False
    )

    # If your final high cutoff were > 60, then notch here would matter.
    # Since FINAL_H_FREQ = 40, no notch is needed.

    raw_final.set_eeg_reference("average", verbose=False)

    print(f"[{subject}] Applying ICA ...")
    ica.apply(raw_final)

    return raw_final


# =========================
# Main loop
# =========================
for subject in subjects:
    print(f"\nProcessing {subject} ...")

    save_dir = os.path.join(deriv_root, subject)
    os.makedirs(save_dir, exist_ok=True)
    save_fpath = os.path.join(save_dir, f"{subject}_task-{TASK_NAME}_preprocessed_raw.fif")

    if SKIP_IF_EXISTS and os.path.exists(save_fpath):
        print(f"[{subject}] Output exists. Skipping.")
        continue

    try:
        raw = prepare_raw(subject, bids_root)
    except FileNotFoundError:
        print(f"[{subject}] EEG file not found. Skipping.")
        continue
    except Exception as e:
        print(f"[{subject}] Failed to load: {e}")
        continue

    try:
        # 1) Remove known non-EEG channels
        raw = remove_non_eeg_channels(raw)

        # 2) Relabel only plausible scalp channels from misc->eeg
        raw = relabel_possible_eeg_channels(raw)

        # 3) Set montage
        safe_set_montage(raw)

        # 4) Mark/interpolate known bad channels
        raw = mark_and_interpolate_bads(raw, subject, bad_channels_map)

        # 5) Fit ICA on dedicated copy
        ica, ic_labels = fit_ica_on_copy(raw, subject)

        # 6) Preprocess final analysis data and apply ICA
        raw_final = preprocess_final_raw(raw, ica, subject)

        # 7) Save cleaned raw
        raw_final.save(save_fpath, overwrite=True)
        print(f"[{subject}] Saved cleaned file to: {save_fpath}")

        # 8) Save ICA object too (helpful for inspection later)
        ica_fpath = os.path.join(save_dir, f"{subject}_task-{TASK_NAME}_ica.fif")
        ica.save(ica_fpath, overwrite=True)
        print(f"[{subject}] Saved ICA to: {ica_fpath}")

    except Exception as e:
        print(f"[{subject}] Preprocessing failed: {e}")
        continue

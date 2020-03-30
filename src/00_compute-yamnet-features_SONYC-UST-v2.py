import sys

import keras
import numpy as np
import os
import pandas as pd
import resampy
import soundfile as sf
import tensorflow as tf
import warnings

import params
import yamnet as yamnet_model

# List WAV names
data_dir = localmodule.get_data_dir()
audio_dir = os.path.join(data_dir, "audio")
annotations_path = os.path.join(data_dir, "annotations.csv")
df = pd.read_csv(annotations_path)
wav_names = list(np.unique(df["audio_filename"]))

# Load YAMNet
tf.get_logger().setLevel('ERROR')
params.PATCH_HOP_SECONDS = 0.96
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet.h5')
yamnet_embedding = tf.keras.Model(
    inputs=yamnet.inputs, outputs=yamnet.layers[-4].output
)

# Initialize HDF5 folder
h5_dir = os.path.join(data_dir, "yamnet-features_h5")

# Compute features
for split_str in ["train", "validate"]:
    h5_name = "yamnet-features_SONYC-UST-v2_" + split_str + ".h5"
    h5_path = os.path.join(h5_dir, h5_name)
    split_wav_names = np.unique(df[df["split"] == split_str]["audio_filename"])
    with h5py.File(h5_path, "w") as h5_file:
        for wav_name in split_wav_names:
            wav_path = os.path.join(audio_dir, wav_name)
            wav_data, sr = sf.read(wav_path, dtype=np.int16)
            waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
            waveform = resampy.resample(waveform, sr, 16000)[np.newaxis, :]
            h5_key = wav_name.split(".")[0]
            h5_value = model.predict(waveform, steps=1)
            h5_file[h5_key] = Y[wav_name]

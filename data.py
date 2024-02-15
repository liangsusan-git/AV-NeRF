import os
import math
import json
import random
import pickle
import einops
import librosa
import numpy as np
from tqdm import tqdm
from PIL import Image
import soundfile as sf

import torch
import torchaudio
import torchvision.transforms as T

class RWAVSDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 sr=22050,
                 no_pos=False,
                 no_ori=False):
        super(RWAVSDataset, self).__init__()
        self.split = split
        self.sr = sr

        clip_len = 0.5 # second
        wav_len = int(2 * clip_len * sr)

        # sound source
        position = json.loads(open(os.path.join(os.path.dirname(data_root[:-1]), "position.json"), "r").read())
        position = np.array(position[data_root.split('/')[-2]]["source_position"][:2]) # (x, y)
        print(f"Split: {split}, sound source: {position}, wav_len: {wav_len}")

        # rgb and depth features
        feats = pickle.load(open(os.path.join(data_root, f"feats_{split}.pkl"), "rb"))

        # audio
        if os.path.exists(os.path.join(data_root, "binaural_syn_re.wav")):
            audio_bi, _ = librosa.load(os.path.join(data_root, "binaural_syn_re.wav"), sr=sr, mono=False)
        else:
            print("Unavilable, re-process binaural...")
            audio_bi_path = os.path.join(data_root, "binaural_syn.wav")
            audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False) # [2, ?]
            audio_bi = audio_bi / np.abs(audio_bi).max()
            sf.write(os.path.join(data_root, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
        
        if os.path.exists(os.path.join(data_root, "source_syn_re.wav")):
            audio_sc, _ = librosa.load(os.path.join(data_root, "source_syn_re.wav"), sr=sr, mono=True)
        else:
            print("Unavilable, re-process source...")
            audio_sc_path = os.path.join(data_root, "source_syn.wav")
            audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True) # [?]
            audio_sc = audio_sc / np.abs(audio_sc).max()
            sf.write(os.path.join(data_root, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

        # pose
        transforms_path = os.path.join(data_root, f"transforms_scale_{split}.json")
        transforms = json.loads(open(transforms_path, "r").read())

        # data
        data_list = []
        for item_idx, item in enumerate(transforms["camera_path"]):
            pose = np.array(item["camera_to_world"]).reshape(4, 4)
            xy = pose[:2,3]
            ori = pose[:2,2]
            data = {"pos": xy}
            ori = relative_angle(position, xy, ori)
            data["ori"] = ori

            if no_pos:
                data["pos"] = np.zeros(2)
            
            if no_ori:
                data["ori"] = 0

            data["rgb"] = feats["rgb"][item_idx]
            data["depth"] = feats["depth"][item_idx]

            # extract key frames at 1 fps
            time = int(item["file_path"].split('/')[-1].split('.')[0])
            data["img_idx"] = time
            st_idx = max(0, int(sr * (time - clip_len)))
            ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
            if ed_idx - st_idx < int(clip_len * sr): continue
            audio_bi_clip = audio_bi[:, st_idx:ed_idx]
            audio_sc_clip = audio_sc[st_idx:ed_idx]

            # padding with zero
            if(ed_idx - st_idx < wav_len):
                pad_len = wav_len - (ed_idx - st_idx)
                audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
                audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
                print(f"padding from {ed_idx - st_idx} -> {wav_len}")
            elif(ed_idx - st_idx > wav_len):
                audio_bi_clip = audio_bi_clip[:, :wav_len]
                audio_sc_clip = audio_sc_clip[:wav_len]
                print(f"cutting from {ed_idx - st_idx} -> {wav_len}")

            # binaural
            spec_bi = stft(audio_bi_clip)
            mag_bi = np.abs(spec_bi) # [2, T, F]
            phase_bi = np.angle(spec_bi) # [2, T, F]
            data["mag_bi"] = mag_bi

            # source
            spec_sc = stft(audio_sc_clip)
            mag_sc = np.abs(spec_sc) # [T, F]
            phase_sc = np.angle(spec_sc) # [T, F]
            data["mag_sc"] = mag_sc

            data["wav_bi"] = audio_bi_clip
            data["phase_bi"] = phase_bi
            data["wav_sc"] = audio_sc_clip
            data["phase_sc"] = phase_sc

            data_list.append(data)
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

def vector_angle(xy):
    radians = math.atan2(xy[0], xy[1])
    return radians / (1.01 * np.pi) # trick to make sure ori in open set (-1, 1)

def relative_angle(source, xy, ori): # (-1, 1)
    s = source - xy
    s = s / np.linalg.norm(s)
    d = ori / np.linalg.norm(ori)
    theta = np.arccos(np.clip(np.dot(s, d), -1, 1)) / (1.01 * np.pi)
    rho = np.arcsin(np.clip(np.cross(s, d), -1, 1))
    if rho < 0:
        theta *= -1
    return theta

def stft(signal):
    spec = librosa.stft(signal, n_fft=512)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c f t -> c t f")
    else:
        raise NotImplementedError
    return spec
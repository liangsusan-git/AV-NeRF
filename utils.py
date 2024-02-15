import librosa
import argparse
import numpy as np
from math import pi
from numpy import linalg as LA
from scipy.signal import hilbert

import torch
import torch.nn.functional as F

def energy_decay(pred_mag, gt_mag):
    # [B, T, F]
    gt_mag = torch.sum(gt_mag ** 2, dim=2)
    pred_mag = torch.sum(pred_mag ** 2, dim=2)
    gt_mag = torch.log1p(gt_mag)
    pred_mag = torch.log1p(pred_mag)

    loss = F.l1_loss(gt_mag, pred_mag)
    return loss

'''
def energy_decay(pred_mag, gt_mag):
    pred_mag = einops.rearrange(pred_mag, "b t f -> b f t")
    gt_mag = einops.rearrange(gt_mag, "b t f -> b f t")
    # [B, F, T]

    gts_fullBandAmpEnv = torch.sum(gt_mag, dim=1)
    power_gts_fullBandAmpEnv = gts_fullBandAmpEnv ** 2
    energy_gts_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_gts_fullBandAmpEnv, [1]), 1), [1])
    valid_loss_idxs = ((energy_gts_fullBandAmpEnv != 0.).type(energy_gts_fullBandAmpEnv.dtype))[:, 1:]
    db_gts_fullBandAmpEnv = 10 * torch.log10(energy_gts_fullBandAmpEnv + 1.0e-13)
    norm_db_gts_fullBandAmpEnv = db_gts_fullBandAmpEnv - db_gts_fullBandAmpEnv[:, :1]
    norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv[:, 1:]
    weighted_norm_db_gts_fullBandAmpEnv = norm_db_gts_fullBandAmpEnv * valid_loss_idxs

    preds_fullBandAmpEnv = torch.sum(pred_mag, dim=1)
    power_preds_fullBandAmpEnv = preds_fullBandAmpEnv ** 2
    energy_preds_fullBandAmpEnv = torch.flip(torch.cumsum(torch.flip(power_preds_fullBandAmpEnv, [1]), 1), [1])
    db_preds_fullBandAmpEnv = 10 * torch.log10(energy_preds_fullBandAmpEnv + 1.0e-13)
    norm_db_preds_fullBandAmpEnv = db_preds_fullBandAmpEnv - db_preds_fullBandAmpEnv[:, :1]
    norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv[:, 1:]
    weighted_norm_db_preds_fullBandAmpEnv = norm_db_preds_fullBandAmpEnv * valid_loss_idxs

    loss = F.l1_loss(weighted_norm_db_preds_fullBandAmpEnv, weighted_norm_db_gts_fullBandAmpEnv)
    return loss
'''

def istft(mag, phase):
    mag = mag.cpu().numpy()
    phase = phase.cpu().numpy()
    spec = mag * np.exp(1j * phase)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c t f -> c f t")
    else:
        raise NotImplementedError
    wav = librosa.istft(spec, n_fft=512)
    if wav.ndim == 2:
        wav = wav.T
    return wav

class Evaluator(object):
    def __init__(self, norm=False):
        self.env_loss = []
        self.mag_loss = []
        self.snr_loss = []
        self.snr_norm_loss = []
        self.norm = norm
    
    def update(self, mag_prd, mag_gt, wav_prd, wav_gt):
        mag_loss = np.mean(np.power(mag_prd - mag_gt, 2)) * 2
        self.mag_loss.append(mag_loss)
        env_loss = self.Envelope_distance(wav_prd, wav_gt)
        self.env_loss.append(env_loss)
        snr_loss = self.SNR(wav_prd, wav_gt)
        self.snr_loss.append(snr_loss)

        wav_prd = normalize(wav_prd)
        wav_gt = normalize(wav_gt)
        snr_norm_loss = self.SNR(wav_prd, wav_gt)
        self.snr_norm_loss.append(snr_norm_loss)
        
        return [mag_loss, env_loss, snr_loss, snr_norm_loss]

    def report(self):
        item_len = len(self.mag_loss)
        return {
                "env": sum(self.env_loss) / item_len,
                "mag": sum(self.mag_loss) / item_len,
                "snr": sum(self.snr_loss) / item_len,
                "snr_norm": sum(self.snr_norm_loss) / item_len
                }
    
    def STFT_L2_distance(self, predicted_binaural, gt_binaural):
        #channel1
        predicted_spect_channel1 = librosa.stft(predicted_binaural[0,:], n_fft=512)
        gt_spect_channel1 = librosa.stft(gt_binaural[0,:], n_fft=512)
        real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
        imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
        predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
        real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
        imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
        gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
        channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

        #channel2
        predicted_spect_channel2 = librosa.stft(predicted_binaural[1,:], n_fft=512)
        gt_spect_channel2 = librosa.stft(gt_binaural[1,:], n_fft=512)
        real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
        imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
        predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
        real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
        imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
        gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
        channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

        #sum the distance between two channels
        stft_l2_distance = channel1_distance + channel2_distance
        return float(stft_l2_distance)

    def Envelope_distance(self, predicted_binaural, gt_binaural):
        #channel1
        pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
        gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
        channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))
    
        #channel2
        pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
        gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
        channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))
    
        #sum the distance between two channels
        envelope_distance = channel1_distance + channel2_distance
        return float(envelope_distance)

    def SNR(self, predicted_binaural, gt_binaural):
        mse_distance = np.mean(np.power((predicted_binaural - gt_binaural), 2))
        snr = 10. * np.log10((np.mean(gt_binaural**2) + 1e-4) / (mse_distance + 1e-4))

        return float(snr)
    
    def Magnitude_distance(self, predicted_binaural, gt_binaural):
        predicted_spect_channel1 = librosa.stft(predicted_binaural[0,:], n_fft=512)
        gt_spect_channel1 = librosa.stft(gt_binaural[0,:], n_fft=512)
        predicted_spect_channel2 = librosa.stft(predicted_binaural[1,:], n_fft=512)
        gt_spect_channel2 = librosa.stft(gt_binaural[1,:], n_fft=512)
        stft_mse1 = np.mean(np.power(np.abs(predicted_spect_channel1) - np.abs(gt_spect_channel1), 2))
        stft_mse2 = np.mean(np.power(np.abs(predicted_spect_channel2) - np.abs(gt_spect_channel2), 2))

        return float(stft_mse1 + stft_mse2)

    def Angle_Diff_distance(self, predicted_binaural, gt_binaural):
        gt_diff = gt_binaural[0] - gt_binaural[1]
        pred_diff = predicted_binaural[0] - predicted_binaural[1]
        gt_diff_spec = librosa.stft(gt_diff, n_fft=512)
        pred_diff_spec = librosa.stft(pred_diff, n_fft=512)
        _, pred_diff_phase = librosa.magphase(pred_diff_spec)
        _, gt_diff_phase = librosa.magphase(gt_diff_spec)
        pred_diff_angle = np.angle(pred_diff_phase)
        gt_diff_angle = np.angle(gt_diff_phase)
        angle_diff_init_distance = np.abs(pred_diff_angle - gt_diff_angle)
        angle_diff_distance = np.mean(np.minimum(angle_diff_init_distance, np.clip(2 * pi - angle_diff_init_distance, a_min=0, a_max=2*pi))) 

        return float(angle_diff_distance)

def normalize(samples):
    return samples / np.maximum(1e-20, np.max(np.abs(samples)))

def myhibert(x, axis=1):
    # Make input a real tensor
    x = torch.as_tensor(x, device=x.device if isinstance(x, torch.Tensor) else None)
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    x = x.to(dtype=torch.float)

    if (axis < 0) or (axis > len(x.shape) - 1):
        raise ValueError(f"Invalid axis for shape of x, got axis {axis} and shape {x.shape}.")

    n = x.shape[axis]
    if n <= 0:
        raise ValueError("N must be positive.")
    x = torch.as_tensor(x, dtype=torch.complex64)
    # Create frequency axis
    f = torch.cat(
        [
            torch.true_divide(torch.arange(0, (n - 1) // 2 + 1, device=x.device), float(n)),
            torch.true_divide(torch.arange(-(n // 2), 0, device=x.device), float(n)),
        ]
    )
    xf = torch.fft.fft(x, n=n, dim=axis)
    # Create step function
    u = torch.heaviside(f, torch.tensor([0.5], device=f.device))
    u = torch.as_tensor(u, dtype=x.dtype, device=u.device)
    new_dims_before = axis
    new_dims_after = len(xf.shape) - axis - 1
    for _ in range(new_dims_before):
        u.unsqueeze_(0)
    for _ in range(new_dims_after):
        u.unsqueeze_(-1)

    ht = torch.fft.ifft(xf * 2 * u, dim=axis)

    # Apply transform
    return torch.as_tensor(ht, device=ht.device, dtype=ht.dtype)

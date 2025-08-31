from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

def calculate_psnr(sr, hr):
    sr = sr.clamp(0, 1).cpu().numpy()
    hr = hr.clamp(0, 1).cpu().numpy()
    psnr = 0.0
    for i in range(sr.shape[0]):
        psnr += peak_signal_noise_ratio(
            hr[i].transpose(1, 2, 0), 
            sr[i].transpose(1, 2, 0), 
            data_range=1.0
        )
    return psnr / sr.shape[0]

def calculate_ssim(sr, hr):
    sr = sr.clamp(0, 1).cpu().numpy()
    hr = hr.clamp(0, 1).cpu().numpy()
    ssim = 0.0
    for i in range(sr.shape[0]):
        ssim += structural_similarity(
            hr[i].transpose(1, 2, 0), 
            sr[i].transpose(1, 2, 0), 
            channel_axis=-1,      # modern replacement for multichannel=True
            data_range=1.0,
            win_size=7            # force valid default window size
        )
    return ssim / sr.shape[0]

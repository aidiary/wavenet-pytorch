import numpy as np
import scipy.signal
import librosa
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class WaveNetDataset(torch.utils.data.Dataset):
    pass


class STFT(torch.nn.Module):

    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(win_length >= filter_length)

            fft_window = scipy.signal.get_window(
                window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def forward(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        input_data = input_data.to(device)
        self.forward_basis = \
            self.forward_basis.requires_grad_(False).to(device)

        forward_transform = F.conv1d(
            input_data.to(device),
            self.forward_basis,
            stride=self.hop_length,
            padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part.data, real_part.data)

        return magnitude, phase


class TacotronSTFT(nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800,
                 n_mel_channels=80, sampling_rate=16000, mel_fmin=0.0, mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        mel_basis = librosa.filters.mel(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(magnitudes, min=clip_val) * C)

    def spectral_denormalize(self, magnitudes, C=1):
        return torch.exp(magnitudes) / C

    def mel_spectrogram(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

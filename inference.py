import numpy as np
import scipy.io.wavfile
import torch

import nv_wavenet
from model import WaveNet
from data import TacotronSTFT, load_wav_to_torch

MAX_WAV_VALUE = 32768.0

filter_length = 800
hop_length = 200
win_length = 800
sampling_rate = 16000

gpu_id = 2
device = torch.device('cuda:{}'.format(gpu_id)
                      if torch.cuda.is_available() else 'cpu')

stft = TacotronSTFT(filter_length=filter_length,
                    hop_length=hop_length,
                    win_length=win_length,
                    sampling_rate=sampling_rate)


def get_mel(audio):
    # TODO: WaveNetDatasetのメソッドと同じなのでutilsに切り出し
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = audio_norm.requires_grad_(False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def get_cond_input(mel, model):
    cond_input = model.upsample(mel)
    time_cutoff = model.upsample.kernel_size[0] - model.upsample.stride[0]
    cond_input = cond_input[:, :, :-time_cutoff]
    cond_input = model.cond_layers(cond_input).detach()
    cond_input = cond_input.view(
        cond_input.size(0), model.n_layers, -1, cond_input.size(2))
    # (channels, batch, num_layers, samples) のテンソルになるように入れ替え
    cond_input = cond_input.permute(2, 0, 1, 3)
    return cond_input


# TODO: data.mu_law_encode()とともにutilsに切り出し
def mu_law_decode_numpy(x, mu_quantization=256):
    # xはmu_law_encodeされている音声
    assert np.max(x) <= mu_quantization
    assert np.min(x) >= 0
    mu = mu_quantization - 1.0
    signal = 2 * (x / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) * np.abs(signal) - 1)
    return np.sign(signal) * magnitude


def main():
    model = WaveNet()
    checkpoint = torch.load(
        'runs/Oct09_11-24-52_K-00030-LIN/checkpoint_9000.pth')
    model.load_state_dict(checkpoint['model'])
    weights = model.export_weights()
    wavenet = nv_wavenet.NVWaveNet(**weights)

    # TODO: とりあえずバッチサイズ1で実験
    # TODO: 複数の音声をまとめて推論するときは長さをpaddingする必要あり
    filename = 'data/arctic_a0001.wav'
    audio, sampling_rate = load_wav_to_torch(filename)
    mel = get_mel(audio)
    mel.unsqueeze_(0)
    print(mel.shape)

    # NVWaveNetの入力に合うように整形
    # (channels, batch=1, num_layers, samples)
    cond_input = get_cond_input(mel, model)

    # 波形を生成
    # 生成された波形は mu-law された状態なので元に戻す必要がある
    audio_data = wavenet.infer(cond_input, nv_wavenet.Impl.AUTO)
    print(audio_data.shape)
    print(audio_data.min(), audio_data.max())

    # wavenet.Aはmu_quantization
    audio = mu_law_decode_numpy(audio_data[0].cpu().numpy(), wavenet.A)
    audio = MAX_WAV_VALUE * audio
    wavdata = audio.astype('int16')
    scipy.io.wavfile.write('gen.wav', 16000, wavdata)


if __name__ == "__main__":
    main()

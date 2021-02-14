import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, input_wavs_dir, output_wavs_dir, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, n_cache_reuse=1, shuffle=True,
                 fmax_loss=None, device=None, base_mels_path=None):
        self.audio_files = training_files
        self.input_wavs_dir = input_wavs_dir
        self.output_wavs_dir = output_wavs_dir
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        input_path = os.path.join(self.input_wavs_dir, filename + '.wav')
        output_path = os.path.join(self.output_wavs_dir, filename + '.wav')
        if self._cache_ref_count == 0:
            input_audio, sampling_rate = load_wav(input_path)
            input_audio = input_audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                input_audio = normalize(input_audio) * 0.95
            self.cached_input_wav = input_audio

            output_audio, sampling_rate_ = load_wav(output_path)
            output_audio = output_audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                output_audio = normalize(output_audio) * 0.95
            self.cached_output_wav = output_audio

            if sampling_rate != self.sampling_rate or sampling_rate != sampling_rate_:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            input_audio = self.cached_input_wav
            output_audio = self.cached_output_wav
            self._cache_ref_count -= 1

        input_audio = torch.FloatTensor(input_audio)
        input_audio = input_audio.unsqueeze(0)

        output_audio = torch.FloatTensor(output_audio)
        output_audio = output_audio.unsqueeze(0)

        assert input_audio.size(1) == output_audio.size(1), "Inconsistent dataset length, unable to sampling"

        if not self.fine_tuning:
            if self.split:
                if input_audio.size(1) >= self.segment_size:
                    max_audio_start = input_audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    input_audio = input_audio[:, audio_start:audio_start+self.segment_size]
                    output_audio = output_audio[:, audio_start:audio_start+self.segment_size]
                else:
                    input_audio = torch.nn.functional.pad(input_audio, (0, self.segment_size - input_audio.size(1)), 'constant')
                    output_audio = torch.nn.functional.pad(output_audio, (0, self.segment_size - input_audio.size(1)), 'constant')

            mel = mel_spectrogram(output_audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            mel = np.load(
                os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if input_audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    input_audio = input_audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                    output_audio = output_audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    input_audio = torch.nn.functional.pad(input_audio, (0, self.segment_size - input_audio.size(1)), 'constant')
                    output_audio = torch.nn.functional.pad(output_audio, (0, self.segment_size - input_audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(output_audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), input_audio.squeeze(0), output_audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from librosa.util import normalize
from scipy.io.wavfile import write
from dataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from generator import Generator
from utils import HParam
h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(
        x, hp.audio.filter_length, hp.audio.n_mel_channels,
        hp.audio.sampling_rate, hp.audio.hop_length,
        hp.audio.win_length, hp.audio.mel_fmin, None
    )


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, with_postnet=False):
    generator = Generator(hp.model.in_channels).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    #generator.remove_weight_norm()
    with torch.no_grad():
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav = wav / MAX_WAV_VALUE
            wav = normalize(wav) * 0.95
            wav = torch.FloatTensor(wav)
            wav = wav.reshape((1, 1, wav.shape[0],)).to(device)
            before_y_g_hat, y_g_hat = generator(wav, with_postnet)
            audio = before_y_g_hat.reshape((before_y_g_hat.shape[2],))
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(
                a.output_dir,
                os.path.splitext(filename)[0] + '_generated.wav'
            )
            write(output_file, hp.audio.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('-c', '--config', default='config.yaml')


    args = parser.parse_args()

    global hp
    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    torch.manual_seed(hp.train.seed)
    global device
    device = torch.device('cpu')

    inference(args)


if __name__ == '__main__':
    main()

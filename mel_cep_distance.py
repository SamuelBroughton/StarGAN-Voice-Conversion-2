import argparse
import mcd.metrics as mt
import os
import csv
import glob
import numpy as np
from mcd import dtw
from preprocess import get_sampling_rate, get_spk_world_feats


def mel_cep_dtw_dist(target, converted):
    """
    Compute the distance between two unaligned speech waveforms
    :param target: reference speech numpy array
    :param converted: synthesized speech numpy array
    :return: mel cep distance in dB
    """
    total_cost = 0
    total_frames = 0

    for (tar, conv) in zip(target, converted):
        tar, conv = tar.astype('float64'), conv.astype('float64')
        cost, _ = dtw.dtw(tar, conv, mt.logSpecDbDist)
        frames = len(tar)
        total_cost += cost
        total_frames += frames

    return total_cost / total_frames


def process_mcd_csv(convert_dir, spk_to_spks, output_csv):
    with open(os.path.join(convert_dir, output_csv), 'wt') as csv_f:
        csv_w = csv.writer(csv_f, delimiter=',')
        csv_w.writerow(['SPK_to_SPK', 'REFERENCE', 'SYNTHESIZED', 'MCD'])

        for spk_to_spk in spk_to_spks:
            _, trg = spk_to_spk.split('_to_')
            trg_files = glob.glob(os.path.join(convert_dir, spk_to_spk, f'{trg}*.wav'))
            vcto_files = glob.glob(os.path.join(convert_dir, spk_to_spk, '*-vcto-*.wav'))

            sample_rate = get_sampling_rate(trg_files[0])
            get_spk_world_feats(trg, trg_files, os.path.join(convert_dir, spk_to_spk), sample_rate)
            get_spk_world_feats('vcto', vcto_files, os.path.join(convert_dir, spk_to_spk), sample_rate)

            trg_files = glob.glob(os.path.join(convert_dir, spk_to_spk, f'{trg}*.npy'))
            vcto_files = glob.glob(os.path.join(convert_dir, spk_to_spk, '*-vcto-*.npy'))

            for idx, ref in enumerate(trg_files):
                synth = vcto_files[idx]
                dist = mel_cep_dtw_dist(np.load(ref), np.load(synth))
                print(f'MCD | {ref} to {synth} = {dist}')
                csv_w.writerow([spk_to_spk, os.path.basename(ref), os.path.basename(synth), dist])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    convert_dir_default = '../data/VCTK-Data/converted'
    output_csv_default = '../data/VCTK-Data/mcd.csv'

    parser.add_argument('--convert_dir', type=str, default=convert_dir_default, help='Dir containing converted speaker folders.')
    parser.add_argument('--spk_to_spk', type=str, nargs='+', required=True, help='spk_to_spk dirs in convert dir.')
    parser.add_argument('--output_csv', type=str, default=output_csv_default, help='csv file of results.')

    argv = parser.parse_args()

    convert_dir = argv.convert_dir
    spk_to_spk = argv.spk_to_spk
    output_csv = argv.output_csv

    process_mcd_csv(convert_dir, spk_to_spk, output_csv)

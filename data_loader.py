from torch.utils import data
import torch
import glob
from os.path import join, basename
import argparse
import numpy as np

min_length = 256   # Since we slice 256 frames from each utterance when training.


class DataSpeakers:
    """Storage of data speakers"""
    def __init__(self, dataset_using):
        # Build a dict useful when we want to get one-hot representation of speakers.
        if dataset_using == 'VCC2016':
            self.speakers = ['sf1', 'sf2', 'sf3', 'sm1', 'sm2', 'tm1', 'tm2', 'tm3', 'tf1', 'tf2']
        else:
            self.speakers = ['p262', 'p272', 'p229', 'p232', 'p292', 'p293', 'p360', 'p361', 'p248', 'p251']

        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    From Keras np_utils
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class MyDataset(data.Dataset):
    """Dataset for MCEP features and speaker labels."""
    def __init__(self, dataset_using, data_dir):
        data_speaker = DataSpeakers(dataset_using)
        self.speakers = data_speaker.speakers
        self.prefix_len = data_speaker.prefix_length
        self.spk2idx = data_speaker.spk2idx

        mc_files = glob.glob(join(data_dir, '*.npy'))
        mc_files = [i for i in mc_files if basename(i)[:self.prefix_len] in self.speakers]
        self.mc_files = self.rm_too_short_utt(mc_files)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        for f in self.mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!") 

    def rm_too_short_utt(self, mc_files, min_length=min_length):
        new_mc_files = []
        for mc_file in mc_files:
            mc = np.load(mc_file)
            if mc.shape[0] > min_length:
                new_mc_files.append(mc_file)
        return new_mc_files

    def sample_seg(self, feat, sample_len=min_length):
        assert feat.shape[0] - sample_len >= 0
        s = np.random.randint(0, feat.shape[0] - sample_len + 1)
        return feat[s:s+sample_len, :]

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        filename = self.mc_files[index]
        spk = basename(filename).split('_')[0]
        spk_idx = self.spk2idx[spk]
        mc = np.load(filename)
        mc = self.sample_seg(mc)
        mc = np.transpose(mc, (1, 0))  # (T, D) -> (D, T), since pytorch need feature having shape
        # to one-hot
        spk_cat = np.squeeze(to_categorical([spk_idx], num_classes=len(self.speakers)))

        return torch.FloatTensor(mc), torch.LongTensor([spk_idx]).squeeze_(), torch.FloatTensor(spk_cat)
        

class TestDataset(object):
    """Dataset for testing."""
    def __init__(self, dataset_using, data_dir, wav_dir, src_spk='p262', trg_spk='p272'):
        data_speaker = DataSpeakers(dataset_using)
        self.speakers = data_speaker.speakers
        self.spk2idx = data_speaker.spk2idx

        self.src_spk = src_spk
        self.trg_spk = trg_spk
        self.mc_files = sorted(glob.glob(join(data_dir, '{}*.npy'.format(self.src_spk))))

        self.src_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(src_spk)))
        self.trg_spk_stats = np.load(join(data_dir.replace('test', 'train'), '{}_stats.npz'.format(trg_spk)))
        
        self.logf0s_mean_src = self.src_spk_stats['log_f0s_mean']
        self.logf0s_std_src = self.src_spk_stats['log_f0s_std']
        self.logf0s_mean_trg = self.trg_spk_stats['log_f0s_mean']
        self.logf0s_std_trg = self.trg_spk_stats['log_f0s_std']
        self.mcep_mean_src = self.src_spk_stats['coded_sps_mean']
        self.mcep_std_src = self.src_spk_stats['coded_sps_std']
        self.mcep_mean_trg = self.trg_spk_stats['coded_sps_mean']
        self.mcep_std_trg = self.trg_spk_stats['coded_sps_std']
        self.src_wav_dir = f'{wav_dir}/{src_spk}'
        self.spk_idx = self.spk2idx[trg_spk]
        spk_cat = to_categorical([self.spk_idx], num_classes=len(self.speakers))
        self.spk_c_trg = spk_cat

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mc_file = self.mc_files[i]
            filename = basename(mc_file).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data       


def get_loader(dataset_using, data_dir, batch_size=32, mode='train', num_workers=1):
    dataset = MyDataset(dataset_using, data_dir)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test data loader')

    dataset_using_default = ['VCTK', 'VCC2016']
    train_dir_default = '../VCTK-Data/mc/train'

    # Data config.
    parser.add_argument('--dataset_using', type=str, default=dataset_using_default[0], help='VCTK or VCC2016')
    parser.add_argument('--train_dir', type=str, default=train_dir_default, help='Train dir path')

    argv = parser.parse_args()
    dataset_using = argv.dataset_using
    train_dir = argv.train_dir

    loader = get_loader(dataset_using, train_dir)
    data_iter = iter(loader)
    for i in range(10):
        mc_real, spk_label_org, spk_c_org = next(data_iter)
        print('-'*50)
        print(mc_real.size())
        print(spk_label_org.size())
        print(spk_c_org.size())
        print('-'*50)

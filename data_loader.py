from torch.utils import data
import torch
import glob
from os.path import join, basename
import numpy as np

min_length = 256  # Since we slice 256 frames from each utterance when training.


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

    def __init__(self, speakers_using, data_dir):
        self.speakers = speakers_using
        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])

        mc_files = glob.glob(join(data_dir, '*.npy'))
        mc_files = [i for i in mc_files if basename(i)[:self.prefix_length] in self.speakers]
        self.mc_files = self.rm_too_short_utt(mc_files)
        self.num_files = len(self.mc_files)
        print("\t Number of training samples: ", self.num_files)
        for f in self.mc_files:
            mc = np.load(f)
            if mc.shape[0] <= min_length:
                print(f)
                raise RuntimeError(
                    f"The data may be corrupted! We need all MCEP features having more than {min_length} frames!")

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
        return feat[s:s + sample_len, :]

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

    def __init__(self, speakers_using, data_dir, wav_dir, src_spk='p262', trg_spk='p272'):
        self.speakers = speakers_using
        self.spk2idx = dict(zip(self.speakers, range(len(self.speakers))))
        self.prefix_length = len(self.speakers[0])

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
        self.spk_idx_src, self.spk_idx_trg = self.spk2idx[src_spk], self.spk2idx[trg_spk]
        spk_cat_src = to_categorical([self.spk_idx_src], num_classes=len(self.speakers))
        spk_cat_trg = to_categorical([self.spk_idx_trg], num_classes=len(self.speakers))
        self.spk_c_org = spk_cat_src
        self.spk_c_trg = spk_cat_trg

    def get_batch_test_data(self, batch_size=8):
        batch_data = []
        for i in range(batch_size):
            mc_file = self.mc_files[i]
            filename = basename(mc_file).split('-')[-1]
            wavfile_path = join(self.src_wav_dir, filename.replace('npy', 'wav'))
            batch_data.append(wavfile_path)
        return batch_data


def get_loader(speakers_using, data_dir, batch_size=32, mode='train', num_workers=1):
    dataset = MyDataset(speakers_using, data_dir)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader

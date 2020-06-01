# StarGAN-Voice-Conversion-2

\** Converted samples coming soon  \**

A pytorch implementation based on: StarGAN-VC2: https://arxiv.org/pdf/1907.12279.pdf.

* Currently does not implement source-and-target adversarial loss.
* Makes use of gradient penalty.
* Doesnt make use of PS in G.

# Installation

**Tested on Python version 3.6.2 in a linux VM environment**

Recommended to use a linux environment - not tested for mac or windows OS 

## Python

* Create a new environment using Anaconda
```shell script
conda create -n stargan-vc python=3.6.2
```
* Install conda dependencies
```shell script
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
conda install pillow=5.4.1
conda install -c conda-forge librosa=0.6.1
conda install -c conda-forge tqdm=4.43.0
```

* Intall dependencies not available through conda using pip
```shell script
pip install pyworld=0.2.8
pip install mcd=0.4
```

**NB:** For mac users who cannot install pyworld see: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder

## Libraries

* Install binaries
  * SoX: https://sourceforge.net/projects/sox/files/sox/14.4.2/ 
  * libsndfile: http://linuxfromscratch.org/blfs/view/svn/multimedia/libsndfile.html
  * yasm: http://www.linuxfromscratch.org/blfs/view/svn/general/yasm.html
  * ffmpeg: https://ffmpeg.org/download.html
  * libav: https://libav.org/download/

# Usage

## Download Dataset

* [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

```shell script
mkdir ../data/VCTK-Data
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y
unzip VCTK-Corpus.zip -d ../data/VCTK-Data
```

If the downloaded VCTK is in tar.gz, run this:

```shell script
tar -xzvf VCTK-Corpus.tar.gz -C ../data/VCTK-Data
```

* VCC2016 and 2018 are yet to be included

## Preprocessing data

We will use Mel-Cepstral coefficients(MCEPs) here.

This example script is for the VCTK data which needs resampling to 16kHz, the script allows you to preprocess the data without resampling either. This script assumes the data dir to be `../data/VCTK-Data/`

```shell script
# VCTK-Data
python preprocess.py --perform_data_split y \
                     --resample_rate 16000 \
                     --origin_wavpath ../data/VCTK-Data/VCTK-Corpus/wav48 \
                     --target_wavpath ../data/VCTK-Data/VCTK-Corpus/wav16 \
                     --mc_dir_train ../data/VCTK-Data/mc/train \
                     --mc_dir_test ../data/VCTK-Data/mc/test \
                     --speaker_dirs p262 p272 p229 p232
```

# Training

* Currently only tested with conversion between 4 speakers
* Not yet tested with use of tensorboard

Example script:
```shell script
# example with VCTK
python main.py --train_data_dir ../data/VCTK-Data/mc/train \
               --test_data_dir ../data/VCTK-Data/mc/test \
               --use_tensorboard False \
               --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
               --model_save_dir ../data/VCTK-Data/models \
               --sample_dir ../data/VCTK-Data/samples \
               --num_iters 200000 \
               --batch_size 8 \
               --speakers p262 p272 p229 p232 \
               --num_speakers 4
```

If you encounter an error such as:

```shell script
ImportError: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found
```

You may need to export export LD_LIBRARY_PATH: (See [Stack Overflow](https://stackoverflow.com/questions/49875588/importerror-lib64-libstdc-so-6-version-cxxabi-1-3-9-not-found))

```shell script
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/<PATH>/<TO>/<YOUR>/.conda/envs/<ENV>/lib/
```

## Conversion

For example: restore model at step 120000 and specify the speakers

```shell script
# example with VCTK
python convert.py --resume_model 120000 \
                  --sampling_rate 16000 \
                  --num_speakers 4 \
                  --speakers p262 p272 p229 p232 \
                  --train_data_dir ../data/VCTK-Data/mc/train/ \
                  --test_data_dir ../data/VCTK-Data/mc/test/ \
                  --wav_dir ../data/VCTK-Data/VCTK-Corpus/wav16 \
                  --model_save_dir ../data/VCTK-Data/models \
                  --convert_dir ../data/VCTK-Data/converted \
                  --num_converted_wavs 4
```

This saves your converted flies to `../data/VCTK-Data/converted/120000/`

## Calculate Mel Cepstral Distortion

Calculate the Mel Cepstral Distortion of the reference speaker vs the synthesized speaker. Use `--spk_to_spk` tag to define multiple speaker to speaker folders generated with the convert script.

```shell script
python mel_cep_distance.py --convert_dir ../data/VCTK-Data/converted/120000 \
                           --spk_to_spk p262_to_p272 \
                           --output_csv p262_to_p272.csv
```

# TODO:
- [ ] Include converted samples
- [ ] Include MCD examples
- [ ] Include s-t loss like original paper

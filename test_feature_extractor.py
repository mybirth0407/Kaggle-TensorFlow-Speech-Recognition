import librosa
import numpy as np
import os
from os import listdir
from os import mkdir
from os.path import isfile
from os.path import isdir
import multiprocessing
from multiprocessing import Pool
import sys
import h5py
import scipy.signal
import json
import shutil

# audio paths
test_audio_path = './test/audio/'

# information file
if not isdir('./feature'):
  mkdir('./feature')
test_feature_path = './feature/test/'
if not isdir(test_feature_path):
  mkdir(test_feature_path)

# error list
error_list = []

# global variable
eps = np.spacing(1)

# parameter
global param

def main(argv):
  # must be meta data file
  if 1 >= len(argv):
    print('error!, meta.json plz..')
    sys.exit(0)

  # meta data file must be json file extension
  _, ext = os.path.splitext(argv[1])
  # no more need
  _ = None
  if '.json' != ext:
    print('error!, first arugment must be json file extension')
    sys.exit(0)

  with open(argv[1]) as json_file:
    global param
    param = json.load(json_file)

  print(param)

  get_test_feature_extract()


def get_test_feature_extract():
  # number of cpu thread 
  n_processes = multiprocessing.cpu_count()
  pool = Pool(processes=n_processes - 1)
  files = listdir(test_audio_path)

  pool.map(
    get_feature, files
  )
  pool.close()

  print('test feature extractor done!')

def get_feature(file):
  basename = os.path.basename(os.path.normpath(file))
  file_test, _ = os.path.splitext(basename)
  file_test = test_feature_path + file_test
  file_test += '.h5'
  if isfile(file_test):
    return

  try:
    file = test_audio_path + file
    y, sr = load_audio(file)
    mfcc = get_mfcc(y)
    mel = get_mel(y)
  except Exception as e:
    print(e)
    return

  feature_vector = np.empty((0, len(mfcc[0]) + len(mel[0])))

  for i in range(len(mfcc)):
    feature = np.hstack([mfcc[i], mel[i]])
    feature_vector = np.vstack((feature_vector, feature))

  save_hdf(file_test, feature_vector)

def load_audio(file_path):
  y, sr = librosa.load(file_path, sr=param.get('sample_rate'))
  if len(y) > sr:
    a = len(y) // sr
    y = y[:sr * a]
  else:
    y = np.pad(y, (0, max(0, sr - len(y))), "constant")

  return y, sr

def save_hdf(file, arr):
  h5f = h5py.File(file, 'w')
  h5f.create_dataset('feature', data=arr)
  h5f.close()

def get_window(win_len, win_type):
  if win_type == 'hamming_asymmetric':
      return scipy.signal.hamming(win_len, sym=False)
  elif win_type == 'hamming_symmetric':
      return scipy.signal.hamming(win_len, sym=True)
  elif win_type == 'hann_asymmetric':
      return scipy.signal.hann(win_len, sym=False)
  elif win_type == 'hann_symmetric':
      return scipy.signal.hann(win_len, sym=True)

def get_mel(y):
  win_length = int(param.get('win_length') * param.get('sample_rate'))
  hop_length = int(param.get('hop_length') * param.get('sample_rate'))

  window_ = get_window(win_length, param.get('window'))

  mel_basis = librosa.filters.mel(
        sr=param.get('sample_rate'),
        n_fft=param.get('n_fft'),
        n_mels=param.get('n_mels'),
        fmin=param.get('fmin'),
        fmax=param.get('fmax'),
        htk=param.get('htk_mel')
  )

  spectrogram_ = np.abs(librosa.stft(
      y + eps,
      n_fft=param.get('n_fft'),
      win_length=win_length,
      hop_length=hop_length,
      center=param.get('center'),
      window=window_
  ))
  
  mel_spectrum = np.dot(mel_basis, spectrogram_)
  if param.get('log_mel'):
      mel_spectrum = np.log(mel_spectrum + eps)

  return mel_spectrum.T

def get_mfcc(y):
  win_length = int(param.get('win_length') * param.get('sample_rate'))
  hop_length = int(param.get('hop_length') * param.get('sample_rate'))
  
  window_ = get_window(win_length, param.get('window'))

  mel_basis = librosa.filters.mel(
      sr=param.get('sample_rate'),
      n_fft=param.get('n_fft'),
      n_mels=param.get('n_mels'),
      fmin=param.get('fmin'),
      fmax=param.get('fmax'),
      htk=param.get('htk_mfcc')
  )
  
  spectrogram_ = np.abs(librosa.stft(
      y + eps,
      n_fft=param.get('n_fft'),
      win_length=win_length,
      hop_length=hop_length,
      center=param.get('center'),
      window=window_
  ))
  
  mel_spectrum = np.dot(mel_basis, spectrogram_)

  mfcc = librosa.feature.mfcc(
      S=librosa.logamplitude(mel_spectrum),
      n_mfcc=param.get('n_mfcc')
  )
  
  return mfcc.T


if __name__ == '__main__':
  main(sys.argv)

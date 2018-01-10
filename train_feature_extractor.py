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
train_audio_path = './train/audio/'
train_feature_path = './feature/train/'
try:
  if sys.argv[2] == 'aug':
    print('augmentation data')
    if not isdir('./feature/augmentation'):
      mkdir('./feature/augmentation')
    train_audio_path = './augmentation/audio/'
    train_feature_path = './feature/augmentation/train/'
except Exception as e:
  print(e)

# information file
validation_file = './train/validation_list.txt'
testing_file = './train/testing_list.txt'
if not isdir('./feature'):
  mkdir('./feature')
if not isdir(train_feature_path):
  mkdir(train_feature_path)

# error list
error_list = []

# except labels
except_label = ['_background_noise_']

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

  with open(validation_file) as val_file:
    val_list = np.loadtxt(val_file, dtype='str')

  with open(testing_file) as test_file:
    test_list = np.loadtxt(test_file, dtype='str')

  # check parameter
  print(param)

  get_train_feature_extract(val_list, test_list)


def get_train_feature_extract(val_list, test_list):
  # number of cpu thread 
  n_processes = multiprocessing.cpu_count()

  # get train audio label
  labels = listdir(train_audio_path)
  labels.sort()
  # using process pool(like thread pool)
  pool = Pool(processes=n_processes)
  pool.map(
      get_feature_path, [(train_audio_path + label) for label in labels]
  )
  pool.close()

  try:
    if sys.argv[2] == 'aug':
      return
  except Exception as e:
    print(e)
  # move files according data set
  # validation set
  if not isdir(train_feature_path + 'validation'):
    mkdir(train_feature_path + 'validation')

  for val_file in val_list:
    label = val_file.split('/')[0]
    file, _ = os.path.splitext(val_file)
    file += '.h5'

    if not isdir(train_feature_path + 'validation/' + label):
      mkdir(train_feature_path + 'validation/' + label)

    shutil.move(
        train_feature_path + file,
        train_feature_path + 'validation/' + file
    )

  # test set
  if not isdir(train_feature_path + 'test'):
    mkdir(train_feature_path + 'test')

  for test_file in test_list:
    label = test_file.split('/')[0] 
    file, _ = os.path.splitext(test_file)
    file += '.h5'

    if not isdir(train_feature_path + 'test/' + label):
      mkdir(train_feature_path + 'test/' + label)

    shutil.move(train_feature_path + file,
        train_feature_path + 'test/' + file
    )

  # train set
  if not isdir(train_feature_path + 'train'):
    mkdir(train_feature_path + 'train')

  will_move = listdir(train_feature_path)
  will_move.remove('validation')
  will_move.remove('test')

  for f in will_move:
    shutil.move(train_feature_path + f, train_feature_path + 'train')

  print('train feature extractor done!')

def get_feature_path(arg):
  path = arg
  file_list = listdir(path)

  # extract last dir
  # if dir '/a/b/c', we get 'c'
  label = os.path.basename(os.path.normpath(path))
  print(label + ' is start!')
  for file in file_list:
    get_feature(path + '/' + file, label)
  print(label + ' is done!')
  return

def get_feature(file, label):
  if label in except_label:
    label = 'silence'

  try:
    basename = os.path.basename(os.path.normpath(file))
    file_test, _ = os.path.splitext(basename)
    file_test = train_feature_path + label + '/' + file_test
    file_test += '.h5'
  except Exception as e:
    print(e)
    return
  
  if not isdir(train_feature_path + label):
    mkdir(train_feature_path + label)

  if isfile(file_test):
    return
    
  try:
    y, sr = load_audio(file)
    # start, end = split_silence(y)
    # mel = get_mel(y)
    # mel = get_mel(y)[start:end]
    mfcc = get_mfcc(y[0])
    for i in range(1, len(y)):
      mfcc = np.vstack((mfcc, get_mfcc(y[i])))
    # mfcc_del = get_mfcc_delta(mfcc)
    # mfcc_acc = get_mfcc_acceleration(mfcc)
    # print(mel.shape)
    # print(mfcc.shape)
    # print(mfcc_del.shape)
    # print(mfcc_acc.shape)

  except Exception as e:
    print(e)
    return

  # 1 + len(mel[0]) ...
  # 1 is label location
  feature_vector = np.empty((
      0, 1 + len(mfcc[0])
  ))

  for i in range(len(mfcc)):
    feature = np.hstack(
        [mfcc[i], str.encode(label)]
    )
    feature_vector = np.vstack((feature_vector, feature))

  save_hdf(file_test, feature_vector)

def stop():
  while True: pass

def load_audio(file_path):
  y, sr = librosa.load(file_path, sr=param.get('sample_rate'))
  if len(y) > sr:
    a = len(y) // sr
    y = y[:sr * a]
    y = [y[i::a] for i in range(a)]
  else:
    y = [np.pad(y, (0, max(0, sr - len(y))), "constant")]

  return y, sr

def split_silence(y):
  win_length = int(param.get('win_length') * param.get('sample_rate'))
  hop_length = int(param.get('hop_length') * param.get('sample_rate'))

  c = []
  start = 0

  for i in range(0, len(y), hop_length):
      x = y[i:i + win_length]
      c.append(np.abs(np.var(x)))

  end = len(c)

  for i in range(1, len(c) - 1):
      if c[i] > 3 * c[i-1]:
          start = i - 1
          break

  for i in range(1, len(c) - 1):
      if 3 * c[i] < c[i-1]:
          end = i + 1

  if start + 5 > end:
    end = len(c) - 1

  return start, end

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

def get_mfcc_delta(mfcc):
  delta = librosa.feature.delta(mfcc, param.get('width'))

  # mfcc is already .T
  return delta

def get_mfcc_acceleration(mfcc):
  acceleration = librosa.feature.delta(
      mfcc,
      order=2,
      width=param.get('width')
  )

  # mfcc is already .T
  return acceleration


if __name__ == '__main__':
  main(sys.argv)

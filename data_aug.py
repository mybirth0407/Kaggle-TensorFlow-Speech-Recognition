import numpy as np
import random
import librosa
import os
from os import listdir
from os import mkdir
from os.path import isdir
from os.path import isfile
import json
import sys
import multiprocessing
from multiprocessing import Pool

if not isdir('./augmentation'):
  mkdir('./augmentation')

if not isdir('./augmentation/audio'):
  mkdir('./augmentation/audio')

meaningful_label = ['down', 'go', 'left', 'no', 'off',
                    'on', 'right', '_background_noise_', 'stop', 'up', 'yes']
meaningful_label.sort()

with open(sys.argv[1]) as json_file:
  global param
  param = json.load(json_file)

# audio paths
train_audio_path = './train/audio/'

def main(argv):
  n_processes = multiprocessing.cpu_count()
  # get train audio label

  # using process pool(like thread pool)
  pool = Pool(processes=n_processes)
  pool.map(
    aug_path, [(train_audio_path + label) for label in meaningful_label]
  )
  pool.close()


def aug_path(arg):
  path = arg
  file_list = listdir(path)

  # extract last dir
  # if dir '/a/b/c', we get 'c'
  label = os.path.basename(os.path.normpath(path))
  print(label + ' is start!')
  for file in file_list:
    do_aug(os.path.join(path, file))
  print(label + ' is done!')
  return

def do_aug(file_path):
  y, sr = load_audio(file_path)
  if sr == 0:
    return
  add_white_noise(file_path, y, sr)
  amp_sound(file_path, y, sr)
  shift_sound_right(file_path, y, sr)
  shift_sound_left(file_path, y, sr)
  y, sr = None, None
  return

def add_white_noise(file_path, y, sr):
  # Adding white noise 
  new_y = y
  start, end = split_silence(y)

  # add noise
  white_noise = np.random.randn(start)
  new_y[:start] += 0.005 * white_noise

# amplification
  new_y[start:end] *= 1.5

# add noise
  white_noise = np.random.randn(len(y) - end)
  new_y[end:] += 0.005 * white_noise

  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    if label == '_background_noise_':
      label = 'silence'
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += '_wn' + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )

def amp_sound(file_path, y, sr):
  new_y = y
  new_y = y * 1.5

  file_path = file_path.replace('train', 'augmentation')

  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += '_amp' + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )

def shift_sound_right(file_path, y, sr):
  # Adding white noise 
  new_y = y
  start, end = split_silence(y)

  # add noise
  white_noise = np.random.randn(1)
  new_y[:start] += 0.005 * white_noise

# amplification
  new_y[start:end] *= 1.5

# add noise
  white_noise = np.random.randn(1)
  new_y[end:] += 0.005 * white_noise[0]

  new_y = np.roll(y, int(-sr / 5))

  file_path = file_path.replace('train', 'augmentation')

  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += 'sft_r' + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )

def shift_sound_left(file_path, y, sr):
  new_y = np.roll(y, int(-sr / 5))

  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += 'sft_l' + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )

def load_audio(file_path):
  try:
    y, sr = librosa.load(file_path, sr=param.get('sample_rate'))
    if len(y) > sr:
      a = len(y) // sr
      y = y[:sr * a]
    else:
      y = [np.pad(y, (0, max(0, sr - len(y))), "constant")]

    return y, sr
  except Exception as e:
    print(file_path)
    print(e)
    return 0, 0


def split_silence(y):
  win_length = int(0.04 * 16000)
  hop_length = int(0.02 * 16000)

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

if __name__ == '__main__':
  main(sys.argv)
  
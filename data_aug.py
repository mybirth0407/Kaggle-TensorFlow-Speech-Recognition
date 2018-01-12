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

  labels = listdir(train_audio_path)
  # using process pool(like thread pool)
  pool = Pool(processes=n_processes)
  pool.map(
    aug_path, [(train_audio_path + label) for label in labels]
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
    
  dir_name = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(dir_name))

  amp_noise_shift(file_path, y, sr)
  
  if label in meaningful_label:
    if label.find('_background_noise_') != -1:
      silence(file_path, y, sr)
    else:
      amp_sound(file_path, y, sr)
      add_white_noise(file_path, y, sr)
      shift_sound_right(file_path, y, sr)
      shift_sound_left(file_path, y, sr)
      stretch_sound(file_path, y, sr, 1.2)
      stretch_sound(file_path, y, sr, 1.1)
      stretch_sound(file_path, y, sr, 0.8)
      stretch_sound(file_path, y, sr, 0.9)
      stretch_sound2(file_path, y, sr, 0.85)
      stretch_sound2(file_path, y, sr, 1.15)
  y, sr = None, None
  return
  
def silence(file_path, y, sr):
  num_sample = 100
  divide_num = int(len(y) / sr) + 1
  for i in range(0, divide_num - 1) :
    new_y = np.zeros(sr, dtype=float)
    new_y[:] = y[sr * i:sr * (i + 1)]

    for j in range(1, num_sample):
      ratio = 0.99 - 0.001 * j
      new_y *= ratio
      # save
      cur_dir = os.path.dirname(file_path)
      label = os.path.basename(os.path.normpath(cur_dir))
      if not isdir('./augmentation/audio/' + label):
        mkdir('./augmentation/audio/' + label)
      path = cur_dir.replace('train', 'augmentation')
      file = os.path.basename(os.path.normpath(file_path))
      file, ext = os.path.splitext(file)
      file += '_' + str(j) + '_' + str(i) + 'th' + ext
      fname = os.path.join(path, file)

      if isfile(fname):
        return
      else:
        librosa.output.write_wav(
            fname, y=new_y, sr=sr, norm=False
        )
        
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


def stretch_sound(file_path, y, sr, volume):
  # stretch the sound (by volume)
  start, end = split_silence(y)
  white_noise = np.random.randn()

  new_y_len = int(len(y) * volume)
  new_y = np.array(y, dtype=np.float)

  for i in range(1, len(y)):
    position_num = len(y) / new_y_len * i
    position_index = int(position_num) + start

    if position_index < len(y) - 1:
      interpolation = position_num - int(position_num)
      new_y[i] = interpolation * y[position_index + 1]\
          + (1 - interpolation) * y[position_index]
    else:
      new_y[i] = 0.005 * white_noise

  # save
  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += '_stretch_'+ str(volume) + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )


def stretch_sound2(file_path, y, sr, volume):
  # stretch sound and make some effect at sound
  start, end = split_silence(y)
  white_noise = np.random.randn()
  new_y = librosa.effects.time_stretch(y, volume)

  # save
  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)

  file += '_stretch_'+ str(volume) + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )


def amp_noise_shift(file_path, y, sr):
  # shift the sound by (500 + start) and amp it by 1.5
  # and mix the noise (0.005 * white_noise)
  start, end = split_silence(y)
  new_y = y
  for i in range(0, len(y) - 1):
    test = i + start + 500
    if test > len(y) - 1:
      j = test - len(y)
    else:
      j = i + start + 500
    new_y[i] = 1.5 * y[j]
    new_y[i] += 0.005 * np.random.randn()

  # save
  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += '_amp_noise_shift_' + ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )


def amp_sound(file_path, y, sr):
  # amp the sound by 1.5
  new_y = y
  new_y = y * 1.5
  # save
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
      y = np.pad(y, (0, max(0, sr - len(y))), "constant")
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
  



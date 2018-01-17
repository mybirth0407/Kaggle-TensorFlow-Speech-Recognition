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
  labels.sort(key=str.lower)
  # sort here, if you don't make the silence set first with background noise
  # there can be a error

  s_files = listdir(train_audio_path + '_background_noise_')
  pool = Pool(processes=n_processes)
  pool.map(
    aug_file, [(train_audio_path + '_background_noise_/' + file) for file in s_files]
  )
  pool.close()

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

def aug_file(arg):
  y, sr = load_audio(arg)
  
  if sr == 0:
    return
    
  silence(arg, y, sr)

def do_aug(file_path):
  y, sr = load_audio(file_path)
  
  if sr == 0:
    return
    
  dir_name = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(dir_name))

  if label in meaningful_label:
    if label.find('_background_noise_') != -1:
      silence(file_path, y, sr)

    else:
      amp_sound(file_path, y, sr)
      stretch_sound(file_path, y, sr, 1.2)
      stretch_sound(file_path, y, sr, 0.8)
      stretch_sound2(file_path, y, sr, 0.85)
      stretch_sound2(file_path, y, sr, 1.15)
      pitch_up(file_path, y, sr)
      pitch_down(file_path, y, sr)
      shift_sound_right(file_path, y, sr)
      shift_sound_left(file_path, y, sr)

  else:
    aug_type_1 = 0 
    aug_type_2 = 0

    while(aug_type_1 == aug_type_2):
      aug_type_1 = random.randrange(1,10)
      aug_type_2 = random.randrange(1,10)

    if aug_type_1 == 1 or aug_type_2 == 1:
      amp_sound(file_path, y, sr)
    if aug_type_1 == 2 or aug_type_2 == 2:
      stretch_sound(file_path, y, sr, 1.2)
    if aug_type_1 == 3 or aug_type_2 == 3:
      stretch_sound(file_path, y, sr, 0.8)
    if aug_type_1 == 4 or aug_type_2 == 4:
      stretch_sound2(file_path, y, sr, 0.85)
    if aug_type_1 == 5 or aug_type_2 == 5:
      stretch_sound2(file_path, y, sr, 1.15)
    if aug_type_1 == 6 or aug_type_2 == 6:
      pitch_up(file_path, y, sr)
    if aug_type_1 == 7 or aug_type_2 == 7:
      pitch_down(file_path, y, sr)
    if aug_type_1 == 8 or aug_type_2 == 8:
      shift_sound_right(file_path, y, sr)
    if aug_type_1 == 9 or aug_type_2 == 9:
      shift_sound_left(file_path, y, sr)
      




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

  file += '_wn'

  file += ext

  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )


def stretch_sound(file_path, y, sr, ratio):
  # stretch the sound (by ratio)
  start, end = split_silence(y)
  white_noise = np.random.randn()

  new_y_len = int(len(y) * ratio)
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

  file += '_stretch_'+ str(ratio)

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext

  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )


def stretch_sound2(file_path, y, sr, ratio):
  # stretch sound and make some effect at sound
  start, end = split_silence(y)
  white_noise = np.random.randn()
  new_y = librosa.effects.time_stretch(y, ratio)

  # save
  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)

  file += '_stretch_'+ str(ratio)

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
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

  y_len = len(y)

  for i in range(0, y_len - 1):
    test = i + start + 500
    if test > y_len - 1:
      j = test - y_len
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

  file += '_amp_noise_shift'

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
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
  file += '_amp'

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )

def pitch_down(file_path, y, sr):
  new_y = np.array(y, dtype=np.float)
  new_y = librosa.effects.pitch_shift(y, sr, n_steps=-20, bins_per_octave=60)
  
  # save
  file_path = file_path.replace('train', 'augmentation')
  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += '_pitch_down'

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )

def pitch_up(file_path, y, sr):
  new_y = np.array(y, dtype=np.float)
  new_y = librosa.effects.pitch_shift(y, sr, n_steps=10, bins_per_octave=40)
  
  # save
  file_path = file_path.replace('train', 'augmentation')
  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += '_pitch_up'

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
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
  file += 'sft_r'

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
        fname, y=new_y, sr=sr, norm=False
    )

def shift_sound_left(file_path, y, sr):
  new_y = np.roll(y, int(sr / 5))

  cur_dir = os.path.dirname(file_path)
  label = os.path.basename(os.path.normpath(cur_dir))
  if not isdir('./augmentation/audio/' + label):
    mkdir('./augmentation/audio/' + label)
  path = cur_dir.replace('train', 'augmentation')
  file = os.path.basename(os.path.normpath(file_path))
  file, ext = os.path.splitext(file)
  file += 'sft_l'

  noise_num_1 = 0
  noise_num_2 = 0
  noise_num_3 = 0

  while(noise_num_1 == noise_num_2 or noise_num_2 == noise_num_3):
    noise_num_1 = random.randrange(1,7)
    noise_num_2 = random.randrange(1,7)
    noise_num_3 = random.randrange(1,7)

  if noise_num_1 == 1 or noise_num_2 == 1 or noise_num_3 == 1:
    noise1(path, file, new_y, sr)
  if noise_num_1 == 2 or noise_num_2 == 2 or noise_num_3 == 2:
    noise2(path, file, new_y, sr)
  if noise_num_1 == 3 or noise_num_2 == 3 or noise_num_3 == 3:
    noise3(path, file, new_y, sr)
  if noise_num_1 == 4 or noise_num_2 == 4 or noise_num_3 == 4:
    noise4(path, file, new_y, sr)
  if noise_num_1 == 5 or noise_num_2 == 5 or noise_num_3 == 5:
    noise5(path, file, new_y, sr)
  if noise_num_1 == 6 or noise_num_2 == 6 or noise_num_3 == 6:
    noise6(path, file, new_y, sr)

  file += ext
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

def noise1(path, file, y, sr):
  # noise1 is a white noise

  noise_power = random.randrange(1,15)
  max_y = max(y)

  if max_y > 0.65:
    noise_power += 80 
    noise_file = './augmentation/audio/_background_noise_/white_noise_' + str(noise_power) + '_1th.wav'
  elif max_y > 0.25:
    noise_power += 65 
    noise_file = './augmentation/audio/_background_noise_/white_noise_' + str(noise_power) + '_1th.wav'
  else:
    noise_power += 50 
    noise_file = './augmentation/audio/_background_noise_/white_noise_' + str(noise_power) + '_1th.wav'
  
  y1, sr1 = load_audio(noise_file)
  new_y = np.array(y1, dtype=np.float)
  len_min = min(len(y), len(y1)) - 1
  new_y[:len_min] = y[:len_min] + y1[:len_min]

    # save

  file += '_noise1' + '.wav'
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )

def noise2(path, file, y, sr):
  # noise2 is a pink noise

  noise_power = random.randrange(1,15)
  max_y = max(y)

  if max_y > 0.65:
    noise_power += 80 
    noise_file = './augmentation/audio/_background_noise_/pink_noise_' + str(noise_power) + '_1th.wav'
  elif max_y > 0.25:
    noise_power += 65 
    noise_file = './augmentation/audio/_background_noise_/pink_noise_' + str(noise_power) + '_1th.wav'
  else:
    noise_power += 50 
    noise_file = './augmentation/audio/_background_noise_/pink_noise_' + str(noise_power) + '_1th.wav'
  
  y1, sr1 = load_audio(noise_file)
  new_y = np.array(y1, dtype=np.float)
  len_min = min(len(y), len(y1)) - 1
  new_y[:len_min] = y[:len_min] + y1[:len_min]

    # save

  file += '_noise2' + '.wav'
  fname = os.path.join(path, file)
 
  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )

def noise3(path, file, y, sr):
  # noise3 is a running tap

  noise_power = random.randrange(1,15)
  max_y = max(y)

  if max_y > 0.65:
    noise_power += 80 
    noise_file = './augmentation/audio/_background_noise_/running_tap_' + str(noise_power) + '_1th.wav'
  elif max_y > 0.25:
    noise_power += 65 
    noise_file = './augmentation/audio/_background_noise_/running_tap_' + str(noise_power) + '_1th.wav'
  else:
    noise_power += 50 
    noise_file = './augmentation/audio/_background_noise_/running_tap_' + str(noise_power) + '_1th.wav'
  
  y1, sr1 = load_audio(noise_file)
  new_y = np.array(y1, dtype=np.float)
  len_min = min(len(y), len(y1)) - 1
  new_y[:len_min] = y[:len_min] + y1[:len_min]



    # save

  file += '_noise3' + '.wav'
  fname = os.path.join(path, file)
  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )


def noise4(path, file, y, sr):
  # noise4 is a bike

  noise_power = random.randrange(1,15)
  max_y = max(y)

  if max_y > 0.65:
    noise_power += 80 
    noise_file = './augmentation/audio/_background_noise_/exercise_bike_' + str(noise_power) + '_1th.wav'
  elif max_y > 0.25:
    noise_power += 65 
    noise_file = './augmentation/audio/_background_noise_/exercise_bike_' + str(noise_power) + '_1th.wav'
  else:
    noise_power += 50 
    noise_file = './augmentation/audio/_background_noise_/exercise_bike_' + str(noise_power) + '_1th.wav'
  
  y1, sr1 = load_audio(noise_file)
  new_y = np.array(y1, dtype=np.float)
  len_min = min(len(y), len(y1)) - 1
  new_y[:len_min] = y[:len_min] + y1[:len_min]

    # save

  file += '_noise4' + '.wav'
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )


def noise5(path, file, y, sr):
  # noise5 is a miaowing

  noise_power = random.randrange(1,15)
  max_y = max(y)

  if max_y > 0.65:
    noise_power += 80 
    noise_file = './augmentation/audio/_background_noise_/dude_miaowing_' + str(noise_power) + '_1th.wav'
  elif max_y > 0.25:
    noise_power += 65 
    noise_file = './augmentation/audio/_background_noise_/dude_miaowing_' + str(noise_power) + '_1th.wav'
  else:
    noise_power += 50 
    noise_file = './augmentation/audio/_background_noise_/dude_miaowing_' + str(noise_power) + '_1th.wav'
  
  y1, sr1 = load_audio(noise_file)
  new_y = np.array(y1, dtype=np.float)
  len_min = min(len(y), len(y1)) - 1
  new_y[:len_min] = y[:len_min] + y1[:len_min]


    # save

  file += '_noise5' + '.wav'
  fname = os.path.join(path, file)
 
  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )

def noise6(path, file, y, sr):
  # noise6 is a dish

  y1, sr1 = load_audio(
    './augmentation/audio/_background_noise_/doing_the_dishes_50_9th.wav')

  noise_power = random.randrange(1,15)
  max_y = max(y)

  if max_y > 0.65:
    noise_power += 80 
    noise_file = './augmentation/audio/_background_noise_/doing_the_dishes_' + str(noise_power) + '_1th.wav'
  elif max_y > 0.25:
    noise_power += 65 
    noise_file = './augmentation/audio/_background_noise_/doing_the_dishes_' + str(noise_power) + '_1th.wav'
  else:
    noise_power += 50 
    noise_file = './augmentation/audio/_background_noise_/doing_the_dishes_' + str(noise_power) + '_1th.wav'
  
  y1, sr1 = load_audio(noise_file)
  new_y = np.array(y1, dtype=np.float)
  len_min = min(len(y), len(y1)) - 1
  new_y[:len_min] = y[:len_min] + y1[:len_min]



    # save

  file += '_noise6' + '.wav'
  fname = os.path.join(path, file)

  if isfile(fname):
    return
  else:
    librosa.output.write_wav(
      fname, y=new_y, sr=sr, norm=False
    )


if __name__ == '__main__':
  main(sys.argv)
  


# TFSpeechRocgnitionChallenge

base directory structure

|-- cnn/
|   |-- feature/
|       |-- augmentation/
|           |-- train/
|               |-- down/
|               |-- go/
|               |-- .. meaningful labels(include silence)
|       |-- train/
|           |-- bed/
|           |-- bird/
|           |-- .. all labels(include silence)
|   |-- hdf/
|   |-- model/
|   |-- pred/
|   |-- cnn_model.py
|   |-- cnn_model_all.py
|   |-- cnn_test.py
|   |-- data_combine.py
|   |-- data_summary.py
|   |-- meta.json
|   |-- test_feature_extractor.py
|   |-- train_feature_extractor.py
|-- dnn/ # todo
|-- README.md
|-- data_aug.py
|-- meta.json
|-- sample_submission.7z
|-- small.ipynb
|-- test/ # without git content, download kaggle
|-- train/ # without git content, download kaggle


Usage:
  First, git repo clone to your working directory
  > git clone https://github.com/mybirth0407/TFSpeechRocgnitionChallenge
  > cd TFSpeechRocgnitionChallenge/

  If you wanna data augmentation(only meaningful label)
  Follow this step,
  > python3 data_aug.py meta.json

  This operation making directory 'augmentation/' in base directory

  |-- augmentation/
  |   |-- audio/
  |       | -- _background_noise_/
  |       | -- down/
  |       | -- .. meaningful labels(include '_background_noise_') # feature is silence

  Each label directory has augmentationed wav file(amp, shift, add white noise)

  Second, into 'cnn/' or 'dnn/', feature extract
  This example is CNN
  Follow this steps,
  > cd cnn
  > python3 train_feature_extractor.py meta.json # train data feature extract

  If you have done augmentation,
  > python3 train_feature_extractor.py meta.json aug # augmentation data feature extract

  > python3 test_feature_extractor.py meta.json # test feature extract

  This operation making directory 'feature/' in 'cnn/'

  |-- feature/
  |   |-- augmentation/
  |       |-- train/
  |           |-- down/
  |               |-- .h5 feature files
  |           |-- go/
  |               |-- .h5 feature files
  |           |-- .. meaningful labels(include silence)
  |   |-- test/
  |       |-- .h5 feature files(158538 files)
  |   |-- train/
  |       |-- test/
  |           |-- bed/
  |               |-- .h5 feature files
  |           |-- bird/
  |               |-- .h5 feature files
  |           |-- .. labels based on 'testing_list.txt'
  |       |-- train/
  |           |-- bed/
  |               |-- .h5 feature files
  |           |-- bird/
  |               |-- .h5 feature files
  |           |-- .. labels
  |               |-- .h5 feature files
  |       |-- validation/
  |           |-- bed/
  |               |-- .h5 feature files
  |           |-- bird/
  |               |-- .h5 feature files
  |           |-- .. labels based on 'validation_list.txt'
  |               |-- .h5 feature files

  Next, combine the train feature files created in the second step(To reduce file IO)
  > python3 data_combine.py train 40

  If you have done augmentation,
  > python3 data_combine.py aug 60

  > python3 data_combine.py test 3
  > python3 data_combine.py validation 3

  40, 60, 3, 3 is number of files after combine

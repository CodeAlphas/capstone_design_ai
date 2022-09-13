import pandas as pd
import os
import librosa
import numpy as np
from collections import defaultdict
from soundfile import SoundFile

class DataExtractor:
    def __init__(self, audio_config, emotions, balance, verbose):

        self.audio_config = audio_config # {'mfcc': True, 'chroma': True, 'mel': True}
        self.emotions = emotions # ['angry', 'neutral', 'happy', 'fear']
        self.balance = balance # 각 감정으로 레이블된 데이터의 수를 동일하게 맞출지 여부
        self.verbose = verbose # log 출력 여부, 1은 출력
        self.features_folder_name = "features" # 음성 데이터의 특성값을 저장한 npy 파일을 담을 폴더 이름 지정

    # 음성 데이터에서 특성값과 레이블된 감정값, 저장된 경로값 등을 추출하고 추출한 데이터를 처리해주는 함수
    def load_metadata(self, desc_files, partition, shuffle):
        self.load_metadata_from_desc_file(desc_files, partition)
        if self.balance:
            self.balance_data(partition)
        if shuffle:
            self.shuffle_data_by_partition(partition)

    # partition별로 음성 데이터에서 특성값, 레이블된 감정값, 저장 경로값 등의 정보를 추출하는 함수
    def load_metadata_from_desc_file(self, desc_files, partition):
        if not os.path.isdir(self.features_folder_name):
            os.mkdir(self.features_folder_name)

        df = pd.DataFrame({'path': [], 'emotion': []}) # Columns(열): [path, emotion]

        for desc_file in desc_files:
            df = pd.concat([df, pd.read_csv(desc_file)]) # 데이터 프레임 df에 csv 파일에서 불러온 data를 합친다. (path : D:\data\EmoDB\03a01Fa.wav ... emotion : happy ...)

        audio_paths = list(df['path']); emotions = list(df['emotion'])

        n_samples = len(audio_paths)
        name = os.path.join(self.features_folder_name, f"{partition}_{n_samples}.npy")

        if os.path.isfile(name):
            features = np.load(name)
        else:
            features = []
            for audio_file in audio_paths:
                feature = extract_feature(audio_file, **self.audio_config) # {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False}
                features.append(feature)

            features = np.array(features)
            np.save(name, features) # features 배열을 name 이라는 이름의 Numpy format의 바이너리 파일로 저장

        if partition == "train":
            self.train_audio_paths = audio_paths # ['D:\\data\\EmoDB\\03a01Fa.wav', 'D:\\data\\EmoDB\\03a01Nc.wav', ...]
            self.train_emotions = emotions # ['happy', 'neutral', 'angry', 'happy', 'neutral', 'angry', ...]
            self.train_features = features # [[-2.32655716e+02  7.83584671e+01 -5.68447971e+00 ...  2.35245144e-03..], [...], ...]

        elif partition == "test":
            self.test_audio_paths = audio_paths
            self.test_emotions = emotions
            self.test_features = features

    # 훈련 데이터셋을 로드해주는 함수
    def load_train_data(self, desc_files, shuffle):
        self.load_metadata(desc_files, "train", shuffle)

    # 테스트 데이터셋을 로드해주는 함수
    def load_test_data(self, desc_files, shuffle):
        self.load_metadata(desc_files, "test", shuffle)

    # 훈련이나 테스트에 사용할 음성 데이터의 수를 각각의 감정이 모두 동일한 수(가장 적은 감정으로 레이블된 음성 데이터의 수)가 되도록 맞춰주는 함수
    def balance_data(self, partition):

        if partition == "train":
            emotions = self.train_emotions
            features = self.train_features
            audio_paths = self.train_audio_paths
        elif partition == "test":
            emotions = self.test_emotions
            features = self.test_features
            audio_paths = self.test_audio_paths
        
        count = []
        for emotion in self.emotions:
            count.append(len([ e for e in emotions if e == emotion])) # [3215, 2957, 2267, 2140] / [146, 177, 107, 130]
        minimum_emotion = min(count)

        dic = defaultdict(list) # defaultdict(<class 'list'>, {})
        counter = {e: 0 for e in self.emotions } # {'angry': 0, 'neutral': 0, 'happy': 0, 'fear': 0}

        for emotion, feature, audio_path in zip(emotions, features, audio_paths):
            if counter[emotion] >= minimum_emotion:
                continue
            counter[emotion] += 1
            dic[emotion].append((feature, audio_path))

        emotions, features, audio_paths = [], [], []
        for emotion, features_audio_paths in dic.items():
            for feature, audio_path in features_audio_paths:
                emotions.append(emotion)
                features.append(feature)
                audio_paths.append(audio_path)

        if partition == "train":
            self.train_emotions = emotions
            self.train_features = features
            self.train_audio_paths = audio_paths
        elif partition == "test":
            self.test_emotions = emotions
            self.test_features = features
            self.test_audio_paths = audio_paths

    # partition별로 특성값, 레이블된 감정값, 저장 경로값을 무작위로 섞어주는 함수
    def shuffle_data_by_partition(self, partition):
        if partition == "train":
            self.train_audio_paths, self.train_emotions, self.train_features = self.shuffle_data(self.train_audio_paths,
            self.train_emotions, self.train_features)
        elif partition == "test":
            self.test_audio_paths, self.test_emotions, self.test_features = self.shuffle_data(self.test_audio_paths,
            self.test_emotions, self.test_features)

    # 음성 데이터의 특성값, 레이블된 감정값, 저장 경로값을 무작위로 섞어주는 함수 -> 각각의 음성 데이터의 (특성값, 레이블된 감정값, 저장 경로값)을 익덱스로 구분하여 해당 인덱스를 섞어준다.
    def shuffle_data(self, audio_paths, emotions, features):
        p = np.random.permutation(len(audio_paths)) # 0이상 len(audio_paths)미만의 숫자를 랜덤하게 섞은 배열 객체를 생성
        return [audio_paths[i] for i in p], [emotions[i] for i in p], [features[i] for i in p]

# 음성에서 특성값을 추출해주는 함수
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc"); chroma = kwargs.get("chroma"); mel = kwargs.get("mel")
    result = np.array([])

    with SoundFile(file_name) as sound_file:
        x = sound_file.read(dtype="float32")
        sample_rate = 16000
        if chroma:
            stft = np.abs(librosa.stft(x))
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, tuning=0.0).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# 음성에서 특성값과 레이블된 감정값 등을 추출하고 추출한 데이터에 접근할 수 있도록 해주는 함수
def load_data(train_dataset, test_dataset, audio_config, shuffle, balance, emotions):

    extractor = DataExtractor(audio_config, emotions, balance, 1)
    extractor.load_train_data(train_dataset, shuffle)
    extractor.load_test_data(test_dataset, shuffle)

    return {
        "x_train": np.array(extractor.train_features),
        "x_test": np.array(extractor.test_features),
        "y_train": np.array(extractor.train_emotions),
        "y_test": np.array(extractor.test_emotions),
        "train_audio_paths": extractor.train_audio_paths,
        "test_audio_paths": extractor.test_audio_paths
    }
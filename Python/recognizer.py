import os
import sys
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from preprocess_data import LoadingData
from extract_data import extract_feature

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SpeechEmotionRecognizer(LoadingData):
    def __init__(self, **kwargs):
    
        super().__init__(**kwargs)
        self.model = None # 모델 종류
        self.model_created = False # 모델 생성 여부
        self.model_name = "rnn_classifierModel.h5" # 생성될 model 이름
        self.epochs = kwargs.get("epochs", 100) # 전체 데이터의 학습 횟수, default = 100
        self.batch_size = kwargs.get("batch_size", 64) # 한번의 epoch마다 나누어 학습할 데이터의 개수, default = 64
        self.int2emotions = {index: emotion for index, emotion in enumerate(self.emotions)} # {0: 'angry', 1: 'neutral', 2: 'happy', 3: 'fear'}
        self.emotions2int = {emotion: index for index, emotion in self.int2emotions.items()} # {'angry': 0, 'neutral': 1, 'happy': 2, 'fear': 3}
        self.compute_input_length()

    # 모델을 만드는 함수
    def create_model(self):
        if self.model_created:
            return
        if not self.data_loaded:
            self.load_data()

        model = Sequential()

        model.add(LSTM(units=1024, return_sequences=True, input_shape=(None, 180)))
        model.add(Dropout(0.3))

        model.add(LSTM(units=1024, return_sequences=True))
        model.add(Dropout(0.3))

        model.add(Dense(units=1024, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(units=4, activation='softmax'))

        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
        self.model_created = True

        if self.verbose > 0:
            print("LSTM 모델 생성 완료")

    # 모델을 훈련하고 테스트할 데이터를 로드하는 함수
    def load_data(self):
        super().load_data()
        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        self.X_train = self.X_train.reshape((1, X_train_shape[0], X_train_shape[1]))
        self.X_test = self.X_test.reshape((1, X_test_shape[0], X_test_shape[1]))

        self.y_train = to_categorical([ self.emotions2int[str(e)] for e in self.y_train ])
        self.y_test = to_categorical([ self.emotions2int[str(e)] for e in self.y_test ])
        
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape
        self.y_train = self.y_train.reshape((1, y_train_shape[0], y_train_shape[1]))
        self.y_test = self.y_test.reshape((1, y_test_shape[0], y_test_shape[1]))

    # 모델을 훈련하는 함수
    def train(self, override=False):
        if not self.model_created:
            self.create_model()

        if not override:
            model_name = self.check_model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        model_filename = self.get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.X_test, self.y_test),
                                      callbacks=[self.checkpointer], verbose=self.verbose)
        self.model_trained = True
        if self.verbose > 0:
            print("LSTM 모델 훈련 완료")

    # 완성된 모델의 이름을 반환하는 함수
    def get_model_filename(self):
        return "results/" + self.model_name

    # 모델이 존재하는지 여부를 확인하는 함수
    def check_model_exists(self):
        filename = self.get_model_filename()
        return filename if os.path.isfile(filename) else None

    # 특성값의 크기를 추출하는 함수
    def compute_input_length(self):
        if not self.data_loaded:
            self.load_data()
        self.input_length = self.X_train[0].shape[1] # 한 음성 데이터에서 추출한 특성값 개수, shape[1]은 2차원 배열에서 열의 개수 출력

    # 훈련된 모델을 이용해 특정 경로에 있는 음성 데이터에 실린 감정을 분석해 반환하는 함수
    def predict(self, audio_path): # 화남
        int2k_emotions = {0: '화남', 1: '중립', 2: '행복', 3: '두려움'}
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        prediction = self.model.predict(feature)
        prediction = np.argmax(np.squeeze(prediction))
        return int2k_emotions[prediction]

    # 훈련된 모델을 이용해 특정 경로에 있는 음성 데이터에 실린 각 감정의 가능성을 반환하는 함수
    def predict_probability(self, audio_path): # {'화남': 0.2916815, '중립': 0.15005872, '행복': 0.27652138, '두려움': 0.2817385}
        emotions = ['화남', '중립', '행복', '두려움']
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        proba = self.model.predict(feature)[0][0]
        result = {}
        for prob, emotion in zip(proba, emotions):
            result[emotion] = prob
        return result

    # 모델의 테스트 데이터셋에 대한 정확도를 반환하는 함수
    def test_score(self):
        y_test = self.y_test[0] # [[0. 1. 0. 0.][] ...]
        y_pred = self.model.predict(self.X_test)[0]
        y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
        y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
        return accuracy_score(y_true=y_test, y_pred=y_pred)

    # 모델의 트레인 데이터셋에 대한 정확도를 반환하는 함수
    def train_score(self):
        y_train = self.y_train[0]
        y_pred = self.model.predict(self.X_train)[0]
        y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
        y_train = [np.argmax(y, out=None, axis=None) for y in y_train]
        return accuracy_score(y_true=y_train, y_pred=y_pred)

from extract_data import load_data
from glob import glob
from os.path import basename
import pandas as pd

class LoadingData:
    def __init__(self, **kwargs):

        self.data_loaded = False # 데이터 로드 여부
        self.model_trained = False # 모델 훈련 여부
        self.emotions = kwargs.get("emotions", ['angry', 'neutral', 'happy', 'fear']) # 구분할 감정 종류, default = ['angry', 'neutral', 'happy', 'fear']
        self.features = kwargs.get("features", ["mfcc", "chroma", "mel"]) # 음성 데이터로부터 추출할 특성 종류, default = ["mfcc", "chroma", "mel"]
        self.audio_config = {'mfcc': True, 'chroma': True, 'mel': True}
        self.emodb = kwargs.get("emodb", True) # Berlin Database of Emotional Speech(Emo-DB) 사용 여부, default = True
        self.tess = kwargs.get("tess", True) # Toronto Emotional Speech Set(TESS) 사용 여부, default = True
        self.ravdess = kwargs.get("ravdess", True) # The Ryerson Audio-Visual Database of Emotional Speech and Song(RAVDESS) 사용 여부, default = True
        self.cremaD = kwargs.get("cremaD", True) # Crowd Sourced Emotional Multimodal Actors Dataset(CREMA-D) 사용 여부, default = True
        self.shEMO = kwargs.get("shEMO", True) # Persian Speech Emotion Detection Database(ShEMO) 사용 여부, default = True
        self.savee = kwargs.get("savee", True) # Surrey Audio-Visual Expressed Emotion(SAVEE) database 사용 여부, default = True
        self.Cafe = kwargs.get("Cafe", True) # The Canadian French Emotional(CaFE) speech dataset 사용 여부, default = True
        self.KoreanEMO = kwargs.get("KoreanEMO", True) # 기타 한국어 dataset 사용 여부, default = True

        self.balance = kwargs.get("balance", True) # 각 감정으로 레이블된 데이터의 수를 동일하게 맞출지 여부, default = True
        self.shuffle = kwargs.get("shuffle", True) # 데이터를 섞을지 여부, default = True
        self.verbose = kwargs.get("verbose", 1) # log 출력 여부, default = 1

        self.emodb_name = "emodb.csv"; self.tess_name = "tess.csv"; self.ravdess_name = "ravdess.csv"
        self.cremaD_name = "cremaD.csv"; self.shEMO_name = "shEMO.csv"; self.savee_name = "savee.csv"
        self.Cafe_name = "Cafe.csv"; self.KoreanEMO_name = "KoreanEMO.csv"

        self.create_csv_filename()
        self.create_csv_file()

    # 각 음성 데이터베이스에 포함된 음성의 저장 경로와 레이블된 감정 종류를 저장할 csv 파일 이름을 생성하는 함수
    def create_csv_filename(self):
        train_dataset = []; test_dataset = []

        if self.emodb:
            train_dataset.append("train_" + self.emodb_name); test_dataset.append("test_" + self.emodb_name)
        if self.tess:
            train_dataset.append("train_" + self.tess_name); test_dataset.append("test_" + self.tess_name)
        if self.ravdess:
            train_dataset.append("train_" + self.ravdess_name); test_dataset.append("test_" + self.ravdess_name)
        if self.cremaD:
            train_dataset.append("train_" + self.cremaD_name); test_dataset.append("test_" + self.cremaD_name)
        if self.shEMO:
            train_dataset.append("train_" + self.shEMO_name); test_dataset.append("test_" + self.shEMO_name)
        if self.savee:
            train_dataset.append("train_" + self.savee_name); test_dataset.append("test_" + self.savee_name)
        if self.Cafe:
            train_dataset.append("train_" + self.Cafe_name); test_dataset.append("test_" + self.Cafe_name)
        if self.KoreanEMO:
            train_dataset.append("train_" + self.KoreanEMO_name); test_dataset.append("test_" + self.KoreanEMO_name)

        self.train_dataset = train_dataset; self.test_dataset = test_dataset

    # csv 파일을 생성하는 함수
    def create_csv_file(self):
        n = len(self.train_dataset)
        for i in range(n):
            train_csv_file, test_csv_file = self.train_dataset[i], self.test_dataset[i]

            if self.emodb_name in train_csv_file:
                self.create_emodb_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("EMO-DB CSV File 생성 완료")
            elif self.tess_name in train_csv_file:
                self.create_tess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("TESS CSV File 생성 완료")
            elif self.ravdess_name in train_csv_file:
                self.create_ravdess_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("RAVDESS CSV File 생성 완료")
            elif self.cremaD_name in train_csv_file:
                self.create_cremaD_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("CREMA-D CSV File 생성 완료")
            elif self.shEMO_name in train_csv_file:
                self.create_shEMO_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("shEMO CSV File 생성 완료")
            elif self.savee_name in train_csv_file:
                self.create_savee_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("SAVEE CSV File 생성 완료")
            elif self.Cafe_name in train_csv_file:
                self.create_Cafe_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("Cafe CSV File 생성 완료")
            elif self.KoreanEMO_name in train_csv_file:
                self.create_KoreanEMO_csv(self.emotions, train_name=train_csv_file, test_name=test_csv_file)
                if self.verbose: print("KoreanEMO CSV File 생성 완료")

    # 생성된 데이터를 로드하는 함수
    def load_data(self):
        if not self.data_loaded:
            result = load_data(self.train_dataset, self.test_dataset, self.audio_config, self.shuffle, self.balance, self.emotions)
            self.X_train = result['x_train']
            self.X_test = result['x_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.train_audio_paths = result['train_audio_paths']
            self.test_audio_paths = result['test_audio_paths']
            self.data_loaded = True

            if self.verbose:
                print("데이터 로드 완료")

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_emodb_csv(self, emotions, train_name, test_name, train_rate=0.95): # train_rate 조정 필요

        emotion_label = {
            "W": "angry", "L": "boredom", "E": "disgust", "A": "fear", "F": "happy", "T": "sad", "N": "neutral"
        }

        target = {"path": [], "emotion": []}
        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\EmoDB\*.wav"):
            try:
                emotion = emotion_label[basename(file)[5]] # 음성이 저장된 경로에서 파일 이름을 가져와 레이블된 감정을 확인
            except:
                continue # 원치 않는 감정으로 레이블된 데이터는 해당 정보가 csv파일에 저장되지 않도록 제외해줌
            target["path"].append(file); target["emotion"].append(emotion)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_tess_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "neutral.wav": "neutral", "happy.wav": "happy", "sad.wav": "sad", "fear.wav": "fear", "angry.wav": "angry"
        }

        target = {"path": [], "emotion": []}
        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\TESS\*.wav"):
            try:
                emotion = emotion_label[basename(file).split("_")[2]]
            except:
                continue
            target['emotion'].append(emotion); target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_ravdess_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust", "08": "surprised"
        }

        target = {"path": [], "emotion": []}

        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\RAVDESS\speech\Actor_*\*.wav"):
            try:
                emotion = emotion_label[basename(file).split("-")[2]]
            except:
                continue
            target['emotion'].append(emotion)
            target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_cremaD_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fear", "DIS": "disgust"
        }

        target = {"path": [], "emotion": []}

        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\CREMA-D\*.wav"):
            try:
                emotion = emotion_label[basename(file).split("_")[2]]
            except:
                continue
            target['emotion'].append(emotion)
            target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_shEMO_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "N": "neutral", "H": "happy", "S": "sad", "A": "angry", "F": "fear", "W": "surprised"
        }

        target = {"path": [], "emotion": []}
        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\ShEMO\*.wav"):
            try:
                emotion = emotion_label[basename(file)[3]]
            except:
                continue
            target['emotion'].append(emotion)
            target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_savee_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "n": "neutral", "h": "happy", "sa": "sad", "a": "angry", "f": "fear", "d": "disgust", "su": "surprised"
        }

        target = {"path": [], "emotion": []}

        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\SAVEE\Actor_*\*.wav"):
            try:
                emotion = emotion_label[basename(file).split("_")[0]]
            except:
                continue
            target['emotion'].append(emotion)
            target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_Cafe_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "N": "neutral", "J": "happy", "P": "fear", "C": "angry"
        }

        target = {"path": [], "emotion": []}
        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\Cafe\*.wav"):
            try:
                emotion = emotion_label[basename(file).split("-")[1]]
            except:
                continue
            target['emotion'].append(emotion)
            target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

    # 음성 데이터의 경로와 레이블된 감정을 속성값으로 갖도록 DataFrame을 만들어 csv파일로 저장해주는 함수
    def create_KoreanEMO_csv(self, emotions, train_name, test_name, train_rate=0.95):

        emotion_label = {
            "neutral.wav": "neutral", "happy.wav": "happy", "fear.wav": "fear", "angry.wav": "angry"
        }

        target = {"path": [], "emotion": []}
        temp = list(emotion_label.items())
        for code, emotion in temp:
            if emotion not in emotions:
                del emotion_label[code]

        for file in glob("D:\data\KoreanEMO\*.wav"):
            try:
                emotion = emotion_label[basename(file).split("_")[1]]
            except:
                continue
            target['emotion'].append(emotion)
            target['path'].append(file)

        train_size = int(len(target["path"]) * train_rate)
        train_path = target["path"][0:train_size]; train_emotion = target["emotion"][0:train_size]
        test_path = target["path"][train_size:]; test_emotion = target["emotion"][train_size:]
        pd.DataFrame({"path": train_path, "emotion": train_emotion}).to_csv(train_name)
        pd.DataFrame({"path": test_path, "emotion": test_emotion}).to_csv(test_name)

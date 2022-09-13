## 구해줘(AI part)
'구해줘' 프로젝트의 음성 감정인식 AI모델 구현 관련 레포지토리입니다. 

음성 감정인식 AI모델 구현과 '구해줘' 프로젝트 적용에는 아래 프로젝트들의 소스코드가 적극 활용되었음을 미리 밝힙니다.
- [Speech Emotion Recognition](https://github.com/x4nth055/emotion-recognition-using-speech)
- [jlibrosa](https://github.com/Subtitle-Synchronizer/jlibrosa)

구해줘는 사용자의 키워드를 인식하고 해당 키워드에 실린 감정에 따라 자동으로 112에 신고 문자를 보내주는 안드로이드 애플리케이션 개발 프로젝트입니다. 프로젝트의 상세 설명과 감정인식 AI모델 구현 이외의 코드는 아래 레포지토리에 수록되어 있습니다.
- [구해줘](https://github.com/haesungJoo/capstone_design)

## 구성 파일 및 설명
- Python 파일

|파일명|설명|
|------|---|
|main.py|음성 감정인식 모델 생성 및 훈련 코드|
|recognizer.py|음성 감정인식 LSTM 모델 생성관련 코드|
|preprocess_data.py|음성 감정인식 모델 훈련에 사용될 음성 데이터 처리관련 코드|
|extract_data.py|모델 훈련에 사용될 음성 데이터에서 특성값을 추출하는 코드|
|check_trained_model.py|훈련된 음성 감정인식 모델의 동작을 확인하는 코드|
|convert_tflite.py| h5 파일을 pb 파일로 변환하는 코드|
|convert_tflite2.py| pb 파일을 tflite 파일로 변환하는 코드|

- Java 파일

|파일명|설명|
|------|---|
|Classifier.java|사용자의 음성에 실린 감정을 분류하는 코드|
|Result.java|사용자의 음성에 실린 감정을 분류하는 코드|
|JLibrosa.java|사용자의 음성 데이터에서 특징값을 추출하는 코드|
|AudioFeatureExtraction.java|사용자의 음성 데이터에서 특징값을 추출하는 코드|
|FileFormatNotSupportedException.java|예외처리 관련 코드|
|WavFileException.java|예외처리 관련 코드|
|WavFile.java|WAV 파일 처리 관련 코드|

## 음성 감정 인식 모델 훈련에 사용된 Dataset
- Emo-DB(Berlin Database of Emotional Speech)
- TESS(Toronto Emotional Speech Set)
- RAVDESS(The Ryerson Audio-Visual Database of Emotional Speech and Song)
- CREMA-D(Crowd Sourced Emotional Multimodal Actors Dataset)
- ShEMO(Persian Speech Emotion Detection Database)
- SAVEE(Surrey Audio-Visual Expressed Emotion) database
- CaFE(The Canadian French Emotional) speech dataset
- 기타 한국어 음성 dataset

## 구해줘 프로젝트에 적용 과정
1. 음성 데이터셋을 PC의 지정된 경로에 저장(preprocess_data.py 파일에서 모델 훈련에 사용할 데이터셋과 저장 경로 수정 가능)
2. main.py 파일에서 모델 생성 및 훈련 -> h5 파일 생성
3. convert_tflite.py, convert_tflite2.py 파일을 통해 h5 파일을 tflite 파일로 변환(h5 모델을 텐서플로우 라이트 모델로 변환)
4. 안드로이드 프로젝트에 assets 폴더를 만들고 변환한 모델을 저장
5. 텐서플로우 라이트 관련 모듈을 사용할 수 있도록 gradle 파일에 관련 내용을 추가
6. Chroma feature 추출 코드를 추가한 JLibrosa 라이브러리를 이용하여 사용자의 음성 데이터에서 특징값(MFCC, MEL Spectrogram Frequency, Chroma feature) 추출
7. TensorFlow Lite Interpreter를 이용하여 tflite 파일을 로딩, Interpreter class의 run 함수(매개변수로 음성 데이터 특징값을 넣어줌)를 이용하여 결과값 반환
8. 반환된 결과값을 바탕으로 사용자 음성에 실린 감정(화남, 중립, 행복, 두려움)을 특정하고 프로젝트에 사용

## 구해줘 프로젝트 적용 시연 영상
- 프로젝트에 사용된 다양한 라이브러리들의 버젼 관련 issue와 서버 실행 관련 issue로 추후(2021년 6월 이후) 구해줘 프로젝트가 정상적으로 동작하지 않을 수 있습니다. 따라서 프로젝트의 동작은 아래 시연 영상을 참고해주시면 감사하겠습니다.
- 해당 영상은 중간에 사이렌 소리가 나오니 적당한 음량으로 조정하여 이어폰으로 들으시기를 권장드립니다.
- [시연 영상 링크](https://youtu.be/YhQe7rPS-oM)

## 한계
- 완성된 음성 감정인식 모델의 정확도는 약 62%입니다. 물론 구해줘 앱은 사용자가 특정 키워드를 말하였을 때만 감정을 분석하고 112에 신고 문자가 보내지더라도 주변 상황이 녹음되어 함께 보내지기에 오작동으로 인해 경찰이 출동하는 일은 최대한 방지할 수 있습니다. 그러나 그럼에도 불구하고 상용화 하기에는 어려운 정확도입니다.

## 개선 방법
- 음성 감정 인식 모델의 정확도를 높이기 위해서는 첫째, 양질의 한국어 음성 감정 데이터가 필요하다고 생각합니다. 인터넷에서 구할 수 있는 데이터는 외국어 음성 감정 데이터 위주이고 그 마저도 수가 적습니다. 만약 다양한 연령대의 발화자가 다양한 감정으로 말하는 한국어 데이터셋이 확보된다면 음성 감정인식 모델의 성능을 높일 수 있다고 생각합니다.
- 둘째, 감정 인식 모델의 정확도가 낮은 이유가 데이터셋이 아닌 LSTM 모델 자체의 성능 문제일 가능성도 있습니다. 따라서 다양한 머신러닝 모델에 대한 테스트를 통해 음성 감정 인식에 더 적합한 모델을 찾는다면 모델의 정확도 문제를 상당부분 개선할 수 있다고 생각합니다.

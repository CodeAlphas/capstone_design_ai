## 구해줘(AI part)
'구해줘' 프로젝트의 음성 감정인식 AI모델 구현 관련 레포지토리입니다. 

음성 감정인식 AI모델 구현과 '구해줘' 프로젝트 적용에는 아래 프로젝트들의 소스코드가 적극 활용되었음을 미리 밝힙니다.
- [Speech Emotion Recognition](https://github.com/x4nth055/emotion-recognition-using-speech)
- [jlibrosa](https://github.com/Subtitle-Synchronizer/jlibrosa)

구해줘는 사용자의 키워드를 인식하고 해당 키워드에 실린 감정에 따라 자동으로 112에 신고 문자를 보내주는 안드로이드 애플리케이션 구현 프로젝트입니다. 프로젝트의 상세 설명과 감정인식 AI모델 구현 이외의 코드는 아래 레포지토리에 수록되어 있습니다.
- [구해줘](https://github.com/haesungJoo/capstone_design)

## 각 구성 파일의 역할
|파일명|설명|
|------|---|
|main.py|음성 감정인식 모델 생성 및 훈련 코드|
|recognizer.py|음성 감정인식 LSTM 모델 생성관련 코드|
|preprocess_data.py|음성 감정인식 모델 훈련에 사용될 음성 데이터 처리관련 코드|
|extract_data.py|모델 훈련에 사용될 음성 데이터에서 특성값을 추출해주는 코드|
|check_trained_model.py|음성 감정인식 모델의 동작을 확인해보는 코드|
|convert_tflite.py|.h5 파일을 .pb 파일로 변환해주는 코드|
|convert_tflite2.py|.pb파일을 .tflite 파일로 변환해주는 코드|

## 음성 감정 인식 모델 훈련에 사용된 Dataset
- Emo-DB(Berlin Database of Emotional Speech)
- TESS(Toronto Emotional Speech Set)
- RAVDESS(The Ryerson Audio-Visual Database of Emotional Speech and Song)
- CREMA-D(Crowd Sourced Emotional Multimodal Actors Dataset)
- ShEMO(Persian Speech Emotion Detection Database)
- SAVEE(Surrey Audio-Visual Expressed Emotion) database
- CaFE(The Canadian French Emotional) speech dataset
- 

## 구해줘 프로젝트에 적용 과정
1. 

## 구해줘 프로젝트 적용 시연 영상
- 영상 중간에 사이렌 소리가 나오니 적당한 음량으로 조절하여 이어폰으로 들으시기를 권장드립니다.

## 한계 및 느낀점
- 

from recognizer import SpeechEmotionRecognizer

lstm_model = SpeechEmotionRecognizer() # 모델 생성
lstm_model.train() # 모델 훈련
print(f"LSTM 모델의 정확도는 {lstm_model.test_score()*100 : 0.3f}% 입니다.") # 훈련이 완료된 모델의 테스트 데이터셋에 대한 정확도 확인
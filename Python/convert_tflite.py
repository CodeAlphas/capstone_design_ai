# 훈련을 마친 h5 확장자 모델을 pb 파일로 변환
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.models.load_model('results/rnn_classifierModel.h5', compile=False)
 
export_path = 'pb_rnn'
model.save(export_path, save_format="tf")
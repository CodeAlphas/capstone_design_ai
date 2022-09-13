# pb 파일을 tflite 파일로 변환
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

saved_model_dir = "pb_rnn"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

with open(r"D:/SpeechEmotion_RNN/speechemotion.tflite", 'wb') as f:
  f.write(tflite_model)
  
print("speechemotion.tflite written")
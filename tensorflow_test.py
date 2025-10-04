import tensorflow as tf

# TensorFlow 버전 출력
print("TensorFlow Version: ", tf.__version__)

# GPU 장치 목록 확인
gpu_list = tf.config.list_physical_devices('GPU')

if gpu_list:
  print("GPU가 성공적으로 인식되었습니다.")
  print(gpu_list)
else:
  print("GPU를 찾을 수 없습니다. CPU 버전을 확인해주세요.")
import os
import numpy as np
from sklearn.mixture import GaussianMixture
import librosa

# 语音文件的目录
audio_folder = '622音频'

# 读取音频文件并提取特征
speakers = []
models = []
                  #列出目录中的文件名
for filename in os.listdir(audio_folder):
    # 文件名不包含扩展名的部分作为说话人名字
    speaker_name = os.path.splitext(filename)[0]
    speakers.append(speaker_name)

    # 加载音频文件
    # y：音频信号的时间序列（时域表示）。
    # sr：音频的采样率。
    file_path = os.path.join(audio_folder, filename)
    y, sr = librosa.load(file_path, sr=1600)  # 使用原始采样率

    # 提取 MFCC 特征
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 特征归一化
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_normalized = (mfcc - mean) / std

    # 使用 GMM 建模
    n_components = 8
    gmm = GaussianMixture(n_components=n_components, max_iter=200, covariance_type='diag', n_init=3)
    gmm.fit(mfcc_normalized.T)

    # 保存模型
    models.append(gmm)

# 将说话人列表和他们的模型保存为.npy文件
np.save('gmm_models.npy', models)
np.save('speakers.npy', speakers)

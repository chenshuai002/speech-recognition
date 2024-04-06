import os
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
from scipy.io.wavfile import write

# 加载先前保存的模型和说话人名字
models = np.load('gmm_models.npy', allow_pickle=True)
speakers = np.load('speakers.npy', allow_pickle=True)

def process_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=1600)

    # 提取 MFCC 特征
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # 特征归一化
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_normalized = (mfcc - mean) / std

    # 使用模型评估音频
    scores = [gmm.score(mfcc_normalized.T) for gmm in models]

    # 找到得分最高的说话人
    best_speaker_index = np.argmax(scores)

    # 输出结果
    result_text = f'对于音频文件 {os.path.basename(file_path)}, 最可能的说话人是 {speakers[best_speaker_index]} ，得分 {scores[best_speaker_index]}'
    result_label.config(text=result_text)

#截取音频文件的文件名传给file_path传给函数process_audio_file进行GMM测试
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_audio_file(file_path)

#录制语言函数 的 功能
def record_audio():
    fs = 16000  # 采样率
    seconds = 5  # 录音时间
    #显示功能和录制音频文件
    messagebox.showinfo("录音", "请录制5秒的音频")
    #语音录制
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    #等待结束
    sd.wait()

    # 保存音频文件
    output_file = "output.wav"
    write(output_file, fs, recording)

    # 处理音频文件，将音频文件传送给GMM模型进行测试
    process_audio_file(output_file)

# 创建交互的主窗口
root = tk.Tk()
root.title("说话人识别系统")

# 添加按钮和标签
open_file_button = tk.Button(root, text="打开音频文件", command=open_file_dialog)
                       #长      宽
open_file_button.pack(padx=20, pady=20)

record_audio_button = tk.Button(root, text="录制音频", command=record_audio)
record_audio_button.pack(padx=20, pady=20)

#设置空白标签  展示测试结果
result_label = tk.Label(root, text="", wraplength=400)
result_label.pack(padx=20, pady=20)

# 启动事件循环
root.mainloop()
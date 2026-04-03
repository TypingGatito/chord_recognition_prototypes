import librosa.display
import matplotlib.pyplot as plt

file_path = input("Введите путь к аудио файлу: ").strip()
length = float(input("Длительность (сек): "))

y, sr = librosa.load(file_path, duration=length)

chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

plt.figure(figsize=(12,4))
librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
plt.colorbar()

plt.savefig("chroma.png", dpi=300)
print("Файл сохранён: chroma.png")
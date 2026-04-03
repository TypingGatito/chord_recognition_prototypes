# Chord Recognition Prototypes

## Описание

Проект реализует распознавание аккордов из аудио с использованием нескольких подходов:

* метрический (template-based)
* библиотечный (librosa / FMP)
* вероятностный HMM
* Gaussian HMM (с обучением на датасете)

Поддерживается оценка качества через метрику **AOS (Average Overlap Score)**.

---

## Зависимости

Установить:

```bash
pip install numpy librosa
```

Дополнительно:

```bash
pip install scipy
```

---

## Структура данных

Ожидается датасет:

```
data_sets/
└── IDMT-SMT-CHORDS/
    └── guitar/
        ├── *.wav
        └── guitar_annotation.lab
```

Формат `.lab`:

```
start_time end_time chord_label
```

Пример:

```
0.0 1.5 C:maj
1.5 3.0 G:maj
```

---

## Способы запуска

### 1. Распознавание файла

```bash
python main.py --file path/to/file.wav --type manual
```

Типы:

* `manual`
* `fmp`
* `hmm`
* `hmm_gauss`

---

### 2. Тест на первых 20 секундах датасета

```bash
python main.py --test --dataset path/to/dataset --type hmm
```

---

### 3. Оценка AOS для одного файла

```bash
python main.py --file path/to/file.wav --aos --type hmm
```

Требуется файл:

```
file.lab
```

---

### 4. Оценка AOS на всём датасете

```bash
python main.py --dataset path/to/dataset --aos --type hmm
```

---

### 5. Использование микрофона

```bash
python main.py --type manual
```

---

## Обучение Gaussian HMM

При выборе:

```bash
--type hmm_gauss
```

происходит:

* автоматическая загрузка всех `.wav` из:

```
data_sets/IDMT-SMT-CHORDS/guitar
```

* обучение модели HMM
* использование обученной модели для распознавания

---

## Метрика AOS

AOS = доля времени, где предсказанный аккорд совпадает с эталоном.

```
AOS = correct_time / total_time
```

---

## Особенности

* используется chroma-признаки (librosa)
* HMM декодирование через Viterbi
* Gaussian emissions (для hmm_gauss)
* нормализация аккордов перед сравнением

---

## Отладка

Включены debug-выводы:

* выбранный метод
* обработка файлов
* промежуточные результаты

---

## Примечания

* hmm_gauss требует достаточного объёма данных для обучения
* один `.lab` файл применяется ко всем `.wav` в датасете (в текущей конфигурации)
* поддерживаются только maj/min аккорды

---

## Датасеты

Можно загрузить по [ссылке]()

### IDMT-SMT-CHORDS

Источник: https://www.idmt.fraunhofer.de/en/publications/datasets/chords.html

Содержит записи отдельных аккордов (гитара, пианино и др.), записанных в контролируемых условиях.
Используется для:

* обучения моделей
* тестирования на чистых сигналах

Особенность:

* фиксированные аккорды, без последовательностей

---

### Rock Dataset

Источник: https://github.com/artteam8/chords-dataset

Содержит реальные музыкальные треки с разметкой аккордов.
Используется для:

* оценки качества (AOS)
* тестирования на реальной музыке

---

### Mine (собственный датасет)

Ручные записи аккордов.

Особенности:

* разметка может быть неточной
* полезен для быстрых экспериментов

---

### Major vs Minor Guitar Chords Dataset

https://www.kaggle.com/datasets/mehanat96/major-vs-minor-guitar-chords

Датасет из изолированных гитарных аккордов, разделённых на два класса: major и minor. Используется для классификации и обучения эмиссий.


## Окружения (venv)

Проект использует **3 отдельных виртуальных окружения**:

### 1. librosa (основное)

Используется для:

* chroma-признаков
* HMM / Gaussian HMM
* основной логики распознавания

Минимальная установка:

```bash
pip install numpy scipy librosa scikit-learn soundfile
```

---

### 2. madmom

Используется для:

* альтернативных методов обработки аудио
* beat / feature extraction

Установка:

```bash
pip install madmom librosa==0.8.1 numpy==1.23.5
```

---

### 3. omnizart

Используется для:

* продвинутых моделей (DL-based)
* экспериментов с готовыми решениями

Установка (важно: старые версии зависимостей):

```bash
pip install omnizart==0.5.0 tensorflow==2.5.0
```

---

## Важно

* окружения **несовместимы между собой** (разные версии numpy/librosa)
* использовать нужно **отдельные venv**
* активировать перед запуском:

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

---

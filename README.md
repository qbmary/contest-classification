# Image Classification Project (PyTorch)

Проект для классификации изображений на PyTorch с поддержкой:
- обучения собственной модели с нуля;
- дообучения предобученной модели (fine-tuning);
- оценки качества модели;
- предсказания класса для одного изображения;
- нескольких форматов датасета через единый `config.py`.

## Что умеет проект

Проект поддерживает два режима обучения:

1. **Обучение с нуля (`scratch`)**  
   Используется простая сверточная сеть `SimpleCNN`.

2. **Дообучение (`finetune`)**  
   Используется предобученная `ResNet18`, у которой заменяется финальный классификатор под нужное число классов.

Также в проекте есть:
- универсальная загрузка датасета;
- автоматическое определение списка классов;
- сохранение лучшей модели;
- оценка по метрикам `accuracy` и `F1`;
- построение `confusion matrix`;
- построение `ROC curve`;

---

## Важно

В репозиторий обычно **не включаются**:
- датасеты (`data/`);
- обученные модели (`models/*.pth`);
- результаты обучения и оценки (`outputs/`);
- локальные окружения (`venv/`, `.venv/`).

После скачивания проекта пользователь должен самостоятельно:
1. установить зависимости;
2. настроить `config.py`;
3. поместить данные в папку `data`;
4. обучить модель;
5. при необходимости выполнить оценку и предсказание.

---

## Структура проекта

Пример актуальной структуры:

```text
classification/
|-- data/
|-- models/
|-- outputs/
|-- config.py
|-- dataset_loader.py
|-- models.py
|-- train_utils.py
|-- train_scratch.py
|-- train_finetune.py
|-- evaluate_model.py
|-- predict_image.py
|-- requirements.txt
|-- README.md
```

### Назначение основных файлов

- `config.py` — все основные настройки проекта: пути, формат датасета, размеры, batch size, количество эпох и т.д.
- `dataset_loader.py` — универсальная загрузка датасета из папок или CSV.
- `models.py` — создание моделей и загрузка модели из чекпоинта.
- `train_utils.py` — функции обучения, валидации, сохранения истории и чекпоинтов.
- `train_scratch.py` — обучение собственной модели с нуля.
- `train_finetune.py` — дообучение предобученной модели.
- `evaluate_model.py` — оценка модели на тестовых данных.
- `predict_image.py` — предсказание класса для одного изображения.
- `requirements.txt` — список зависимостей.

---

## Поддерживаемые форматы датасета

Проект поддерживает несколько вариантов хранения данных. Выбор формата задаётся в `config.py` через переменную:

```python
DATASET_FORMAT = "..."
```

Поддерживаются следующие значения:

### 1. `folder_separate`

Данные уже заранее разделены на `train`, `val`, `test`.

Структура:

```text
data/
|-- train/
│   |-- class_1/
│   |-- class_2/
│   |-- ...
|-- val/
│   |-- class_1/
│   |-- class_2/
│   |-- ...
|-- test/
    |-- class_1/
    |-- class_2/
    |-- ...
```

Если папки `val` нет, валидация может быть собрана автоматически из `train`.

---

### 2. `folder_single`

Все данные лежат в одной папке, разбитой по классам.  
Код сам делит выборку на `train / val / test`.

Структура:

```text
data/
|-- cat/
|-- dog/
```

или

```text
data/
|-- images/
    |-- cat/
    |-- dog/
```

В этом режиме важно правильно указать `FULL_DATA_DIR`.

Пример для структуры:

```text
data/
|-- cat/
|-- dog/
```

нужно поставить:

```python
DATASET_FORMAT = "folder_single"
FULL_DATA_DIR = DATA_DIR
```

---

### 3. `csv_separate`

Данные описываются CSV-файлами `train.csv`, `val.csv`, `test.csv`.

Пример структуры:

```text
data/
|-- images/
|-- train.csv
|-- val.csv
|-- test.csv
```

Пример `train.csv`:

```csv
image_path,label
images/cat_001.jpg,cat
images/dog_001.jpg,dog
```

---

### 4. `csv_single`

Есть один общий CSV-файл, а код сам делит его на train/val/test.

Пример структуры:

```text
data/
|-- images/
|-- annotations.csv
```

Пример `annotations.csv`:

```csv
image_path,label
images/cat_001.jpg,cat
images/dog_001.jpg,dog
images/cat_002.jpg,cat
```

---

## Установка и запуск

### 1. Создать виртуальное окружение

Рекомендуется работать в отдельном окружении.

```bash
python -m venv venv
```

### 2. Активировать окружение

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / Mac:**

```bash
source venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

---

## Зависимости

Пример минимального `requirements.txt`:

```txt
torch
torchvision
torchaudio
numpy
pandas
matplotlib
scikit-learn
pillow
opencv-python
pyqt5
```

---

## Настройка `config.py`

Перед запуском нужно обязательно открыть `config.py` и проверить базовый путь:

```python
BASE_DIR = Path(r"C:\Users\admin\Desktop\classification")
```

Замените его на путь к своей папке проекта.

Далее проверьте основные параметры:

```python
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
```

Если используется режим `folder_single`, проверьте:

```python
DATASET_FORMAT = "folder_single"
FULL_DATA_DIR = DATA_DIR
```

Если используется режим `folder_separate`, проверьте:

```python
DATASET_FORMAT = "folder_separate"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
```

Если используется CSV-режим, проверьте пути:

```python
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
CSV_IMAGES_ROOT = DATA_DIR
IMAGE_COLUMN = "image_path"
LABEL_COLUMN = "label"
```

Также можно настроить:
- `IMG_SIZE`
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `VAL_SPLIT`
- `TEST_SPLIT`
- `SEED`

---

## Как определяется тип модели: scratch или finetune

Тип модели определяется в момент обучения и сохраняется внутри `.pth` файла.

### Если запускать:

```bash
python train_scratch.py
```

то создаётся `SimpleCNN`, а в чекпоинт записывается:

```python
model_type = "scratch"
```

### Если запускать:

```bash
python train_finetune.py
```

то создаётся предобученная `ResNet18`, а в чекпоинт записывается:

```python
model_type = "finetune"
```

При последующем запуске `predict_image.py` или `evaluate_model.py` модель восстанавливается автоматически по полю `model_type`.

---

## Обучение моделей

### Обучение с нуля

```bash
python train_scratch.py
```

Что происходит:
- загружается датасет согласно `config.py`;
- строится `SimpleCNN`;
- запускается цикл обучения;
- лучшая модель по `val_accuracy` сохраняется в папку `models/`.

По умолчанию результат сохраняется в:

```text
models/classifier_scratch.pth
```

Также история обучения сохраняется в:

```text
outputs/scratch_training_history.json
```

---

### Дообучение предобученной модели

```bash
python train_finetune.py
```

Что происходит:
- загружается датасет согласно `config.py`;
- берётся предобученная `ResNet18`;
- последний слой заменяется под число классов;
- выполняется fine-tuning;
- лучшая модель сохраняется в `models/`.

По умолчанию результат сохраняется в:

```text
models/classifier_finetune.pth
```

Также история обучения сохраняется в:

```text
outputs/finetune_training_history.json
```

---

## Оценка модели

Открыть `evaluate_model.py` и выбрать нужный файл модели:

```python
MODEL_FILENAME = config.FINETUNE_MODEL_NAME
```

или

```python
MODEL_FILENAME = config.SCRATCH_MODEL_NAME
```

После этого выполнить:

```bash
python evaluate_model.py
```

Скрипт:
- загружает модель из чекпоинта;
- автоматически определяет архитектуру;
- загружает тестовую выборку;
- считает метрики;
- строит графики.

Результаты сохраняются в папке:

```text
outputs/evaluation/
```

Обычно там появляются:
- `*_confusion_matrix.png`
- `*_roc_curve.png`
- `*_metrics.json`

### Вычисляемые метрики

- `Accuracy`
- `F1 (weighted)`
- `Confusion Matrix`
- `ROC Curve`
- `ROC AUC`

### Важно про ROC

Если тестовый набор слишком маленький или в нём присутствует только один класс, `ROC AUC` может не определяться. Это не обязательно ошибка кода — это ограничение самих данных.

---

## Предсказание для одного изображения

Для запуска:

```bash
python predict_image.py
```

Перед запуском нужно открыть файл `predict_image.py` и указать:

1. путь к изображению:

```python
IMAGE_PATH = Path(r"C:\путь\к\изображению.jpg")
```

2. модель, которую нужно использовать:

```python
MODEL_PATH = config.MODELS_DIR / config.FINETUNE_MODEL_NAME
```

или

```python
MODEL_PATH = config.MODELS_DIR / config.SCRATCH_MODEL_NAME
```

Скрипт автоматически:
- загрузит чекпоинт;
- определит тип модели;
- восстановит архитектуру;
- выполнит предсказание;
- выведет:
  - имя изображения;
  - предсказанный класс;
  - уверенность модели (`confidence`).

---

## Пример рабочего сценария

### Вариант 1. Датасет лежит прямо в `data/` по папкам классов

Структура:

```text
data/
├── cat/
└── dog/
```

Тогда в `config.py` нужно указать:

```python
DATASET_FORMAT = "folder_single"
FULL_DATA_DIR = DATA_DIR
```

Далее можно запускать:

```bash
python train_scratch.py
python train_finetune.py
python evaluate_model.py
python predict_image.py
```

---

### Вариант 2. Датасет уже разделён на train/val/test

Структура:

```text
data/
├── train/
├── val/
└── test/
```

Тогда в `config.py` нужно указать:

```python
DATASET_FORMAT = "folder_separate"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
```

---

### Вариант 3. Данные через CSV

В `config.py` нужно выбрать:
- `csv_separate`
- или `csv_single`

и проверить, что правильно указаны:
- путь к CSV;
- колонка с путём до изображения;
- колонка с меткой;
- корневая папка изображений.

---

## Что хранится в чекпоинте модели

В `.pth` файле сохраняется не только `state_dict`, но и служебная информация:
- `model_state_dict`
- `class_names`
- `img_size`
- `model_type`

Благодаря этому `evaluate_model.py` и `predict_image.py` могут автоматически восстановить нужную архитектуру без ручного переписывания кода.

---

## Что делать, если возникают ошибки

### Ошибка `TRAIN_DIR not found`

Значит, выбран режим `folder_separate`, но по указанному пути нет папки `train`.

Проверь:
- `DATASET_FORMAT`
- `TRAIN_DIR`

---

### Ошибка `FULL_DATA_DIR not found`

Значит, выбран режим `folder_single`, но путь к общей папке с классами указан неверно.

Пример:
если данные лежат так:

```text
data/
├── cat/
└── dog/
```

то должно быть:

```python
FULL_DATA_DIR = DATA_DIR
```

а не `DATA_DIR / "images"`.

---

### Ошибка загрузки старой модели

Если старая модель была сохранена в другом формате, может не совпасть поле `model_type`. В таком случае нужно либо переобучить модель новым кодом, либо адаптировать загрузчик под старый формат чекпоинта.

---

## Быстрый старт

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

### 2. Настроить `config.py`

Пример для структуры:

```text
data/
├── cat/
└── dog/
```

в `config.py`:

```python
DATASET_FORMAT = "folder_single"
FULL_DATA_DIR = DATA_DIR
```

### 3. Обучить модель

```bash
python train_finetune.py
```

### 4. Оценить модель

```bash
python evaluate_model.py
```

### 5. Проверить одно изображение

```bash
python predict_image.py
```

---

## Возможные улучшения проекта

В дальнейшем в проект можно добавить:
- более сильные аугментации;
- дополнительные архитектуры моделей;
- сохранение графиков обучения;
- интерфейс для выбора модели через GUI;
- инференс по видеопотоку;
- экспорт лучших параметров обучения.

---

## Автор

Проект подготовлен как учебный/конкурсный шаблон для задач классификации изображений на PyTorch.

# Image Classification Project (PyTorch + PyQt5)

Проект для задачи классификации изображений с обучением моделей, оценкой качества и запуском распознавания в видеопотоке.

## Важно

В репозиторий не включены:
- обученные модели (`models/*.pth`);
- датасеты (`data/`);
- результаты оценки (`outputs/`).

После скачивания проекта пользователь должен самостоятельно:
1. поместить данные в папку `data`;
2. обучить модель через `train_scratch.py` или `train_finetune.py`;
3. после этого запускать `evaluate_model.py` и `video_window.py`.

---

### Создать виртуальное окружение (рекомендуется)

```bash
python -m venv venv
```

Активировать:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / Mac:**

```bash
source venv/bin/activate
```

---

### Установить зависимости

```bash
pip install -r requirements.txt
```

---

## Настройка путей

Открыть файл `config.py` и проверить:

```python
BASE_DIR = Path(r"C:\Users\admin\Desktop\classification")
```

Заменить на путь к своей папке проекта.

---

## Подготовка данных

Структура данных:

```text
data
|--- train
│   |--- class_1
│   |--- class_2
│   |--- ...
|--- val (необязательно)
|--- test
    |--- class_1
    |--- class_2
    |--- ...
```

Если папки `val` нет — она создаётся автоматически.

---

## Обучение моделей

### Обучение с нуля

```bash
python train_scratch.py
```

Результат:

* модель сохраняется в `models/classifier_scratch.pth`

---

### Дообучение

```bash
python train_finetune.py
```

Результат:

* модель сохраняется в `models/classifier_finetune.pth`

---

## Проверка одной картинки

### Для модели с нуля

```bash
python predict_image.py
```

### Для дообученной модели

```bash
python predict_image_finetune.py
```

Перед запуском указать путь к изображению внутри файла
(обычно это C:\Users\admin\Desktop\classification\samples\картинка):

```python
IMAGE_PATH = Path("путь_к_картинке")
```

---

## Оценка модели

Открыть `evaluate_model.py` и выбрать модель:

```python
MODEL_FILENAME = "classifier_finetune.pth"
```

Запустить:

```bash
python evaluate_model.py
```

Результат:

Папка:

```text
outputs/evaluation
```

Содержит:

* confusion matrix (PNG)
* ROC curve (PNG)
* metrics.json

---

## Запуск приложения (камера)

```bash
python video_window.py
```

Дальше в окне:

1. Выбрать камеру
2. Выбрать модель
3. Нажать **Запуск распознавания**

---

## Быстрый старт (коротко)

```bash
pip install -r requirements.txt
python train_finetune.py
python evaluate_model.py
python video_window.py
```

---

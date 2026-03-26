import random
from pathlib import Path

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets


def set_seed(seed: int):
    random.seed(seed)


class CSVDataset(Dataset):
    def __init__(self, dataframe, images_root, class_to_idx, transform=None,
                 image_column="image_path", label_column="label"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_root = Path(images_root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        self.classes = list(class_to_idx.keys())

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = Path(row[self.image_column])
        if not image_path.is_absolute():
            image_path = self.images_root / image_path

        label_value = row[self.label_column]

        if isinstance(label_value, str):
            label = self.class_to_idx[label_value]
        else:
            label = int(label_value)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def split_indices(dataset_len, val_split=0.2, test_split=0.1, seed=42):
    indices = list(range(dataset_len))
    rng = random.Random(seed)
    rng.shuffle(indices)

    test_size = int(dataset_len * test_split)
    val_size = int(dataset_len * val_split)
    train_size = dataset_len - val_size - test_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices


def get_class_names_from_dataframe(df, label_column, class_names=None):
    if class_names is not None:
        return list(class_names)

    labels = df[label_column].tolist()

    if len(labels) == 0:
        return []

    if all(isinstance(x, (int, float)) for x in labels):
        unique_labels = sorted({int(x) for x in labels})
        return [str(x) for x in unique_labels]

    unique_labels = sorted({str(x) for x in labels})
    return unique_labels


def make_class_to_idx(class_names):
    return {name: idx for idx, name in enumerate(class_names)}


def load_datasets(config, train_transform, eval_transform):
    dataset_format = config.DATASET_FORMAT

    if dataset_format == "folder_separate":
        return load_folder_separate(config, train_transform, eval_transform)

    if dataset_format == "folder_single":
        return load_folder_single(config, train_transform, eval_transform)

    if dataset_format == "csv_separate":
        return load_csv_separate(config, train_transform, eval_transform)

    if dataset_format == "csv_single":
        return load_csv_single(config, train_transform, eval_transform)

    raise ValueError(f"Неизвестный DATASET_FORMAT: {dataset_format}")


def load_folder_separate(config, train_transform, eval_transform):
    if not config.TRAIN_DIR.exists():
        raise FileNotFoundError(f"Не найдена TRAIN_DIR: {config.TRAIN_DIR}")

    train_dataset = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
    class_names = train_dataset.classes

    val_dataset = None
    test_dataset = None

    if config.VAL_DIR.exists():
        val_dataset = datasets.ImageFolder(config.VAL_DIR, transform=eval_transform)
        if val_dataset.classes != class_names:
            raise ValueError("Классы в val не совпадают с train.")

    if config.TEST_DIR.exists():
        test_dataset = datasets.ImageFolder(config.TEST_DIR, transform=eval_transform)
        if test_dataset.classes != class_names:
            raise ValueError("Классы в test не совпадают с train.")

    if val_dataset is None:
        full_train_for_split = datasets.ImageFolder(config.TRAIN_DIR, transform=train_transform)
        full_eval_for_split = datasets.ImageFolder(config.TRAIN_DIR, transform=eval_transform)

        train_indices, val_indices, _ = split_indices(
            len(full_train_for_split),
            val_split=config.VAL_SPLIT,
            test_split=0.0,
            seed=config.SEED
        )

        train_dataset = Subset(full_train_for_split, train_indices)
        val_dataset = Subset(full_eval_for_split, val_indices)

    return train_dataset, val_dataset, test_dataset, class_names


def load_folder_single(config, train_transform, eval_transform):
    if not config.FULL_DATA_DIR.exists():
        raise FileNotFoundError(f"Не найдена FULL_DATA_DIR: {config.FULL_DATA_DIR}")

    full_train_dataset = datasets.ImageFolder(config.FULL_DATA_DIR, transform=train_transform)
    full_eval_dataset = datasets.ImageFolder(config.FULL_DATA_DIR, transform=eval_transform)

    class_names = full_train_dataset.classes

    train_indices, val_indices, test_indices = split_indices(
        len(full_train_dataset),
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=config.SEED
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_eval_dataset, val_indices)
    test_dataset = Subset(full_eval_dataset, test_indices) if len(test_indices) > 0 else None

    return train_dataset, val_dataset, test_dataset, class_names


def load_csv_separate(config, train_transform, eval_transform):
    if not config.TRAIN_CSV.exists():
        raise FileNotFoundError(f"Не найден TRAIN_CSV: {config.TRAIN_CSV}")

    train_df = pd.read_csv(config.TRAIN_CSV)
    class_names = get_class_names_from_dataframe(
        train_df,
        config.LABEL_COLUMN,
        config.CLASS_NAMES
    )
    class_to_idx = make_class_to_idx(class_names)

    train_dataset = CSVDataset(
        train_df,
        config.CSV_IMAGES_ROOT,
        class_to_idx,
        transform=train_transform,
        image_column=config.IMAGE_COLUMN,
        label_column=config.LABEL_COLUMN,
    )

    val_dataset = None
    test_dataset = None

    if config.VAL_CSV.exists():
        val_df = pd.read_csv(config.VAL_CSV)
        val_dataset = CSVDataset(
            val_df,
            config.CSV_IMAGES_ROOT,
            class_to_idx,
            transform=eval_transform,
            image_column=config.IMAGE_COLUMN,
            label_column=config.LABEL_COLUMN,
        )

    if config.TEST_CSV.exists():
        test_df = pd.read_csv(config.TEST_CSV)
        test_dataset = CSVDataset(
            test_df,
            config.CSV_IMAGES_ROOT,
            class_to_idx,
            transform=eval_transform,
            image_column=config.IMAGE_COLUMN,
            label_column=config.LABEL_COLUMN,
        )

    if val_dataset is None:
        full_train_dataset = CSVDataset(
            train_df,
            config.CSV_IMAGES_ROOT,
            class_to_idx,
            transform=train_transform,
            image_column=config.IMAGE_COLUMN,
            label_column=config.LABEL_COLUMN,
        )
        full_eval_dataset = CSVDataset(
            train_df,
            config.CSV_IMAGES_ROOT,
            class_to_idx,
            transform=eval_transform,
            image_column=config.IMAGE_COLUMN,
            label_column=config.LABEL_COLUMN,
        )

        train_indices, val_indices, _ = split_indices(
            len(full_train_dataset),
            val_split=config.VAL_SPLIT,
            test_split=0.0,
            seed=config.SEED
        )

        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_eval_dataset, val_indices)

    return train_dataset, val_dataset, test_dataset, class_names


def load_csv_single(config, train_transform, eval_transform):
    if not config.ANNOTATIONS_CSV.exists():
        raise FileNotFoundError(f"Не найден ANNOTATIONS_CSV: {config.ANNOTATIONS_CSV}")

    df = pd.read_csv(config.ANNOTATIONS_CSV)

    class_names = get_class_names_from_dataframe(
        df,
        config.LABEL_COLUMN,
        config.CLASS_NAMES
    )
    class_to_idx = make_class_to_idx(class_names)

    full_train_dataset = CSVDataset(
        df,
        config.CSV_IMAGES_ROOT,
        class_to_idx,
        transform=train_transform,
        image_column=config.IMAGE_COLUMN,
        label_column=config.LABEL_COLUMN,
    )

    full_eval_dataset = CSVDataset(
        df,
        config.CSV_IMAGES_ROOT,
        class_to_idx,
        transform=eval_transform,
        image_column=config.IMAGE_COLUMN,
        label_column=config.LABEL_COLUMN,
    )

    train_indices, val_indices, test_indices = split_indices(
        len(full_train_dataset),
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=config.SEED
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_eval_dataset, val_indices)
    test_dataset = Subset(full_eval_dataset, test_indices) if len(test_indices) > 0 else None

    return train_dataset, val_dataset, test_dataset, class_names


def create_dataloaders(config, train_dataset, val_dataset, test_dataset=None):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

    return train_loader, val_loader, test_loader
"""
Treniranje modela za klasifikaciju bolesti rize.

Skripta obuhvata pripremu podataka, treniranje pocetnog CNN modela,
modela sa augmentacijom i EfficientNetB0 modela, kao i fine-tuning
i cuvanje finalnog modela.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Treniranje modela za klasifikaciju bolesti rize."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Putanja do raspakovanog dataseta."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Folder za cuvanje finalnog modela."
    )

    return parser.parse_args()


def set_random_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_data(data_path):
    train_csv_path = data_path / "train.csv"
    train_images_path = data_path / "train_images"

    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"train.csv nije pronadjen: {train_csv_path}"
        )

    if not train_images_path.exists():
        raise FileNotFoundError(
            f"train_images folder nije pronadjen: {train_images_path}"
        )

    train_df = pd.read_csv(train_csv_path)

    required_columns = {
        "image_id",
        "label",
        "variety",
        "age"
    }

    missing_columns = (
        required_columns - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Nedostaju kolone: {sorted(missing_columns)}"
        )

    print("\nDataset je ucitan.")
    print(f"Broj slika: {len(train_df)}")
    print(f"Broj klasa: {train_df['label'].nunique()}")

    return train_df, train_images_path


def split_data(train_df):
    train_data, temp_data = train_test_split(
        train_df,
        test_size=0.30,
        random_state=SEED,
        stratify=train_df["label"]
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_data["label"]
    )

    print("\nPodjela podataka:")
    print(f"Train: {len(train_data)}")
    print(f"Validacija: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    return train_data, val_data, test_data


def create_class_mapping(train_df):
    class_names = sorted(train_df["label"].unique())

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    print("\nKlase:")
    for class_name, index in class_to_index.items():
        print(f"{index}: {class_name}")

    return class_names, class_to_index


def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize_with_pad(
        image,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1]
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(
    dataframe,
    train_images_path,
    class_to_index,
    training=False
):
    image_paths = [
        str(
            train_images_path
            / row["label"]
            / row["image_id"]
        )
        for _, row in dataframe.iterrows()
    ]

    labels = [
        class_to_index[label]
        for label in dataframe["label"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_baseline_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_baseline_model(
    model,
    train_dataset,
    val_dataset
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stopping]
    )

    best_epoch = np.argmin(
        history.history["val_loss"]
    )

    print("\nRezultat pocetnog CNN modela:")
    print(f"Najbolja epoha: {best_epoch + 1}")
    print(
        "Validation accuracy: "
        f"{history.history['val_accuracy'][best_epoch] * 100:.2f}%"
    )
    print(
        "Validation loss: "
        f"{history.history['val_loss'][best_epoch]:.4f}"
    )

    return history


def main():
    args = parse_arguments()

    set_random_seed()

    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)

    print("=" * 50)
    print("PADDY DISEASE CLASSIFICATION")
    print("=" * 50)

    print(f"Dataset: {data_path}")
    print(f"Output folder: {output_path}")
    print(f"Velicina slike: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    train_df, train_images_path = load_data(
        data_path
    )

    train_data, val_data, test_data = split_data(
        train_df
    )

    class_names, class_to_index = create_class_mapping(
        train_df
    )

    train_dataset = create_dataset(
        train_data,
        train_images_path,
        class_to_index,
        training=True
    )

    val_dataset = create_dataset(
        val_data,
        train_images_path,
        class_to_index
    )

    test_dataset = create_dataset(
        test_data,
        train_images_path,
        class_to_index
    )

    print("\nSkupovi podataka su pripremljeni.")

    baseline_model = create_baseline_model(len(class_names))

    print("\nPocetni CNN model:")
    baseline_model.summary()

    history_baseline = train_baseline_model(
        baseline_model,
        train_dataset,
        val_dataset
    )


if __name__ == "__main__":
    main()

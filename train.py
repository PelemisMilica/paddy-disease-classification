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
import matplotlib.pyplot as plt
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


def create_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip(
            "horizontal",
            seed=SEED
        ),
        layers.RandomBrightness(
            0.12,
            value_range=(0.0, 1.0),
            seed=SEED
        ),
        layers.RandomContrast(
            0.15,
            seed=SEED
        )
    ])


def create_augmented_model(
    num_classes,
    data_augmentation
):
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),

        data_augmentation,

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


def train_augmented_model(
    model,
    train_dataset,
    val_dataset
):
    early_stopping_aug = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_augmented = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[early_stopping_aug]
    )

    early_stopping_aug_continue = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_augmented_continue = model.fit(
        train_dataset,
        validation_data=val_dataset,
        initial_epoch=20,
        epochs=35,
        callbacks=[early_stopping_aug_continue]
    )

    return history_augmented, history_augmented_continue


def train_baseline_35(
    model,
    train_dataset,
    val_dataset
):
    early_stopping_baseline_35 = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_baseline_35 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=35,
        callbacks=[early_stopping_baseline_35]
    )

    return history_baseline_35


def compare_cnn_models(
    history_baseline_35,
    history_augmented,
    history_augmented_continue
):
    aug_accuracy = (
        history_augmented.history["accuracy"] +
        history_augmented_continue.history["accuracy"]
    )

    aug_val_accuracy = (
        history_augmented.history["val_accuracy"] +
        history_augmented_continue.history["val_accuracy"]
    )

    aug_loss = (
        history_augmented.history["loss"] +
        history_augmented_continue.history["loss"]
    )

    aug_val_loss = (
        history_augmented.history["val_loss"] +
        history_augmented_continue.history["val_loss"]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_baseline_35.history["val_accuracy"],
        label="CNN bez augmentacije"
    )

    plt.plot(
        aug_val_accuracy,
        label="CNN sa augmentacijom"
    )

    plt.xlabel("Epoha")
    plt.ylabel("Validation accuracy")
    plt.title("Poredjenje validacione tacnosti")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_baseline_35.history["val_loss"],
        label="CNN bez augmentacije"
    )

    plt.plot(
        aug_val_loss,
        label="CNN sa augmentacijom"
    )

    plt.xlabel("Epoha")
    plt.ylabel("Validation loss")
    plt.title("Poredjenje validacione greske")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    best_baseline_epoch = np.argmin(
        history_baseline_35.history["val_loss"]
    )

    best_aug_epoch = np.argmin(
        aug_val_loss
    )

    print("\nCNN bez augmentacije")
    print(f"Najbolja epoha: {best_baseline_epoch + 1}")
    print(
        "Validation accuracy: "
        f"{history_baseline_35.history['val_accuracy'][best_baseline_epoch] * 100:.2f}%"
    )
    print(
        "Validation loss: "
        f"{history_baseline_35.history['val_loss'][best_baseline_epoch]:.4f}"
    )

    print("\nCNN sa augmentacijom")
    print(f"Najbolja epoha: {best_aug_epoch + 1}")
    print(
        "Validation accuracy: "
        f"{aug_val_accuracy[best_aug_epoch] * 100:.2f}%"
    )
    print(
        "Validation loss: "
        f"{aug_val_loss[best_aug_epoch]:.4f}"
    )


def create_effnet_datasets(
    train_dataset,
    val_dataset,
    test_dataset
):
    train_dataset_effnet = train_dataset.map(
        lambda images, labels: (images * 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_dataset_effnet = val_dataset.map(
        lambda images, labels: (images * 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_dataset_effnet = test_dataset.map(
        lambda images, labels: (images * 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return (
        train_dataset_effnet,
        val_dataset_effnet,
        test_dataset_effnet
    )


def create_effnet_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    data_augmentation_effnet = tf.keras.Sequential([
        layers.RandomFlip(
            "horizontal",
            seed=SEED
        ),
        layers.RandomBrightness(
            0.12,
            value_range=(0.0, 255.0),
            seed=SEED
        ),
        layers.RandomContrast(
            0.15,
            seed=SEED
        )
    ])

    inputs = layers.Input(shape=(224, 224, 3))

    x = data_augmentation_effnet(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    effnet_model = tf.keras.Model(
        inputs,
        outputs
    )

    return effnet_model, base_model


def train_effnet_model(
    model,
    train_dataset_effnet,
    val_dataset_effnet
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping_effnet = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_effnet = model.fit(
        train_dataset_effnet,
        validation_data=val_dataset_effnet,
        epochs=15,
        callbacks=[early_stopping_effnet]
    )

    early_stopping_effnet_continue = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_effnet_continue = model.fit(
        train_dataset_effnet,
        validation_data=val_dataset_effnet,
        initial_epoch=15,
        epochs=25,
        callbacks=[early_stopping_effnet_continue]
    )

    early_stopping_effnet_continue2 = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history_effnet_continue2 = model.fit(
        train_dataset_effnet,
        validation_data=val_dataset_effnet,
        initial_epoch=25,
        epochs=35,
        callbacks=[early_stopping_effnet_continue2]
    )

    return (
        history_effnet,
        history_effnet_continue,
        history_effnet_continue2
    )


def show_effnet_results(
    history_effnet,
    history_effnet_continue,
    history_effnet_continue2
):
    effnet_accuracy = (
        history_effnet.history["accuracy"] +
        history_effnet_continue.history["accuracy"] +
        history_effnet_continue2.history["accuracy"]
    )

    effnet_val_accuracy = (
        history_effnet.history["val_accuracy"] +
        history_effnet_continue.history["val_accuracy"] +
        history_effnet_continue2.history["val_accuracy"]
    )

    effnet_loss = (
        history_effnet.history["loss"] +
        history_effnet_continue.history["loss"] +
        history_effnet_continue2.history["loss"]
    )

    effnet_val_loss = (
        history_effnet.history["val_loss"] +
        history_effnet_continue.history["val_loss"] +
        history_effnet_continue2.history["val_loss"]
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        effnet_accuracy,
        label="Training accuracy"
    )
    plt.plot(
        effnet_val_accuracy,
        label="Validation accuracy"
    )

    plt.xlabel("Epoha")
    plt.ylabel("Accuracy")
    plt.title("Tacnost EfficientNetB0 modela")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(
        effnet_loss,
        label="Training loss"
    )
    plt.plot(
        effnet_val_loss,
        label="Validation loss"
    )

    plt.xlabel("Epoha")
    plt.ylabel("Loss")
    plt.title("Greska EfficientNetB0 modela")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    best_effnet_epoch = np.argmin(
        effnet_val_loss
    )

    print(
        f"Najbolja epoha: "
        f"{best_effnet_epoch + 1}"
    )
    print(
        "Validation accuracy: "
        f"{effnet_val_accuracy[best_effnet_epoch] * 100:.2f}%"
    )
    print(
        "Validation loss: "
        f"{effnet_val_loss[best_effnet_epoch]:.4f}"
    )


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

    data_augmentation = create_data_augmentation()

    augmented_model = create_augmented_model(
        len(class_names),
        data_augmentation
    )

    print("\nCNN model sa augmentacijom:")
    augmented_model.summary()

    history_augmented, history_augmented_continue = train_augmented_model(
        augmented_model,
        train_dataset,
        val_dataset
    )

    baseline_model_35 = create_baseline_model(
        len(class_names)
    )

    print("\nCNN bez augmentacije - trening do 35 epoha:")
    baseline_model_35.summary()

    history_baseline_35 = train_baseline_35(
        baseline_model_35,
        train_dataset,
        val_dataset
    )

    compare_cnn_models(
        history_baseline_35,
        history_augmented,
        history_augmented_continue
    )

    train_dataset_effnet, val_dataset_effnet, test_dataset_effnet = (
        create_effnet_datasets(
            train_dataset,
            val_dataset,
            test_dataset
        )
    )

    effnet_model, base_model = create_effnet_model(
        len(class_names)
    )

    print("\nEfficientNetB0 model:")
    effnet_model.summary()

    history_effnet, history_effnet_continue, history_effnet_continue2 = (
        train_effnet_model(
            effnet_model,
            train_dataset_effnet,
            val_dataset_effnet
        )
    )

    show_effnet_results(
        history_effnet,
        history_effnet_continue,
        history_effnet_continue2
    )


if __name__ == "__main__":
    main()

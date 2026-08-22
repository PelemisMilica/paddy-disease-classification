"""Evaluacija finalnog EfficientNetB0 modela."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers

from train import (
    SEED,
    create_class_mapping,
    create_dataset,
    create_effnet_datasets,
    load_data,
    set_random_seed,
    split_data
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "models/paddy_disease_effnet_final.keras"
        )
    )

    return parser.parse_args()


def prepare_data(data_dir):
    train_df, train_images_path = load_data(
        data_dir
    )

    train_data, val_data, test_data = split_data(
        train_df
    )

    class_names, class_to_index = create_class_mapping(
        train_df
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

    _, val_dataset_effnet, test_dataset_effnet = (
        create_effnet_datasets(
            val_dataset,
            val_dataset,
            test_dataset
        )
    )

    return (
        val_data,
        test_data,
        class_names,
        val_dataset_effnet,
        test_dataset_effnet
    )


def plot_confusion_matrix(
    cm,
    class_names,
    title
):
    plt.figure(figsize=(11, 9))

    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predviđena klasa")
    plt.ylabel("Stvarna klasa")

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90
    )

    plt.yticks(
        range(len(class_names)),
        class_names
    )

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


def evaluate_validation(
    model,
    val_dataset_effnet,
    class_names
):
    val_loss, val_accuracy = model.evaluate(
        val_dataset_effnet,
        verbose=1
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )
    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )

    y_true = np.concatenate([
        labels.numpy()
        for _, labels in val_dataset_effnet
    ])

    predictions = model.predict(
        val_dataset_effnet
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=3
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plot_confusion_matrix(
        cm,
        class_names,
        "Matrica konfuzije"
    )

    return y_true, y_pred, predictions


def analyze_errors(
    val_data,
    y_true,
    y_pred,
    predictions,
    class_names,
    data_dir
):
    val_results = (
        val_data
        .reset_index(drop=True)
        .copy()
    )

    val_results["true_class"] = [
        class_names[i]
        for i in y_true
    ]

    val_results["predicted_class"] = [
        class_names[i]
        for i in y_pred
    ]

    val_results["confidence"] = np.max(
        predictions,
        axis=1
    )

    errors_df = val_results[
        val_results["true_class"] !=
        val_results["predicted_class"]
    ].copy()

    print(
        "Ukupan broj slika:",
        len(val_results)
    )
    print(
        "Broj pogrešnih predikcija:",
        len(errors_df)
    )
    print(
        "Procenat pogrešnih predikcija:",
        f"{len(errors_df) / len(val_results) * 100:.2f}%"
    )

    error_pairs = (
        errors_df
        .groupby(
            ["true_class", "predicted_class"]
        )
        .size()
        .sort_values(ascending=False)
    )

    print(error_pairs.head(15))

    sample_errors = errors_df.sort_values(
        "confidence",
        ascending=False
    ).head(12)

    plt.figure(figsize=(15, 12))

    for i, (_, row) in enumerate(
        sample_errors.iterrows()
    ):
        image_path = (
            data_dir
            / "train_images"
            / row["label"]
            / row["image_id"]
        )

        image = Image.open(image_path)

        plt.subplot(3, 4, i + 1)
        plt.imshow(image)

        plt.title(
            f"Stvarno: {row['true_class']}\n"
            f"Predikcija: {row['predicted_class']}\n"
            f"Sigurnost: {row['confidence']:.1%}",
            fontsize=9
        )

        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return errors_df


def get_model_parts(model):
    data_augmentation = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Sequential)
    )

    base_model = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and any(
            base_layer.name == "top_conv"
            for base_layer in layer.layers
        )
    )

    return data_augmentation, base_model


def make_gradcam_heatmap(
    image,
    model,
    base_model,
    data_augmentation
):
    last_conv_layer = base_model.get_layer(
        "top_conv"
    )

    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    gap_layer = next(
        layer
        for layer in model.layers
        if isinstance(
            layer,
            layers.GlobalAveragePooling2D
        )
    )

    dropout_layer = next(
        layer
        for layer in model.layers
        if isinstance(layer, layers.Dropout)
    )

    classifier_layer = model.layers[-1]

    image = tf.expand_dims(
        image,
        axis=0
    )

    with tf.GradientTape() as tape:
        image_aug = data_augmentation(
            image,
            training=False
        )

        conv_output, features = feature_model(
            image_aug,
            training=False
        )

        x = gap_layer(features)
        x = dropout_layer(
            x,
            training=False
        )

        predictions_grad = classifier_layer(x)

        predicted_class = tf.argmax(
            predictions_grad[0]
        )

        class_score = predictions_grad[
            :, predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    heatmap = tf.reduce_sum(
        conv_output[0] * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


def load_image_for_gradcam(image_path):
    image = tf.io.read_file(
        str(image_path)
    )

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize_with_pad(
        image,
        224,
        224
    )

    return tf.cast(
        image,
        tf.float32
    )


def show_gradcam(
    errors_df,
    model,
    data_dir
):
    selected_errors = pd.concat([
        errors_df[
            (errors_df["true_class"] == "blast") &
            (
                errors_df["predicted_class"] ==
                "brown_spot"
            )
        ].nlargest(1, "confidence"),

        errors_df[
            (errors_df["true_class"] == "hispa") &
            (
                errors_df["predicted_class"] ==
                "normal"
            )
        ].nlargest(1, "confidence"),

        errors_df[
            (
                errors_df["true_class"] ==
                "downy_mildew"
            ) &
            (
                errors_df["predicted_class"] ==
                "brown_spot"
            )
        ].nlargest(1, "confidence")
    ])

    print(
        selected_errors[
            [
                "image_id",
                "true_class",
                "predicted_class",
                "confidence"
            ]
        ]
    )

    data_augmentation, base_model = (
        get_model_parts(model)
    )

    plt.figure(figsize=(12, 12))

    for i, (_, row) in enumerate(
        selected_errors.iterrows()
    ):
        image_path = (
            data_dir
            / "train_images"
            / row["label"]
            / row["image_id"]
        )

        image = load_image_for_gradcam(
            image_path
        )

        heatmap = make_gradcam_heatmap(
            image,
            model,
            base_model,
            data_augmentation
        )

        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],
            (224, 224)
        ).numpy().squeeze()

        plt.subplot(3, 2, 2 * i + 1)

        plt.imshow(
            image.numpy().astype("uint8")
        )

        plt.title(
            f"Stvarno: {row['true_class']}\n"
            f"Predikcija: {row['predicted_class']}"
        )

        plt.axis("off")

        plt.subplot(3, 2, 2 * i + 2)

        plt.imshow(
            image.numpy().astype("uint8")
        )

        plt.imshow(
            heatmap,
            cmap="jet",
            alpha=0.4
        )

        plt.title("Grad-CAM")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_test(
    model,
    test_dataset_effnet,
    class_names
):
    test_loss, test_accuracy = model.evaluate(
        test_dataset_effnet,
        verbose=1
    )

    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Test loss: "
        f"{test_loss:.4f}"
    )

    test_predictions = model.predict(
        test_dataset_effnet
    )

    test_pred = np.argmax(
        test_predictions,
        axis=1
    )

    test_true = np.concatenate([
        labels.numpy()
        for _, labels in test_dataset_effnet
    ])

    print(
        classification_report(
            test_true,
            test_pred,
            target_names=class_names,
            digits=3
        )
    )

    test_cm = confusion_matrix(
        test_true,
        test_pred
    )

    plot_confusion_matrix(
        test_cm,
        class_names,
        "Matrica konfuzije - test skup"
    )


def main():
    args = parse_arguments()

    set_random_seed()

    (
        val_data,
        test_data,
        class_names,
        val_dataset_effnet,
        test_dataset_effnet
    ) = prepare_data(args.data_dir)

    model = tf.keras.models.load_model(
        args.model_path
    )

    print(
        "Model učitan:",
        args.model_path
    )

    y_true, y_pred, predictions = (
        evaluate_validation(
            model,
            val_dataset_effnet,
            class_names
        )
    )

    errors_df = analyze_errors(
        val_data,
        y_true,
        y_pred,
        predictions,
        class_names,
        args.data_dir
    )

    show_gradcam(
        errors_df,
        model,
        args.data_dir
    )

    evaluate_test(
        model,
        test_dataset_effnet,
        class_names
    )


if __name__ == "__main__":
    main()

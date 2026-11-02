"""
ai2.py — FER2013 CNN 診斷實驗
資料結構：
archive/
  train/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/
    angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/

安裝：
    pip install tensorflow scikit-learn matplotlib pandas

執行：
    python ai2.py
    python ai2.py --suite architecture
    python ai2.py --suite dropout
    python ai2.py --suite augmentation
    python ai2.py --suite epochs
    python ai2.py --suite regularization
    python ai2.py --suite all --epochs 20

注意：
- 實驗只用 train 資料中的 20% 作 validation；test 不參與超參數選擇。
- 所有模型都會另外在「未做隨機增強的訓練影像」與 validation 影像上評估，
  讓 train/validation 的比較更公平。
- 每個實驗會記錄 history、評估指標、混淆矩陣與模型檔。
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix


# -------------------- 基本設定 --------------------
DATA_DIR = Path("archive")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42
OUTPUT_DIR = Path("ai2_results")
NUM_CLASSES = 7

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def check_data():
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"找不到訓練資料夾：{TRAIN_DIR.resolve()}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"找不到測試資料夾：{TEST_DIR.resolve()}")

    train_classes = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    test_classes = sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir())
    if train_classes != test_classes:
        raise ValueError(
            f"train/test 類別資料夾不一致。\ntrain={train_classes}\ntest={test_classes}"
        )
    if len(train_classes) != NUM_CLASSES:
        print(f"警告：找到 {len(train_classes)} 個類別：{train_classes}")
    print("類別資料夾：", train_classes)
    return train_classes


def count_files_by_class(directory):
    rows = []
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        count = sum(1 for p in class_dir.rglob("*") if p.is_file())
        rows.append({"split": directory.name, "class": class_dir.name, "count": count})
    return rows


def augmentation_config(level):
    """只改變影像增強強度；不改變模型、optimizer 或 epoch。"""
    configs = {
        "none": {},
        "mild": {
            "rotation_range": 5,
            "width_shift_range": 0.03,
            "height_shift_range": 0.03,
            "zoom_range": 0.03,
            "horizontal_flip": True,
            "fill_mode": "nearest",
        },
        "medium": {
            "rotation_range": 10,
            "width_shift_range": 0.07,
            "height_shift_range": 0.07,
            "zoom_range": 0.07,
            "horizontal_flip": True,
            "fill_mode": "nearest",
        },
        "strong": {
            "rotation_range": 15,
            "width_shift_range": 0.10,
            "height_shift_range": 0.10,
            "shear_range": 0.10,
            "zoom_range": 0.10,
            "horizontal_flip": True,
            "fill_mode": "nearest",
        },
    }
    return configs[level]


def make_generators(aug_level="mild", batch_size=BATCH_SIZE):
    # train 的 20% 作 validation；seed 固定以確保不同實驗切分一致。
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        **augmentation_config(aug_level),
    )
    clean_split = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        str(TRAIN_DIR), target_size=IMG_SIZE, color_mode="grayscale",
        batch_size=batch_size, class_mode="categorical", subset="training",
        shuffle=True, seed=SEED,
    )
    # 同一個 train split，但不做隨機影像增強，用來公平評估訓練資料。
    train_eval_gen = clean_split.flow_from_directory(
        str(TRAIN_DIR), target_size=IMG_SIZE, color_mode="grayscale",
        batch_size=batch_size, class_mode="categorical", subset="training",
        shuffle=False, seed=SEED,
    )
    val_gen = clean_split.flow_from_directory(
        str(TRAIN_DIR), target_size=IMG_SIZE, color_mode="grayscale",
        batch_size=batch_size, class_mode="categorical", subset="validation",
        shuffle=False, seed=SEED,
    )
    test_gen = test_datagen.flow_from_directory(
        str(TEST_DIR), target_size=IMG_SIZE, color_mode="grayscale",
        batch_size=batch_size, class_mode="categorical", shuffle=False,
    )

    if train_gen.class_indices != val_gen.class_indices or train_gen.class_indices != test_gen.class_indices:
        raise ValueError(
            "不同資料集的 class_indices 不一致：\n"
            f"train={train_gen.class_indices}\nval={val_gen.class_indices}\n"
            f"test={test_gen.class_indices}"
        )
    return train_gen, train_eval_gen, val_gen, test_gen


def build_model(architecture="baseline", dropout=0.5, l2_strength=0.0, learning_rate=1e-3):
    """架構實驗只改網路容量；其餘設定由實驗表指定。"""
    inputs = layers.Input(shape=(48, 48, 1))
    x = inputs

    if architecture == "small":
        filters = [16, 32]
        dense_units = 64
    elif architecture == "baseline":
        filters = [32, 64, 128]
        dense_units = 128
    elif architecture == "deeper":
        filters = [32, 64, 128, 128]
        dense_units = 128
    else:
        raise ValueError(f"未知架構：{architecture}")

    for i, f in enumerate(filters):
        x = layers.Conv2D(
            f, (3, 3), padding="same", activation="relu",
            kernel_regularizer=regularizers.l2(l2_strength) if l2_strength else None,
            name=f"conv_{i+1}",
        )(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool_{i+1}")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength) if l2_strength else None,
    )(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_experiments(suite, max_epochs):
    # 每個實驗盡量只改一個因素，方便歸因。
    base = dict(architecture="baseline", dropout=0.5, aug="mild",
                epochs=max_epochs, early_stop=False, l2=0.0, lr=1e-3,
                reduce_lr=False)
    experiments = []

    def add(name, **kwargs):
        cfg = base.copy()
        cfg.update(kwargs)
        cfg["name"] = name
        experiments.append(cfg)

    if suite in ("all", "architecture"):
        add("arch_small", architecture="small")
        add("arch_baseline", architecture="baseline")
        add("arch_deeper", architecture="deeper")

    if suite in ("all", "dropout"):
        add("dropout_0.0", dropout=0.0)
        add("dropout_0.3", dropout=0.3)
        add("dropout_0.5", dropout=0.5)
        add("dropout_0.6", dropout=0.6)

    if suite in ("all", "augmentation"):
        add("aug_none", aug="none")
        add("aug_mild", aug="mild")
        add("aug_medium", aug="medium")
        add("aug_strong", aug="strong")

    if suite in ("all", "epochs"):
        # 同一模型與資料增強，分別固定訓練 10/20/40 epochs。
        # 此組刻意不啟用 EarlyStopping，才看得出訓練時間的影響。
        for e in sorted(set([10, 20, 40])):
            add(f"epochs_{e}", epochs=e, early_stop=False)

    if suite in ("all", "regularization"):
        add("l2_1e-4", l2=1e-4)
        add("lr_3e-4", lr=3e-4)
        add("baseline_earlystop", early_stop=True, reduce_lr=True)

    # 避免 all 裡出現重複基準模型；保留不同 suite 的實驗命名。
    unique = {}
    for exp in experiments:
        unique[exp["name"]] = exp
    return list(unique.values())


def evaluate_and_report(model, generator, class_names, prefix, out_dir):
    generator.reset()
    loss, acc = model.evaluate(generator, verbose=0)
    generator.reset()
    probs = model.predict(generator, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = generator.classes
    labels = list(range(len(generator.class_indices)))
    ordered_names = [name for name, idx in sorted(generator.class_indices.items(), key=lambda x: x[1])]

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=ordered_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=ordered_names, columns=ordered_names).to_csv(
        out_dir / f"{prefix}_confusion_matrix.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(report).transpose().to_csv(
        out_dir / f"{prefix}_classification_report.csv", encoding="utf-8-sig"
    )
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_recall": float(report["macro avg"]["recall"]),
    }


def plot_history(history, name, out_dir):
    hist = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, hist.get("loss", []), label="train loss")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], label="validation loss")
    axes[0].set_title(f"{name}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, hist.get("accuracy", []), label="train accuracy")
    if "val_accuracy" in hist:
        axes[1].plot(epochs, hist["val_accuracy"], label="validation accuracy")
    axes[1].set_title(f"{name}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="FER2013 過擬合／欠擬合診斷實驗")
    parser.add_argument(
        "--suite", default="all",
        choices=["all", "architecture", "dropout", "augmentation", "epochs", "regularization"],
        help="要執行的實驗組別；all 會執行全部實驗",
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="非 epochs 實驗的訓練回合數，預設 20")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="額外評估每個實驗的 test；不建議用它挑超參數，避免測試集洩漏")
    args = parser.parse_args()

    set_seed(args.seed)
    classes = check_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 儲存原始類別數量，協助確認資料是否不平衡。
    counts = count_files_by_class(TRAIN_DIR) + count_files_by_class(TEST_DIR)
    pd.DataFrame(counts).to_csv(OUTPUT_DIR / "class_counts.csv", index=False, encoding="utf-8-sig")
    print("\n各類別資料數量已存到 ai2_results/class_counts.csv")

    experiments = get_experiments(args.suite, args.epochs)
    print(f"\n實驗組數：{len(experiments)}；suite={args.suite}")
    print("每組都使用 train 資料中的 80% 訓練、20% validation。")
    print("Test 只作最後參考，不參與選參數。\n")

    results = []
    for i, exp in enumerate(experiments, start=1):
        name = exp["name"]
        print("\n" + "=" * 72)
        print(f"[{i}/{len(experiments)}] {name}")
        print({k: v for k, v in exp.items() if k != "name"})
        print("=" * 72)

        tf.keras.backend.clear_session()
        set_seed(args.seed)

        train_gen, train_eval_gen, val_gen, test_gen = make_generators(
            exp["aug"], args.batch_size
        )
        model = build_model(
            architecture=exp["architecture"],
            dropout=exp["dropout"],
            l2_strength=exp["l2"],
            learning_rate=exp["lr"],
        )

        callbacks = [
            CSVLogger(str(OUTPUT_DIR / f"{name}_epoch_log.csv")),
        ]
        if exp["early_stop"]:
            callbacks.append(EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True,
                min_delta=0.001, verbose=1,
            ))
        if exp["reduce_lr"]:
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2,
                min_lr=1e-6, verbose=1,
            ))

        start = time.time()
        history = model.fit(
            train_gen,
            epochs=exp["epochs"],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1,
        )
        elapsed = time.time() - start
        plot_history(history, name, OUTPUT_DIR)

        # clean train 評估與 validation 評估都不使用隨機增強，且都在 inference mode。
        train_metrics = evaluate_and_report(
            model, train_eval_gen, classes, f"{name}_train_clean", OUTPUT_DIR
        )
        val_metrics = evaluate_and_report(
            model, val_gen, classes, f"{name}_val", OUTPUT_DIR
        )

        row = {
            "experiment": name,
            "architecture": exp["architecture"],
            "dropout": exp["dropout"],
            "augmentation": exp["aug"],
            "max_epochs": exp["epochs"],
            "epochs_ran": len(history.history.get("loss", [])),
            "l2": exp["l2"],
            "learning_rate": exp["lr"],
            "train_clean_loss": train_metrics["loss"],
            "train_clean_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "accuracy_gap_train_minus_val": train_metrics["accuracy"] - val_metrics["accuracy"],
            "loss_gap_val_minus_train": val_metrics["loss"] - train_metrics["loss"],
            "seconds": round(elapsed, 1),
        }

        if args.evaluate_test:
            test_metrics = evaluate_and_report(
                model, test_gen, classes, f"{name}_test", OUTPUT_DIR
            )
            row.update({
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            })

        model.save(OUTPUT_DIR / f"{name}.keras")
        results.append(row)
        pd.DataFrame(results).to_csv(
            OUTPUT_DIR / "experiment_summary_partial.csv",
            index=False, encoding="utf-8-sig",
        )

        print(
            f"\n{name} 完成 | train(clean) acc={train_metrics['accuracy']:.4f} "
            f"| val acc={val_metrics['accuracy']:.4f} "
            f"| val macro-F1={val_metrics['macro_f1']:.4f} "
            f"| acc gap={row['accuracy_gap_train_minus_val']:.4f}"
        )

    summary = pd.DataFrame(results).sort_values(
        by=["val_macro_f1", "val_accuracy"], ascending=False
    )
    summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 72)
    print("全部實驗完成。依 validation Macro F1 排序：")
    cols = [
        "experiment", "epochs_ran", "train_clean_accuracy", "val_accuracy",
        "val_macro_f1", "val_weighted_f1", "accuracy_gap_train_minus_val",
        "loss_gap_val_minus_train", "seconds",
    ]
    print(summary[cols].to_string(index=False))
    print(f"\n完整結果與圖表位於：{OUTPUT_DIR.resolve()}")
    print("主要檔案：")
    print("  experiment_summary.csv          各組比較總表")
    print("  *_curves.png                    每組 Loss/Accuracy 曲線")
    print("  *_epoch_log.csv                 每個 Epoch 的訓練記錄")
    print("  *_classification_report.csv     各類別 Precision/Recall/F1")
    print("  *_confusion_matrix.csv          混淆矩陣")
    print("  class_counts.csv                train/test 各類別數量")


if __name__ == "__main__":
    main()

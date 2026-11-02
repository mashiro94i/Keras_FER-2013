import os
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

OUTPUT_DIR = Path("ai2_results")
TRAIN_DIR = Path("archive/train")
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
SEED = 42

def reconstruct_summary():
    model_paths = list(OUTPUT_DIR.glob("*.keras"))
    if not model_paths:
        print(f"❌ 錯誤：在 [{OUTPUT_DIR.resolve()}] 中找不到任何 .keras 模型檔案！")
        return

    print(f"🔍 找到 {len(model_paths)} 個模型檔案，開始自動重建 experiment_summary.csv...")
    
    # 建立驗證集 generator
    clean_split = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    val_gen = clean_split.flow_from_directory(
        str(TRAIN_DIR), target_size=IMG_SIZE, color_mode="grayscale",
        batch_size=BATCH_SIZE, class_mode="categorical", subset="validation",
        shuffle=False, seed=SEED
    )
    
    # 建立乾淨訓練集 generator (用於評估 train clean accuracy)
    train_eval_gen = clean_split.flow_from_directory(
        str(TRAIN_DIR), target_size=IMG_SIZE, color_mode="grayscale",
        batch_size=BATCH_SIZE, class_mode="categorical", subset="training",
        shuffle=False, seed=SEED
    )

    results = []

    for model_path in model_paths:
        name = model_path.stem  # 例如 arch_small, dropout_0.3 等
        print(f"  正在處理與評估: {name} ...")
        
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"    ⚠️ 無法載入模型 {model_path}: {e}")
            continue

        # 評估 Clean Train
        train_eval_gen.reset()
        train_loss, train_acc = model.evaluate(train_eval_gen, verbose=0)
        
        # 評估 Validation
        val_gen.reset()
        val_loss, val_acc = model.evaluate(val_gen, verbose=0)
        
        val_gen.reset()
        val_probs = model.predict(val_gen, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)
        val_true = val_gen.classes
        val_labels = list(range(len(val_gen.class_indices)))
        val_ordered_names = [k for k, v in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]
        
        val_report = classification_report(
            val_true, val_preds, labels=val_labels, target_names=val_ordered_names,
            output_dict=True, zero_division=0
        )

        # 從對應的 epoch log 讀取實際跑了幾個 epochs
        log_path = OUTPUT_DIR / f"{name}_epoch_log.csv"
        epochs_ran = 0
        if log_path.exists():
            log_df = pd.read_csv(log_path)
            epochs_ran = len(log_df)

        # 依命名規則還原超參數記錄
        architecture = "small" if "small" in name else ("deeper" if "deeper" in name else "baseline")
        
        dropout_val = 0.5
        if "dropout_" in name:
            try:
                dropout_val = float(name.replace("dropout_", ""))
            except:
                pass
        elif name == "dropout_0.0":
            dropout_val = 0.0

        aug_level = name.split("_")[1] if name.startswith("aug_") else "mild"
        l2_val = 1e-4 if "l2_" in name else 0.0
        lr_val = 3e-4 if "lr_" in name else 1e-3
        max_ep = int(name.replace("epochs_", "")) if "epochs_" in name else 20

        row = {
            "experiment": name,
            "architecture": architecture,
            "dropout": dropout_val,
            "augmentation": aug_level,
            "max_epochs": max_ep,
            "epochs_ran": epochs_ran,
            "l2": l2_val,
            "learning_rate": lr_val,
            "train_clean_loss": float(train_loss),
            "train_clean_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_report["macro avg"]["f1-score"]),
            "val_weighted_f1": float(val_report["weighted avg"]["f1-score"]),
            "val_macro_recall": float(val_report["macro avg"]["recall"]),
            "accuracy_gap_train_minus_val": float(train_acc - val_acc),
            "loss_gap_val_minus_train": float(val_loss - train_loss),
            "seconds": 0.0
        }
        results.append(row)

    if results:
        summary = pd.DataFrame(results).sort_values(by=["val_macro_f1", "val_accuracy"], ascending=False)
        summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n✨ 成功重建 experiment_summary.csv！共收錄 {len(summary)} 筆實驗記錄，已存至 {OUTPUT_DIR / 'experiment_summary.csv'}")
    else:
        print("\n❌ 沒有成功還原任何實驗數據。")

if __name__ == "__main__":
    reconstruct_summary()
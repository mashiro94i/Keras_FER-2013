import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# ==========================================
# 0. 設定命令列參數 (argparse)
# ==========================================
parser = argparse.ArgumentParser(description="FER2013 情感辨識模型訓練控制與過擬合評估")
parser.add_argument('--train', type=str, default=None, choices=['1', '2', '2_mod', '3'], 
                    help="指定要強制造排執行的任務: 1(原始), 2(標準平衡), 2_mod(對齊倒數第二), 3(GAN增強)")
parser.add_argument('--tag', type=str, default='1', help="自定義檔案辨識後綴參數")
args = parser.parse_args()

print(f"當前執行參數設定 -> 強制訓練任務: {args.train}, 檔案標籤 (tag): {args.tag}")

# 本地資料夾路徑設定
data_dir = 'archive'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# 載入本地圖片轉為 NumPy 陣列的函數
def load_fer2013_from_folder(folder_path):
    faces = []
    labels = []
    classes = sorted(os.listdir(folder_path))
    classes = [c for c in classes if os.path.isdir(os.path.join(folder_path, c))]
    
    for idx, emotion in enumerate(classes):
        emotion_path = os.path.join(folder_path, emotion)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            try:
                img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')
                faces.append(img_to_array(img) / 255.0)
                labels.append(idx)
            except Exception:
                pass

    return np.asarray(faces), np.asarray(labels), classes

print("正在從本地載入測試集 (Test Set)...")
X_val, y_val_raw, classes = load_fer2013_from_folder(test_dir)
y_val = to_categorical(y_val_raw, num_classes=len(classes))

# 定義共用的 CNN 模型架構
def create_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 檔案路徑定義
model_path_1 = f"model_task1_{args.tag}.keras"
model_path_2 = f"model_task2_{args.tag}.keras"
model_path_2mod = f"model_task2_mod_{args.tag}.keras"
model_path_3 = f"model_task3_{args.tag}.keras"

data_path_1 = f"train_data_1_{args.tag}.npz"
data_path_2 = f"train_data_2_{args.tag}.npz"
data_path_2mod = f"train_data_2mod_{args.tag}.npz"
data_path_3 = f"train_data_3_{args.tag}.npz"

run_1 = (args.train == '1') or (args.train is None and not os.path.exists(model_path_1))
run_2 = (args.train == '2') or (args.train is None and not os.path.exists(model_path_2))
run_2mod = (args.train == '2_mod') or (args.train is None and not os.path.exists(model_path_2mod))
run_3 = (args.train == '3') or (args.train is None and not os.path.exists(model_path_3))

X_train_raw, y_train_raw = None, None
if run_1 or run_2 or run_2mod or run_3:
    print("正在從本地載入訓練集...")
    X_train_raw, y_train_raw, _ = load_fer2013_from_folder(train_dir)

# ==========================================
# 1. 任務一：原始資料直接訓練
# ==========================================
if run_1:
    print("=== 開始訓練：任務一（原始資料不處理） ===")
    y_train_1 = to_categorical(y_train_raw, num_classes=len(classes))
    model_base = create_cnn_model()
    model_base.fit(X_train_raw, y_train_1, epochs=15, batch_size=64, validation_data=(X_val, y_val))
    model_base.save(model_path_1)
    np.savez(data_path_1, X=X_train_raw, y=y_train_1)
else:
    print(f"👉 任務一檔案已存在，跳過訓練。")

# ==========================================
# 2. 任務二：標準資料平衡
# ==========================================
if run_2:
    print("=== 開始訓練：任務二（標準增強平衡） ===")
    unique_classes, counts = np.unique(y_train_raw, return_counts=True)
    target_per_class = max(counts)
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
                                 shear_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest')
    balanced_X_2, balanced_y_2 = [X_train_raw], [y_train_raw]
    for i, c_name in enumerate(classes):
        idx = np.where(y_train_raw == i)[0]
        X_class = X_train_raw[idx]
        current_count = len(X_class)
        if current_count < target_per_class:
            needed = target_per_class - current_count
            gen = datagen.flow(X_class, np.zeros(current_count), batch_size=64)
            aug_imgs, aug_lbls, gen_cnt = [], [], 0
            while gen_cnt < needed:
                bx, _ = next(gen)
                for img in bx:
                    if gen_cnt < needed:
                        aug_imgs.append(img)
                        aug_lbls.append(i)
                        gen_cnt += 1
                    else:
                        break
            balanced_X_2.append(np.array(aug_imgs))
            balanced_y_2.append(np.array(aug_lbls))
    X_train_b2 = np.concatenate(balanced_X_2, axis=0)
    y_train_b2_raw = np.concatenate(balanced_y_2, axis=0)
    y_train_b2 = to_categorical(y_train_b2_raw, num_classes=len(classes))
    
    model_2 = create_cnn_model()
    model_2.fit(X_train_b2, y_train_b2, epochs=15, batch_size=64, validation_data=(X_val, y_val))
    model_2.save(model_path_2)
    np.savez(data_path_2, X=X_train_b2, y=y_train_b2)
else:
    print(f"👉 任務二檔案已存在，跳過訓練。")

# ==========================================
# 3. 任務二改：對齊倒數第二個類別
# ==========================================
if run_2mod:
    print("=== 開始訓練：任務二改（對齊倒數第二個類別） ===")
    unique_classes, counts = np.unique(y_train_raw, return_counts=True)
    sorted_indices = np.argsort(counts)
    target_class_idx = sorted_indices[-2]
    target_count = counts[target_class_idx]
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.15, height_shift_range=0.15,
                                 shear_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest')
    balanced_X_mod, balanced_y_mod = [], []
    for i, c_name in enumerate(classes):
        idx = np.where(y_train_raw == i)[0]
        X_class = X_train_raw[idx]
        current_count = len(X_class)
        if current_count > target_count:
            np.random.seed(42)
            selected_idx = np.random.choice(current_count, target_count, replace=False)
            balanced_X_mod.append(X_class[selected_idx])
            balanced_y_mod.append(np.full(target_count, i))
        elif current_count < target_count:
            needed = target_count - current_count
            balanced_X_mod.append(X_class)
            balanced_y_mod.append(np.full(current_count, i))
            gen = datagen.flow(X_class, np.zeros(current_count), batch_size=64)
            aug_imgs, aug_lbls, gen_cnt = [], [], 0
            while gen_cnt < needed:
                bx, _ = next(gen)
                for img in bx:
                    if gen_cnt < needed:
                        aug_imgs.append(img)
                        aug_lbls.append(i)
                        gen_cnt += 1
                    else:
                        break
            balanced_X_mod.append(np.array(aug_imgs))
            balanced_y_mod.append(np.array(aug_lbls))
        else:
            balanced_X_mod.append(X_class)
            balanced_y_mod.append(np.full(current_count, i))
    X_train_mod = np.concatenate(balanced_X_mod, axis=0)
    y_train_mod_raw = np.concatenate(balanced_y_mod, axis=0)
    y_train_mod = to_categorical(y_train_mod_raw, num_classes=len(classes))

    model_mod = create_cnn_model()
    model_mod.fit(X_train_mod, y_train_mod, epochs=15, batch_size=64, validation_data=(X_val, y_val))
    model_mod.save(model_path_2mod)
    np.savez(data_path_2mod, X=X_train_mod, y=y_train_mod)
else:
    print(f"👉 任務二改檔案已存在，跳過訓練。")

# ==========================================
# 4. 任務三：DCGAN 增強訓練
# ==========================================
if run_3:
    print("=== 開始訓練：任務三（DCGAN 增強） ===")
    latent_dim = 100
    def build_generator():
        return models.Sequential([
            layers.Dense(12 * 12 * 128, input_dim=latent_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((12, 12, 128)),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')
        ])
    def build_discriminator():
        model = models.Sequential([
            layers.Input(shape=(48, 48, 1)),
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    generator = build_generator()
    discriminator = build_discriminator()
    X_train_gan_input = (X_train_raw * 2.0) - 1.0
    half_batch = 32
    for epoch in range(3):
        idx = np.random.randint(0, X_train_gan_input.shape[0], half_batch)
        real_imgs = X_train_gan_input[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        gen_imgs = generator.predict(noise, verbose=0)
        discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

    unique_classes, counts = np.unique(y_train_raw, return_counts=True)
    target_per_class = max(counts)
    synthetic_X, synthetic_y = [X_train_raw], [y_train_raw]
    for i in range(len(classes)):
        current_count = np.sum(y_train_raw == i)
        needed = max(0, target_per_class - current_count)
        if needed > 0:
            noise = np.random.normal(0, 1, (needed, latent_dim))
            gen_synthetic = generator.predict(noise, verbose=0)
            gen_synthetic = np.clip((gen_synthetic + 1.0) / 2.0, 0.0, 1.0)
            synthetic_X.append(gen_synthetic)
            synthetic_y.append(np.full(needed, i))
    X_train_gan_aug = np.concatenate(synthetic_X, axis=0)
    y_train_gan_raw = np.concatenate(synthetic_y, axis=0)
    y_train_gan = to_categorical(y_train_gan_raw, num_classes=len(classes))
    
    model_3 = create_cnn_model()
    model_3.fit(X_train_gan_aug, y_train_gan, epochs=15, batch_size=64, validation_data=(X_val, y_val))
    model_3.save(model_path_3)
    np.savez(data_path_3, X=X_train_gan_aug, y=y_train_gan)
else:
    print(f"👉 任務三檔案已存在，跳過訓練。")

# ==========================================
# 5. 綜合評估：訓練集 vs 測試集（過擬合診斷）
# ==========================================
print("\n" + "="*60)
print("=== 訓練集 vs 測試集效能對比（過擬合 Overfitting 診斷報告） ===")
print("="*60)

tasks_to_eval = [
    ("任務一（原始資料）", model_path_1, data_path_1),
    ("任務二（標準平衡）", model_path_2, data_path_2),
    ("任務二改（對齊倒數第二）", model_path_2mod, data_path_2mod),
    ("任務三（GAN 增強）", model_path_3, data_path_3)
]

for name, model_p, data_p in tasks_to_eval:
    if os.path.exists(model_p) and os.path.exists(data_p):
        model = tf.keras.models.load_model(model_p)
        train_data = np.load(data_p)
        X_train_eval, y_train_eval = train_data['X'], train_data['y']
        
        # 評估訓練集
        train_loss, train_acc = model.evaluate(X_train_eval, y_train_eval, verbose=0)
        # 評估測試集
        test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
        
        acc_gap = train_acc - test_acc
        
        print(f"\n📌 【 {name} 】")
        print(f"   - 訓練集 (Train) -> Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(f"   - 測試集 (Test)  -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
        print(f"   - 準確率落差 (Gap) : {acc_gap:.4f}  (若差距過大代表過擬合嚴重)")
        
        # 輸出測試集的詳細分類報告
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)
        print("\n   📊 測試集詳細分類指標 (Precision / Recall / F1-score):")
        print(classification_report(y_true, y_pred, target_names=classes, digits=4))
        print("-" * 60)
    else:
        print(f"\n👉 [{name}] 模型或訓練資料檔案不存在，請先執行對應訓練。")
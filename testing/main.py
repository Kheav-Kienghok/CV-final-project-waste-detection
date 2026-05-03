import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import BackbonePreprocess

# ----------------------------
# CONFIG
# ----------------------------
MODEL_1_PATH = "/workspaces/CV-final-project-waste-detection/app/models/mobilenetv2.keras"
MODEL_2_PATH = "/workspaces/CV-final-project-waste-detection/app/models/efficientnetb3.keras"

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]
PLASTIC_CLASS = "plastic"

IMAGES_FOLDER = "/workspaces/CV-final-project-waste-detection/images"     # folder name
IMG_SIZE = (224, 224)        # change if your model expects different size
RUNS = 50                    # number of timed predictions
WARMUP_RUNS = 5              # warmup predictions (not counted)


# ----------------------------
# LOAD MODEL WITH TIMER
# ----------------------------
def load_model_with_time(model_path):
    start = time.time()
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "BackbonePreprocess": BackbonePreprocess,
            "SmartWaste>BackbonePreprocess": BackbonePreprocess,
        },
    )
    end = time.time()
    return model, (end - start)


def get_model_name(model_path):
    return os.path.splitext(os.path.basename(model_path))[0]


# ----------------------------
# LOAD IMAGE
# ----------------------------
def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
    img_array = img_array / 255.0                  # normalize (0-1)
    return img_array


def predict_label(model, input_data):
    preds = model.predict(input_data, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])
    return pred_label, confidence


def analyze_model(model, model_name, image_files):
    plastic_count = 0
    rows = []
    start = time.time()

    for image_file in image_files:
        img_path = os.path.join(IMAGES_FOLDER, image_file)
        input_data = load_image(img_path, IMG_SIZE)
        pred_label, confidence = predict_label(model, input_data)

        if pred_label == PLASTIC_CLASS:
            plastic_count += 1

        rows.append((image_file, pred_label, confidence))

    total_time = time.time() - start
    avg_time = total_time / len(image_files)
    fps = 1 / avg_time if avg_time > 0 else float("inf")

    return rows, plastic_count, total_time, avg_time, fps


# ----------------------------
# BENCHMARK FUNCTION
# ----------------------------
def benchmark_model(model, input_data, warmup_runs=5, runs=50):
    # Warmup (important for fair timing)
    for _ in range(warmup_runs):
        _ = model.predict(input_data, verbose=0)

    # Timed inference
    start = time.time()
    for _ in range(runs):
        _ = model.predict(input_data, verbose=0)
    end = time.time()

    total_time = end - start
    avg_time = total_time / runs
    fps = 1 / avg_time

    return total_time, avg_time, fps


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    image_files = sorted(
        f for f in os.listdir(IMAGES_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print("❌ No images found in the images folder.")
        exit()

    print("\n===================================")
    print(" Loading Models...")
    print("===================================")

    model1_name = get_model_name(MODEL_1_PATH)
    model2_name = get_model_name(MODEL_2_PATH)

    model1, load_time1 = load_model_with_time(MODEL_1_PATH)
    model2, load_time2 = load_model_with_time(MODEL_2_PATH)

    print(f"{model1_name} Load Time ({MODEL_1_PATH}): {load_time1:.4f} sec")
    print(f"{model2_name} Load Time ({MODEL_2_PATH}): {load_time2:.4f} sec")

    print("\n===================================")
    print(" Running Predictions On All Images...")
    print("===================================")

    model1_rows, model1_plastic_count, model1_total, model1_avg, model1_fps = analyze_model(
        model1, model1_name, image_files
    )
    model2_rows, model2_plastic_count, model2_total, model2_avg, model2_fps = analyze_model(
        model2, model2_name, image_files
    )

    # Print a consolidated per-image comparison table
    combined_rows = []
    for idx, image_file in enumerate(image_files):
        m1_pred, m1_conf = model1_rows[idx][1], model1_rows[idx][2]
        m2_pred, m2_conf = model2_rows[idx][1], model2_rows[idx][2]
        combined_rows.append((image_file, f"{m1_pred} ({m1_conf:.0%})", f"{m2_pred} ({m2_conf:.0%})"))

    # Compute column widths (include header in calculation)
    header_row = ("Image", model1_name, model2_name)
    name_w = max(len(header_row[0]), max(len(r[0]) for r in combined_rows))
    m1_w = max(len(header_row[1]), max(len(r[1]) for r in combined_rows))
    m2_w = max(len(header_row[2]), max(len(r[2]) for r in combined_rows))

    sep = " | "
    header = f"{'Image'.ljust(name_w)}{sep}{model1_name.ljust(m1_w)}{sep}{model2_name.ljust(m2_w)}"
    rule = "-" * len(header)

    print(rule)
    print(header)
    print(rule)
    for r in combined_rows:
        print(f"{r[0].ljust(name_w)}{sep}{r[1].ljust(m1_w)}{sep}{r[2].ljust(m2_w)}")
    print(rule)

    print("\n===================================")
    print(" Summary")
    print("===================================")

    print("\n========== RESULTS ================")
    print(f"{model1_name} ({MODEL_1_PATH})")
    print(f"  Plastic predictions: {model1_plastic_count}/{len(image_files)}")
    print(f"  Total Time         : {model1_total:.4f} sec ({len(image_files)} images)")
    print(f"  Avg Time           : {model1_avg:.6f} sec/image")
    print(f"  FPS                : {model1_fps:.2f}")

    print(f"\n{model2_name} ({MODEL_2_PATH})")
    print(f"  Plastic predictions: {model2_plastic_count}/{len(image_files)}")
    print(f"  Total Time         : {model2_total:.4f} sec ({len(image_files)} images)")
    print(f"  Avg Time           : {model2_avg:.6f} sec/image")
    print(f"  FPS                : {model2_fps:.2f}")

    print("\n===================================")
    print(" Comparison Summary")
    print("===================================")

    if model1_plastic_count > model2_plastic_count:
        print(f"✅ {model1_name} predicted more plastic images by {model1_plastic_count - model2_plastic_count}")
    elif model2_plastic_count > model1_plastic_count:
        print(f"✅ {model2_name} predicted more plastic images by {model2_plastic_count - model1_plastic_count}")
    else:
        print("✅ Both models predicted the same number of plastic images")
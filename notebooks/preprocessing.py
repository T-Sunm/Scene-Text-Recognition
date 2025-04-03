import os
import json
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import shutil
import os
import json

ds = load_dataset("keremberke/license-plate-object-detection", "full")
def save_images_and_info(dataset, split_name, save_dir="../license-plate-dataset"):
  """
  Lưu ảnh và metadata của từng sample từ dataset vào các thư mục:
    - save_dir/split_name/image: lưu ảnh
    - save_dir/split_name/info: lưu metadata (JSON)
  """
  image_dir = os.path.join(save_dir, split_name, "image")
  info_dir = os.path.join(save_dir, split_name, "info")
  os.makedirs(image_dir, exist_ok=True)
  os.makedirs(info_dir, exist_ok=True)

  for sample in dataset:
    image_id = sample["image_id"]
    image = sample["image"]

    # Lưu ảnh
    image_path = os.path.join(image_dir, f"{image_id}.png")
    image.save(image_path)

    info = {k: sample[k] for k in sample if k != "image"}

    # Lưu metadata vào file JSON
    info_path = os.path.join(info_dir, f"{image_id}.json")
    with open(info_path, "w") as f:
      json.dump(info, f, indent=4)

    print(f"Lưu ảnh: {image_path} và metadata: {info_path}")


save_dir = "../license-plate-dataset"
save_images_and_info(ds["train"], "train", save_dir=save_dir)
save_images_and_info(ds["validation"], "validation", save_dir=save_dir)
save_images_and_info(ds["test"], "test", save_dir=save_dir)


def convert_to_yolo_format(data_list):
  """
  Chuyển đổi dữ liệu bounding box theo định dạng:
    (img_path, [w_img, h_img], [ [x, y, w, h], ... ])
  sang định dạng YOLO:
    (img_path, [ "class_id x_center y_center w h", ... ])

  Mọi giá trị tọa độ và kích thước sẽ được chuẩn hóa theo kích thước ảnh.
  Giả sử chỉ có 1 class (class_id = 0).
  """
  yolo_data = []
  for img_path, img_size, bboxs in data_list:
    data_img = []
    w_img, h_img = img_size
    for bbox in bboxs:
      x, y, w, h = bbox
      # Tính tâm bounding box
      x_center = x + w / 2
      y_center = y + h / 2

      # Chuẩn hóa theo kích thước ảnh
      x_center /= w_img
      y_center /= h_img
      w /= w_img
      h /= h_img

      # Với 1 class, gán class_id = 0
      label = 0
      yolo_label = f"{label} {x_center} {y_center} {w} {h}"
      data_img.append(yolo_label)
    yolo_data.append((img_path, data_img))
  return yolo_data


# Ví dụ sử dụng:
data_list = [
    ('../license-plate-dataset/train/image/0.png',
     [472, 303], [[185.0, 231.0, 107.0, 37.0]]),
    ('../license-plate-dataset/train/image/1.png',
     [472, 303], [[172.0, 95.0, 96.0, 83.0]])
]

yolo_data = convert_to_yolo_format(data_list)
print(yolo_data)

import os
import random
import shutil
import pandas as pd

GTSRB_ROOT = "./gtsrb_data"
OUT_FOLDER = "./test_img"
SAMPLE_CNT = 10  # 提取10张放到test_img

os.makedirs(OUT_FOLDER, exist_ok=True)
test_csv = pd.read_csv(os.path.join(GTSRB_ROOT, "Test.csv"))
selected = test_csv.sample(n=SAMPLE_CNT, replace=False)

for _, row in selected.iterrows():
    src_path = os.path.join(GTSRB_ROOT, row["Path"])
    dst_path = os.path.join(OUT_FOLDER, os.path.basename(src_path))
    shutil.copy(src_path, dst_path)
print(f"成功提取 {SAMPLE_CNT} 张数据集图片至 {OUT_FOLDER}")
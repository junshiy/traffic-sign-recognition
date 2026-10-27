import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
import os
import glob
import random
from PIL import Image

# 网络类定义
class TrafficCNN(nn.Module):
    def __init__(self, num_classes):
        super(TrafficCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 配置区
MODEL_PATH = "traffic_43cls.pth"
TEST_IMG_FOLDER = "./test_img"
GTSRB_TEST_ROOT = "./gtsrb_data"

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# GTSRB 43类别中文标签
CLASS_NAMES = [
    "限速20km/h", "限速30km/h", "限速50km/h", "限速60km/h", "限速70km/h", "限速80km/h",
    "解除限速80km/h", "限速100km/h", "限速120km/h", "禁止超车", "禁止超车（货车）",
    "前方路口优先通行", "优先通行道路", "让行", "停车让行", "禁止通行",
    "禁止货车驶入", "禁止驶入", "注意危险", "向左急弯", "向右急弯",
    "连续弯道", "路面颠簸", "路面湿滑", "右侧车道变窄", "前方施工",
    "注意信号灯", "注意行人", "注意儿童", "注意非机动车", "注意冰雪路面",
    "野生动物出没", "解除所有限速禁令", "向右转弯", "向左转弯", "直行",
    "直行或右转", "直行或左转", "靠右行驶", "靠左行驶", "环岛通行",
    "禁止超车解除", "禁止货车超车解除"
]

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# 工具函数
def predict_one(pil_img):
    img = pil_img.convert("RGB")
    tensor_img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor_img)
        pred_idx = torch.argmax(out, dim=1).item()
    return pred_idx, CLASS_NAMES[pred_idx]

# 获取test_img文件夹图片列表，用于下拉框
# 原来的
def get_test_img_choices():
    exts = ["*.jpg", "*.png", "*.jpeg"]
    file_list = []
    for ext in exts:
        file_list.extend(glob.glob(os.path.join(TEST_IMG_FOLDER, ext)))
    basenames = sorted([os.path.basename(p) for p in file_list])
    return basenames

# 下拉选择test_img图片识别
def select_img_recognize(img_name):
    if img_name is None or img_name == "":
        return None, "请先从下拉框选择一张test_img内的图片！"
    full_path = os.path.join(TEST_IMG_FOLDER, img_name)
    img = Image.open(full_path).convert("RGB")
    cls_id, cls_text = predict_one(img)
    res_text = f"识别结果：类别编号{cls_id}，标识含义：{cls_text}"
    return img, res_text

# 上传图片识别
def upload_img_recognize(pil_img):
    if pil_img is None:
        return None, "请上传图片！"
    cls_id, cls_text = predict_one(pil_img)
    res_text = f"识别结果：类别编号{cls_id}，标识含义：{cls_text}"
    return pil_img, res_text

# 随机抽取5张数据集图片批量识别
def random_five_test():
    csv_path = os.path.join(GTSRB_TEST_ROOT, "Test.csv")
    import pandas as pd
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=5, replace=False)
    img_paths = []
    result_str = ""
    for idx, row in sample_df.iterrows():
        p = os.path.join(GTSRB_TEST_ROOT, row["Path"])
        img_paths.append(p)
    pil_imgs = []
    for i, p in enumerate(img_paths):
        im = Image.open(p).convert("RGB")
        pil_imgs.append(im)
        cid, cname = predict_one(im)
        result_str += f"第{i+1}张：类别{cid} → {cname}\n"
    return pil_imgs, result_str

# Gradio界面
with gr.Blocks(title="交通标识识别交互系统") as demo:
    gr.Markdown("# 🚦 交通标识识别系统（43类GTSRB）")
    gr.Markdown("功能：上传图片识别｜从test_img选择样本识别｜随机抽取5张数据集图片测试")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 方式1：上传图片识别")
            upload_in = gr.Image(type="pil", label="上传交通标志图片")
            btn_upload = gr.Button("开始识别")

            gr.Markdown("### 方式2：从test_img选择测试图片")
            dropdown = gr.Dropdown(choices=get_test_img_choices(), label="选择test_img内图片")
            btn_drop = gr.Button("选中图片识别")

            gr.Markdown("### 方式3：批量随机测试")
            btn_rand5 = gr.Button("随机抽取5张数据集图片测试", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("#### 单图识别结果")
            out_single_img = gr.Image(label="识别图片")
            out_single_text = gr.Textbox(label="识别信息")
        with gr.Column(scale=1):
            gr.Markdown("#### 随机5张批量结果")
            gallery_out = gr.Gallery(label="抽取的5张图片")
            text_five = gr.Textbox(label="批量识别结果", lines=6)

    # 绑定按钮事件
    btn_upload.click(upload_img_recognize, inputs=[upload_in], outputs=[out_single_img, out_single_text])
    btn_drop.click(select_img_recognize, inputs=[dropdown], outputs=[out_single_img, out_single_text])
    btn_rand5.click(random_five_test, outputs=[gallery_out, text_five])

demo.launch(share=False)
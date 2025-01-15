import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class CamouflageDataset(Dataset):
    def __init__(self, image_dir, gt_dir, image_transform=None, gt_transform=None):
        """
        Args:
            image_dir (str): 普通图像文件夹路径。
            gt_dir (str): 掩码图像文件夹路径。
            image_transform (callable, optional): 图像的变换操作。
            gt_transform (callable, optional): 掩码的变换操作。
        """
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.image_transform = image_transform
        self.gt_transform = gt_transform
        self.image_names = os.listdir(image_dir)  # 获取所有图像文件名

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 加载普通图像
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")  # 确保图像是 RGB 格式

        # 加载对应的掩码图像
        gt_name = image_name.replace(".jpg", ".png")  # 假设掩码图像文件名与普通图像文件名一致，只是扩展名不同
        gt_path = os.path.join(self.gt_dir, gt_name)
        gt = Image.open(gt_path).convert("L")  # 转换为灰度图像

        # 将图像和掩码转换为 NumPy 数组
        image = np.array(image)
        gt = np.array(gt)

        # 将掩码二值化（0 表示背景，1 表示目标）
        gt = (gt > 128).astype(np.float32)

        # 应用变换（如果有）
        if self.image_transform:
            image = self.image_transform(image)
        if self.gt_transform:
            gt = self.gt_transform(gt)

        # 返回图像和掩码
        return image, gt

# 定义图像变换
image_transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
    transforms.Resize((300,400)),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 定义掩码变换
gt_transform = transforms.Compose([
    transforms.ToTensor(),  # 将掩码转换为 PyTorch 张量
    transforms.Resize((300,400)),
    transforms.Lambda(lambda x: (x > 0.5).float())  # 将掩码二值化
])

# 定义数据集路径
train_image_dir = "datasets/dataset/train/image"
train_gt_dir = "datasets/dataset/train/GT"

# 创建数据集实例
train_dataset = CamouflageDataset(image_dir=train_image_dir, gt_dir=train_gt_dir, image_transform=image_transform, gt_transform=gt_transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义测试数据集路径
test_image_dir = "datasets/dataset/test/image"
test_gt_dir = "datasets/dataset/test/GT"

# 创建测试数据集实例
test_dataset = CamouflageDataset(image_dir=test_image_dir, gt_dir=test_gt_dir, image_transform=image_transform, gt_transform=gt_transform)

# 创建测试 DataLoader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 特征提取
# 提取颜色直方图特征
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 提取LBP纹理特征
def extract_lbp_feature(image, radius=3, n_points=24):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # 归一化
    return hist

# 提取每张图像的特征
def extract_features(images):
    features = []
    for image in images:
        image = image.permute(1, 2, 0).cpu().numpy()
        # 转换为OpenCV格式
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 提取颜色直方图特征
        hist_feature = extract_color_histogram(image_bgr)
        # 提取LBP纹理特征
        lbp_feature = extract_lbp_feature(image_bgr)
        # 合并特征
        combined_feature = np.hstack([hist_feature, lbp_feature])
        features.append(combined_feature)
    return np.array(features)

"""
# 从 train_loader 中获取一批数据
for images, masks in train_loader:
    # 选择第一个样本
    image = images[6]
    mask = masks[6]
    # 将图像和掩码从张量转换为 NumPy 数组
    image = image.permute(1, 2, 0).numpy()  # 将通道维度放到最后
    mask = mask.squeeze().numpy()  # 去除多余的维度
    feature = extract_features(image)
    # 显示图像和掩码
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Mask')
    plt.show()
    break  # 只显示一个批次中的一个样本
"""

base_estimator = DecisionTreeClassifier(max_depth=1)
clf = AdaBoostClassifier(n_estimators=50, random_state=42)

features_list=[]
labels_list=[]
for iter, (images, masks) in enumerate(train_loader, start=1):
    print(f"Iteration: {iter}")
    batch_features = extract_features(images)
    batch_labels = (masks > 0).type(torch.int).flatten()
    features_list.append(batch_features)
    labels_list.append(batch_labels)

X_train, X_test, y_train, y_test = train_test_split(features_list, labels_list, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



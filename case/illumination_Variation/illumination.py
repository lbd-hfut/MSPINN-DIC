import os
import glob
import numpy as np
from PIL import Image
from scipy.io import savemat

# ==============================
# 固定随机数种子
# ==============================
np.random.seed(42)

# ==============================
# 文件夹
# ==============================
folder = r"C:/01project/SegPINN-DIC/case/speckle_quality/r1_n8000_g6"
save_folder = r"C:/01project/SegPINN-DIC/case/illumination_Variation"


# ==============================
# 读取图像
# ==============================
def read_image(path):

    img = Image.open(path).convert("F")
    img = np.array(img, dtype=np.float32)

    return img


# ==============================
# 保存图像
# ==============================
def save_image(img, path):

    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    Image.fromarray(img).save(path)


# ==============================
# 生成局部光斑 illumination
# ==============================
def generate_spot_field(H, W, A=0.3, sigma=120):

    x0 = np.random.uniform(0, W)
    y0 = np.random.uniform(0, H)

    x = np.arange(W)
    y = np.arange(H)

    X, Y = np.meshgrid(x, y)

    r2 = (X - x0)**2 + (Y - y0)**2

    a = 1 + A * np.exp(-r2 / (sigma**2))

    return a


# ==============================
# 添加 illumination
# ==============================
def apply_illumination(img, A, sigma):

    H, W = img.shape

    field = generate_spot_field(H, W, A, sigma)

    img_new = img * field

    return img_new, field


# ==============================
# 主程序
# ==============================
def main():

    bmp_files = sorted(glob.glob(os.path.join(folder, "*.bmp")))

    if len(bmp_files) < 2:
        print("Need at least two images (ref and def)")
        return

    ref_path = bmp_files[0]
    def_path = bmp_files[1]

    print("Reference image:", os.path.basename(ref_path))
    print("Deformed image :", os.path.basename(def_path))

    ref = read_image(ref_path)
    deformed = read_image(def_path)

    # 两种不同 illumination
    ref_new, ref_field = apply_illumination(ref, A=0.4, sigma=150)
    def_new, def_field = apply_illumination(deformed, A=0.3, sigma=200)
    roi = 255 * np.ones_like(ref_new, dtype=np.uint8)

    # 保存
    ref_save = os.path.join(save_folder, "001.bmp")
    def_save = os.path.join(save_folder, "002.bmp")
    roi_save = os.path.join(save_folder, "003.bmp")

    save_image(ref_new, ref_save)
    save_image(def_new, def_save)
    save_image(roi, roi_save)
    
    mat_save_path = os.path.join(save_folder, "illumination_fields.mat")
    savemat(mat_save_path, {
        "ref_illumination_field": ref_field,
        "def_illumination_field": def_field,
        "A_ref": 0.4,
        "sigma_ref": 150,
        "A_def": 0.5,
        "sigma_def": 200
    })

    print("\nSaved:")
    print(ref_save)
    print(def_save)


# ==============================
if __name__ == "__main__":
    main()
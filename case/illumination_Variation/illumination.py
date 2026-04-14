import os
import glob
import numpy as np
from PIL import Image

np.random.seed(42)

# ==============================
# 路径
# ==============================
folder = r"C:/01project/SegPINN-DIC/case/speckle_quality/r1_n8000_g6"

save_root = r"C:/01project/SegPINN-DIC/case/illumination_Variation"
global_folder = os.path.join(save_root, "global")
local_folder = os.path.join(save_root, "local")

os.makedirs(global_folder, exist_ok=True)
os.makedirs(local_folder, exist_ok=True)


# ==============================
# 读取图像
# ==============================
def read_image(path):
    img = Image.open(path).convert("F")
    return np.array(img, dtype=np.float32)


# ==============================
# 保存图像
# ==============================
def save_image(img, path):
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    Image.fromarray(img).save(path)


# ==============================
# 全局光照
# ==============================
def apply_global(img, alpha, delta):
    return alpha * img + delta


# ==============================
# 局部高斯光照
# ==============================
def generate_spot_field(H, W, A, sigma, mode="center"):
    if mode == "center":
        x0, y0 = W / 2, H / 2
    elif mode == "corner":
        x0, y0 = 0, 0
    elif mode == "mid":
        x0, y0 = W / 4, H / 4
    else:
        raise ValueError

    x = np.arange(W)
    y = np.arange(H)
    X, Y = np.meshgrid(x, y)

    r2 = (X - x0)**2 + (Y - y0)**2

    field = 1 + A * np.exp(-r2 / (2 * sigma**2))

    return field


def apply_local(img, A, sigma, mode):
    H, W = img.shape
    field = generate_spot_field(H, W, A, sigma, mode)
    return img * field


# ==============================
# 主程序
# ==============================
def main():

    bmp_files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    ref = read_image(bmp_files[0])
    deformed = read_image(bmp_files[1])

    roi = 255 * np.ones_like(ref, dtype=np.uint8)

    # ==========================
    # 第一组：global (9组)
    # ==========================
    delta_list = [0, 10, 15]
    alpha_list = [0.8, 1.0, 1.2]

    idx = 0
    for d in delta_list:
        for a in alpha_list:
            idx += 1

            case_folder = os.path.join(global_folder, f"case_{idx:02d}")
            os.makedirs(case_folder, exist_ok=True)

            ref_img = ref.copy()
            def_img = apply_global(deformed, a, d)

            save_image(ref_img, os.path.join(case_folder, "001.bmp"))
            save_image(def_img, os.path.join(case_folder, "002.bmp"))
            save_image(roi, os.path.join(case_folder, "003.bmp"))

    print("Global cases done (9)")

    # ==========================
    # 第二组：local (27组)
    # ==========================
    A_list = [0.1, 0.3, 0.5]
    sigma_list = [32, 64, 128]
    mode_list = ["center", "corner", "mid"]

    idx = 0
    for A in A_list:
        for sigma in sigma_list:
            for mode in mode_list:
                idx += 1

                case_folder = os.path.join(local_folder, f"case_{idx:02d}")
                os.makedirs(case_folder, exist_ok=True)

                ref_img = ref.copy()
                def_img = apply_local(deformed, A, sigma, mode)

                save_image(ref_img, os.path.join(case_folder, "001.bmp"))
                save_image(def_img, os.path.join(case_folder, "002.bmp"))
                save_image(roi, os.path.join(case_folder, "003.bmp"))

    print("Local cases done (27)")


# ==============================
if __name__ == "__main__":
    main()
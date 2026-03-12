import os
import jax
import glob
import numpy as np
import jax.numpy as jnp
from PIL import Image

from segpinndic.DIC_readImg import get_QK, form_bcoef, image_gradient_from_bcoef

def read_image(path):
    """读取灰度图并转为 jax array"""
    img = Image.open(path).convert("F")
    img = np.array(img, dtype=np.float32)
    return jnp.array(img)

def compute_gradient(img, degree=3, border=3):
    """利用 B-spline 系数计算梯度"""
    
    QK = get_QK(degree)

    bcoef = form_bcoef(img, degree, border)

    roi_mask = jnp.ones_like(img, dtype=bool)

    fx, fy = image_gradient_from_bcoef(
        bcoef,
        roi_mask,
        degree,
        border,
        QK
    )

    return fx, fy

def compute_SSE(img, degree=3, alpha=1.5):
    """
    计算 Speckle Spectral Efficiency
    """

    fx, fy = compute_gradient(img, degree)

    # 梯度能量
    g = fx**2 + fy**2

    # FFT
    G = jnp.fft.fftshift(jnp.fft.fft2(g))
    Sg = jnp.abs(G) ** 2

    H, W = g.shape

    kx = jnp.arange(-H//2, H//2)
    ky = jnp.arange(-W//2, W//2)

    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")

    k2 = KX**2 + KY**2

    # NTK 权重
    # Wntk = 1.0 / (k2 + 1) ** (alpha / 2)
    Wntk = 1.0 / jnp.exp(alpha * k2)

    SSE = jnp.sum(Sg * Wntk) / jnp.sum(Sg)

    return float(SSE)

# ------------------------------------------------
# 主程序
# ------------------------------------------------
def main(degree=3, alpha=1.5):

    folder = r"C:/01project/SegPINN-DIC/case/speckle_quality/"

    bmp_files = sorted(glob.glob(os.path.join(folder, "*.bmp")))

    if len(bmp_files) == 0:
        print("No BMP files found.")
        return

    print("Found", len(bmp_files), "images\n")

    results = []

    for path in bmp_files:

        img = read_image(path)

        sse = compute_SSE(img, degree=degree, alpha=alpha)

        name = os.path.basename(path)

        print(f"{name:25s}  SSE = {sse:.5f}")

        results.append(sse)

    print("\n----------------------------------")

    mean_sse = np.mean(results)

    print("Average SSE =", mean_sse)


# ------------------------------------------------

if __name__ == "__main__":
    main(degree=5, alpha=3.0)
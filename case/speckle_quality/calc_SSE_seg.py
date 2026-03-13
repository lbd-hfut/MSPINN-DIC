import os
import glob
import numpy as np
import jax.numpy as jnp
from PIL import Image

from segpinndic.DIC_readImg import get_QK, form_bcoef, image_gradient_from_bcoef


# ------------------------------------------------
# 读取图像
# ------------------------------------------------
def read_image(path):
    """读取灰度图并转为 JAX array"""
    img = Image.open(path).convert("F")
    img = np.array(img, dtype=np.float32)
    return jnp.array(img)


# ------------------------------------------------
# 计算 B-spline 梯度
# ------------------------------------------------
def compute_gradient(img, degree=3, border=3):

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


# ------------------------------------------------
# 根据梯度能量计算 SSE
# ------------------------------------------------
def compute_SSE_from_gradient(g, alpha=1.5):

    # FFT
    G = jnp.fft.fftshift(jnp.fft.fft2(g))
    Sg = jnp.abs(G) ** 2

    H, W = g.shape

    kx = jnp.arange(-H//2, H//2)
    ky = jnp.arange(-W//2, W//2)

    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")

    k2 = KX**2 + KY**2

    # NTK 权重
    Wntk = 1.0 / jnp.exp(alpha * k2)

    SSE = jnp.sum(Sg * Wntk) / jnp.sum(Sg)

    return SSE


# ------------------------------------------------
# 支持分区平均的 SSE
# ------------------------------------------------
def compute_SSE(img, degree=3, alpha=1.5, partition=(1, 1)):

    fx, fy = compute_gradient(img, degree)

    # 梯度能量
    g = fx**2 + fy**2

    nx, ny = partition

    H, W = g.shape

    # 保证能整除
    Hc = (H // nx) * nx
    Wc = (W // ny) * ny

    g = g[:Hc, :Wc]

    hx = Hc // nx
    hy = Wc // ny

    sse_list = []

    for i in range(nx):
        for j in range(ny):

            g_block = g[
                i*hx:(i+1)*hx,
                j*hy:(j+1)*hy
            ]

            sse_block = compute_SSE_from_gradient(g_block, alpha)

            sse_list.append(sse_block)

    sse_mean = jnp.mean(jnp.array(sse_list))

    return float(sse_mean)


# ------------------------------------------------
# 主程序
# ------------------------------------------------
def main(degree=5, alpha=3.0, partition=(4,4)):

    folder = r"C:/01project/SegPINN-DIC/case/speckle_quality/"

    bmp_files = sorted(glob.glob(os.path.join(folder, "*.bmp")))

    if len(bmp_files) == 0:
        print("No BMP files found.")
        return

    print("Found", len(bmp_files), "images\n")

    results = []

    for path in bmp_files:

        img = read_image(path)

        sse = compute_SSE(
            img,
            degree=degree,
            alpha=alpha,
            partition=partition
        )

        name = os.path.basename(path)

        print(f"{name:25s}  SSE = {sse:.6f}")

        results.append(sse)

    print("\n----------------------------------")

    mean_sse = np.max(results)

    print("Average SSE =", mean_sse)


# ------------------------------------------------
# 运行
# ------------------------------------------------
if __name__ == "__main__":

    main(
        degree=1,        # B-spline 阶数
        alpha=3.0,       # NTK权重衰减
        partition=(4,4)  # 分区数量
    )
from DIC_importlib import *

# ============================================
# 线程缓冲区 (用于存储中间计算结果)
# ============================================
class BufferManager:
    QK = None
    QKBQKT_def = None
    QKBQKT_ref = None
    fx = None
    fy = None
    refImg = None
    refImg_pad = None
    defImg = None
    defImg_pad = None
    mask = None
    mask_pad = None

def plus_power(x, p):
    return jnp.where(x > 0, x ** p, 0.0)

@partial(jax.jit, static_argnums=(1,))
def beta5_nth(x, n=0):
    coeffs = jnp.array([1, -6, 15, -20, 15, -6, 1])
    shifts = jnp.array([-3, -2, -1, 0, 1, 2, 3])
    factor = math.factorial(5) // math.factorial(5 - n)

    def body(c, s):
        return c * factor * plus_power(x + s, 5 - n)

    val = jnp.sum(jax.vmap(body)(coeffs, shifts), axis=0)
    return val / 120.0

@jax.jit
def get_QK():
    x = jnp.array([-2, -1, 0, 1, 2, 3])
    def row(n):
        return ((-1)**n) * beta5_nth(x, n) / math.factorial(n)
    return jnp.stack([row(n) for n in range(6)])

@jax.jit
def form_bcoef(img, border=3):
    img = jnp.pad(img, border, mode="edge")
    h, w = img.shape
    x_sample = jnp.array([-2, -1, 0, 1, 2])
    kernel_b = beta5_nth(x_sample, 0)

    def make_kernel(n):
        k = jnp.zeros(n)
        k = k.at[:3].set(kernel_b[2:])
        k = k.at[-2:].set(kernel_b[:2])
        return jnp.fft.fft(k)

    kx = make_kernel(w)
    ky = make_kernel(h)

    img = jnp.real(jnp.fft.ifft(jnp.fft.fft(img, axis=1) / kx, axis=1))
    img = jnp.real(jnp.fft.ifft(jnp.fft.fft(img, axis=0) / ky[:, None], axis=0))

    return img

@jax.jit
def get_QK_B_QKT(plot_bcoef, img, border=3):
    QK = BufferManager.QK
    QKT = QK.T
    offset = 2
    H, W = img.shape
    ys, xs = jnp.meshgrid(
        jnp.arange(H),
        jnp.arange(W),
        indexing="ij"
    )

    top  = ys + border - offset
    left = xs + border - offset

    dy = jnp.arange(6)[:, None]    # (6,1)
    dx = jnp.arange(6)[None, :]    # (1,6)

    blocks = plot_bcoef[
        top[..., None, None] + dy,
        left[..., None, None] + dx
    ]                               # (H, W, 6, 6)

    return jnp.einsum("ij,hwjk,kl->hwil", QK, blocks, QKT)


@jax.jit
def image_gradient_from_bcoef(
    ref_bcoef,        # (H+2b, W+2b)
    roi_mask,         # (H, W), bool or int  
    border=3
):
    QK = BufferManager.QK
    QKT = QK.T
    H, W = roi_mask.shape
    offset = 2

    ys, xs = jnp.meshgrid(
        jnp.arange(H),
        jnp.arange(W),
        indexing="ij"
    )

    top  = ys + border - offset
    left = xs + border - offset

    dy = jnp.arange(6)[:, None]
    dx = jnp.arange(6)[None, :]

    blocks = ref_bcoef[
        top[..., None, None] + dy,
        left[..., None, None] + dx
    ]                              # (H, W, 6, 6)

    M = jnp.einsum("ij,hwjk,kl->hwil", QK, blocks, QKT)

    fx = M[..., 0, 1]
    fy = M[..., 1, 0]

    return fx, fy

def build_seed_buffer_jax(img, mask):
    BufferManager.QK = get_QK()
    plot_bcoef = form_bcoef(img)
    BufferManager.fx, BufferManager.fy = image_gradient_from_bcoef(plot_bcoef, mask)
    BufferManager.QKBQKT_ref = get_QK_B_QKT(plot_bcoef, img)

def build_DIC_buffer_jax(img):
    plot_bcoef = form_bcoef(img)
    BufferManager.QKBQKT_def = get_QK_B_QKT(plot_bcoef, img)
    
    
class ImgDataset:
    def __init__(self, DIC_config, Seed_config):
        image_files = np.array([
            x.path for x in os.scandir(DIC_config.input_dir)
            if x.name.lower().endswith((".bmp", ".png", ".jpg", ".tiff"))
        ])
        if image_files.size == 0:
            raise FileNotFoundError(
                f"[ERROR] No image files found in directory: {DIC_config.input_dir} "
                "(supported: .bmp, .png, .jpg, .tiff)"
            )
        image_files.sort()
        
        # 参考图 & mask
        self.rfimage_file = image_files[0]
        self.mask_file = image_files[-1]
        # 将参考图像和mask图像存入 BufferManager
        self.coarse_subset_radius = Seed_config.coarse_subset_radius
        BufferManager.refImg = self.open_image(self.rfimage_file)
        BufferManager.refImg_pad = jnp.pad(
            BufferManager.refImg,
            pad_width=self.coarse_subset_radius,
            mode='constant',
            constant_values=False
        )
        mask_bin = self.open_image(self.mask_file) > 0
        labeled, num_labels = label(mask_bin)
        if num_labels == 0:
            raise RuntimeError("Mask 中没有前景像素！")
        ROI_list, ROI_list_pad = [], []
        for comp_id in range(1, num_labels + 1):
            roi_i = (labeled == comp_id)
            roi_i = jnp.array(roi_i, dtype=jnp.bool_)
            roi_i_pad = jnp.pad(
                roi_i,
                pad_width=self.coarse_subset_radius,
                mode='constant',
                constant_values=False
            )
            # 创建单连通域 ROI
            ROI_list.append(roi_i)
            ROI_list_pad.append(roi_i_pad)
        BufferManager.mask = ROI_list
        BufferManager.mask_pad = ROI_list_pad
        build_seed_buffer_jax(BufferManager.refImg, mask_bin)

        # 变形图像序列
        self.dfimage_files = image_files[1:-1]
        self.method = DIC_config.interpolation

    def __len__(self):
        return len(self.dfimage_files)

    def get_image(self, idx):
        """只负责取图，不产生副作用"""
        BufferManager.defImg = self.open_image(self.dfimage_files[idx])
        BufferManager.defImg_pad = jnp.pad(
            BufferManager.defImg,
            pad_width=self.coarse_subset_radius,
            mode='constant',
            constant_values=False
        )
        if self.method == "bspline":
            build_DIC_buffer_jax(BufferManager.defImg)

    @staticmethod
    def open_image(name):
        img = Image.open(name).convert("L")
        return jnp.array(img, dtype=jnp.float32)
    
if __name__ == "__main__":
    # refImg = jnp.array(
    #     Image.open("./case/case9/image/img_000.bmp").convert('L'), 
    #     dtype=jnp.float32
    # ) / 255
    
    # defImg =jnp.array(
    #     Image.open("./case/case9/image/img_001.bmp").convert('L'), 
    #     dtype=jnp.float32
    # ) / 255
    
    # mask =jnp.array(
    #     Image.open("./case/case9/image/img_002.bmp").convert('L'), 
    #     dtype=jnp.float32
    # ) > 0 
    
    # build_seed_buffer_jax(refImg, mask)
    # build_DIC_buffer_jax(defImg)
    
    from DIC_config import seed_config_txt, DIC_config_txt
    seed_config_path = "Seed_Configuration.txt"
    dic_config_path = "PINN-DIC-2D.txt"

    DIC_config = DIC_config_txt(dic_config_path)
    Seed_config = seed_config_txt(seed_config_path)
    
    ImgData = ImgDataset(DIC_config, Seed_config)
    ImgData.get_image(0)

from DIC_importlib import *
from DIC_readImg import BufferManager

# ============================================
# 数据结构定义
# ============================================
class RectangleROI:
    """圆形感兴趣区域"""
    def __init__(self):
        self.x = 0              # 中心x坐标
        self.y = 0              # 中心y坐标
        self.radius = 0         # 半径
        self.mask = None        # 掩膜 (2D array)
        self.region = None      # 区域信息
        self.X_flat = None
        self.Y_flat = None
        
# ============================================
# 自动生成种子点位置
# ============================================
class seed_generator:
    def __init__(self, config):
        
        self.config = config
        self.ROI_LIST = BufferManager.mask
        self.seed_points_list = self.sample_kmeans() # np.darray: shape[N,2]
        
    def sample_kmeans(self):
        n_points = self.config.seeds_number
        seed_points_list = []
        for mask in self.ROI_LIST:
            ys, xs = np.nonzero(mask)
            pts = np.column_stack([xs, ys])          # (N,2) 所有 ROI 前景像素坐标
            if len(pts) < n_points:
                raise ValueError("ROI 中像素数量不足")
            # --- 特殊情况：n=1 ---
            if n_points == 1:
                idx = np.random.randint(0, len(pts))
                x, y = pts[idx]
                return [(int(x), int(y))]
            # --- 正常情况：K-means 聚类 ---
            kmeans = KMeans(n_clusters=n_points, n_init='auto').fit(pts)
            centers = np.rint(kmeans.cluster_centers_).astype(int)
            # --- 处理每个中心点 ---
            H, W = mask.shape
            seed_points = []
            for x, y in centers:
                # 中心点合法（落在 ROI 内）
                if 0 <= x < W and 0 <= y < H and mask[y, x]:
                    seed_points.append((int(x), int(y)))
                    continue
                # 中心点无效 → 随机从 ROI 内重新采样
                idx = np.random.randint(0, len(pts))
                xr, yr = pts[idx]
                seed_points.append((int(xr), int(yr)))
            seed_points_list.append(jnp.array(seed_points, dtype=jnp.int32))
        return seed_points_list

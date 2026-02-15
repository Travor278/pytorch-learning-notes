# 08_tensor_operations_and_gpu.py
# 用途：张量操作大全 + GPU 加速入门
#
# PyTorch 本质就是"能用 GPU 跑的 NumPy + 自动求导"。
# 这个文件覆盖你日常最常用的 tensor 操作，以及怎么把计算搬到 GPU 上。

import torch
import time

# ========== 张量创建 ==========
print("=== 张量创建方式 ===")

# 从列表创建
a = torch.tensor([1, 2, 3])
print(f"从列表: {a}, dtype={a.dtype}")

# 指定类型
b = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"指定float32: {b}, dtype={b.dtype}")

# 常用工厂函数
zeros = torch.zeros(2, 3)         # 全0
ones = torch.ones(2, 3)           # 全1
rand = torch.rand(2, 3)           # [0,1) 均匀分布
randn = torch.randn(2, 3)         # 标准正态分布
arange = torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
eye = torch.eye(3)                # 3x3 单位矩阵

print(f"arange: {arange}")
print(f"linspace: {linspace}")
print(f"eye:\n{eye}")

# 跟某个 tensor 同形状
like = torch.zeros_like(rand)     # 跟 rand 一样的形状，全0
print(f"zeros_like 形状: {like.shape}")

print()

# ========== 形状操作（reshape / view / permute）==========
print("=== 形状操作 ===")

x = torch.arange(12)  # [0,1,2,...,11]
print(f"原始: {x}, shape={x.shape}")

# reshape / view：改变形状
# view 要求内存连续，reshape 更通用
r1 = x.view(3, 4)
r2 = x.reshape(3, 4)
print(f"view(3,4):\n{r1}")

# -1 表示自动推断
r3 = x.view(2, -1)  # 2行，列数自动算 = 6
print(f"view(2,-1):\n{r3}")

# squeeze / unsqueeze：去掉/添加维度
t = torch.tensor([1, 2, 3])  # shape: (3,)
t1 = t.unsqueeze(0)   # shape: (1, 3) —— 加了 batch 维
t2 = t.unsqueeze(1)   # shape: (3, 1) —— 变成列向量
print(f"unsqueeze(0): shape={t1.shape}")
print(f"unsqueeze(1): shape={t2.shape}")

t3 = t1.squeeze(0)    # 把那个 1 的维度去掉
print(f"squeeze(0): shape={t3.shape}")

# permute：调整维度顺序
# 比的图像数据从 (H, W, C) 转成 (C, H, W)
img = torch.randn(224, 224, 3)  # numpy/opencv 的格式：(H, W, C)
img_pytorch = img.permute(2, 0, 1)  # PyTorch 要的格式：(C, H, W)
print(f"图像维度转换: {img.shape} -> {img_pytorch.shape}")

# transpose：交换两个维度
mat = torch.randn(3, 5)
mat_t = mat.transpose(0, 1)  # 或者 mat.T（矩阵转置的简写）
print(f"transpose: {mat.shape} -> {mat_t.shape}")

print()

# ========== 索引和切片 ==========
print("=== 索引和切片 ===")

x = torch.arange(12).reshape(3, 4).float()
print(f"x:\n{x}")

# 基本索引（跟 numpy 一样）
print(f"x[0] = {x[0]}")           # 第0行
print(f"x[:, 1] = {x[:, 1]}")     # 第1列
print(f"x[1, 2] = {x[1, 2]}")     # 第1行第2列

# bool 索引
mask = x > 5
print(f"x > 5 的元素: {x[mask]}")

# 花式索引
idx = torch.tensor([0, 2])
print(f"取第0行和第2行:\n{x[idx]}")

print()

# ========== 数学运算 ==========
print("=== 常用数学运算 ===")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 逐元素运算
print(f"a + b = {a + b}")         # 加
print(f"a * b = {a * b}")         # 逐元素乘（不是矩阵乘！）
print(f"a ** 2 = {a ** 2}")       # 平方

# 矩阵乘法（这三种写法等价）
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C1 = A @ B               # 推荐：简洁
C2 = torch.mm(A, B)      # mm = matrix multiply
C3 = torch.matmul(A, B)  # 最通用，支持 batch
print(f"矩阵乘法: ({A.shape}) @ ({B.shape}) = {C1.shape}")

# batch 矩阵乘法（在 Transformer attention 中很常见）
batch_A = torch.randn(8, 3, 4)  # 8个 3×4 矩阵
batch_B = torch.randn(8, 4, 5)  # 8个 4×5 矩阵
batch_C = torch.bmm(batch_A, batch_B)  # 8个 3×5 矩阵
print(f"batch矩阵乘法: {batch_A.shape} @ {batch_B.shape} = {batch_C.shape}")

# 聚合运算
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"sum = {x.sum()}")                    # 全部求和
print(f"sum(dim=0) = {x.sum(dim=0)}")        # 按列求和
print(f"sum(dim=1) = {x.sum(dim=1)}")        # 按行求和
print(f"mean = {x.mean()}")
print(f"max = {x.max()}, argmax = {x.argmax()}")

print()

# ========== 广播机制（Broadcasting）==========
print("=== 广播机制 ===")

# 跟 numpy 的规则一样
a = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
b = torch.tensor([10.0, 20.0, 30.0])     # (3,)

# a(3,1) + b(3,) → a(3,1) + b(1,3) → (3,3)
c = a + b
print(f"a.shape={a.shape}, b.shape={b.shape}")
print(f"a + b (广播后):\n{c}")

print()

# ========== GPU 加速 ==========
print("=== GPU 加速 ===")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    
    # 把 tensor 搬到 GPU 上
    x_cpu = torch.randn(1000, 1000)
    x_gpu = x_cpu.to(device)        # 方式1
    # x_gpu = x_cpu.cuda()          # 方式2（不推荐，灵活性差）
    
    print(f"CPU tensor: device={x_cpu.device}")
    print(f"GPU tensor: device={x_gpu.device}")
    
    # --- 速度对比 ---
    size = 4000
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    
    # 热身（第一次跑 GPU 有初始化开销）
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()  # 等 GPU 算完
    
    # CPU 矩阵乘法
    start = time.time()
    for _ in range(10):
        c_cpu = a_cpu @ b_cpu
    cpu_time = time.time() - start
    
    # GPU 矩阵乘法
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"\n{size}x{size} 矩阵乘法 x10:")
    print(f"  CPU: {cpu_time:.4f}s")
    print(f"  GPU: {gpu_time:.4f}s")
    print(f"  加速比: {cpu_time/gpu_time:.1f}x")
    
    # 注意：CPU 和 GPU 的 tensor 不能直接运算！
    # x_cpu + x_gpu  # 这会报错
    # 必须在同一个 device 上
    
    # 结果搬回 CPU
    result = c_gpu.cpu()  # 或 c_gpu.to('cpu')
    print(f"\n结果搬回 CPU: device={result.device}")
    
else:
    print("没有检测到 GPU，跳过 GPU 部分")
    print("如果你有 NVIDIA 显卡，请安装 CUDA 版本的 PyTorch:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

print()

# ========== tensor 和 numpy 的互转 ==========
print("=== tensor <-> numpy ===")
import numpy as np

# numpy -> tensor
np_arr = np.array([1, 2, 3])
t1 = torch.from_numpy(np_arr)     # 共享内存！修改一个另一个也变
t2 = torch.tensor(np_arr)         # 拷贝，不共享内存
print(f"from_numpy: {t1}")

# tensor -> numpy
t = torch.tensor([4.0, 5.0, 6.0])
n = t.numpy()         # 要求 tensor 在 CPU 上且没有 requires_grad
print(f"tensor.numpy(): {n}")

# 如果 tensor 在 GPU 上或有梯度，要先处理
t_grad = torch.tensor([1.0, 2.0], requires_grad=True)
n2 = t_grad.detach().cpu().numpy()  # 先 detach 再 cpu 再 numpy
print(f"有梯度的 tensor 转 numpy: {n2}")

# 共享内存的坑
np_arr[0] = 999
print(f"修改 numpy 后，from_numpy 的 tensor 也变了: {t1}")
# t2 不会变，因为是拷贝

print()
print("=== 总结 ===")
print("""
1. tensor 操作跟 numpy 几乎一样，学过 numpy 就会用
2. reshape/view/permute/squeeze 是形状操作的核心
3. @ 是矩阵乘法，* 是逐元素乘法，别搞混了
4. .to(device) 在 CPU/GPU 之间搬运数据
5. GPU 的优势在大矩阵运算上才明显，小计算反而比 CPU 慢（传输开销）
6. from_numpy 共享内存，torch.tensor 是拷贝，要注意
""")

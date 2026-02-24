# 08_tensor_operations_and_gpu.py
# 张量操作大全 + GPU 加速
#
# PyTorch tensor 在语义上等价于"支持自动微分的多维数组"。
# 底层基于 C++ 的 at::Tensor，通过 Storage（连续内存块）+ offset/strides/sizes
# 三元组描述数据布局。理解内存布局对于正确使用 view/reshape 以及避免
# 不必要的数据拷贝至关重要。

import torch
import time
import numpy as np

# ========== 张量创建 ==========
print("=== 张量创建 ===")

a = torch.tensor([1, 2, 3])                    # 从 Python 列表拷贝，默认 int64
b = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"int64:   {a}, dtype={a.dtype}")
print(f"float32: {b}, dtype={b.dtype}")

# 工厂函数（不拷贝数据，直接分配内存）
print(f"arange:   {torch.arange(0, 10, 2)}")           # 等差序列，不含右端点
print(f"linspace: {torch.linspace(0, 1, 5)}")           # 均匀分布，含两端点
print(f"eye:\n{torch.eye(3)}")                          # 单位矩阵
print(f"zeros_like shape: {torch.zeros_like(b).shape}") # 与 b 同形同 dtype
print()

# ========== 内存布局：连续性与 view/reshape 的关系 ==========
print("=== 内存布局（contiguous）===")
#
# 每个 tensor 由 Storage（线性内存）+ strides 描述：
#   strides[i] 表示沿第 i 维移动一步需跨越的元素个数。
# 连续 tensor（C-contiguous）的 strides 满足：strides[i] = strides[i+1] * sizes[i+1]。
#
# view() 要求底层内存连续；若非连续，必须先调用 .contiguous() 触发拷贝，
# 或直接使用 reshape()（内部会自动判断，必要时拷贝）。

x = torch.arange(12).reshape(3, 4)
print(f"原始 strides: {x.stride()}  （行主序，沿列方向 stride=1）")

x_t = x.t()   # 转置：逻辑上翻转 strides，不拷贝数据
print(f"转置 strides: {x_t.stride()}  （非连续：stride 顺序颠倒）")
print(f"转置是否连续: {x_t.is_contiguous()}")

# x_t.view(-1)  # ← 会 RuntimeError：非连续 tensor 不能 view
x_t_c = x_t.contiguous()   # 触发内存重排，产生新 Storage
print(f"contiguous 后: {x_t_c.is_contiguous()}")
x_t_c.view(-1)              # 此时可以 view
print()

# ========== 形状操作 ==========
print("=== 形状操作 ===")

x = torch.arange(12)
print(f"原始: {x.shape}")

r = x.reshape(3, 4)
print(f"reshape(3,4):\n{r}")

# squeeze / unsqueeze：增减大小为 1 的维度（不改变元素数量）
t = torch.tensor([1, 2, 3])     # shape: (3,)
print(f"unsqueeze(0): {t.unsqueeze(0).shape}  ← 增加 batch 维")
print(f"unsqueeze(1): {t.unsqueeze(1).shape}  ← 变列向量")
print(f"squeeze:      {t.unsqueeze(0).squeeze(0).shape}  ← 消除 size=1 维")

# permute：任意维度重排（不拷贝，仅修改 strides）
# 典型场景：PIL/OpenCV 读取 (H, W, C)，PyTorch 期望 (C, H, W)
img = torch.randn(224, 224, 3)
print(f"permute (H,W,C)→(C,H,W): {img.shape} → {img.permute(2,0,1).shape}")
print()

# ========== 索引与切片 ==========
print("=== 索引与切片 ===")

x = torch.arange(12).reshape(3, 4).float()
print(f"x:\n{x}")
print(f"x[0]:    {x[0]}")                        # 第 0 行（返回 1D view）
print(f"x[:,1]:  {x[:, 1]}")                      # 第 1 列
print(f"x[1,2]:  {x[1, 2].item()}")               # 标量

# 布尔索引（Boolean Masking）：返回满足条件的元素（1D，非矩阵）
print(f"x[x>5]:  {x[x > 5]}")

# 高级索引（Fancy Indexing）：用 index tensor 选取行/列
idx = torch.tensor([0, 2])
print(f"x[[0,2]]:\n{x[idx]}")                    # 取第 0、2 行
print()

# ========== 数学运算 ==========
print("=== 数学运算 ===")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# * 是 Hadamard 积（逐元素），@ 是矩阵乘（内积/矩阵乘法）
print(f"a * b（Hadamard）= {a * b}")
print(f"a @ b（内积）    = {a @ b}")             # 等价于 torch.dot(a,b)

# 矩阵乘法的三种等价写法
A = torch.randn(2, 3);  B = torch.randn(3, 4)
assert torch.allclose(A @ B, torch.mm(A, B))     # mm 仅支持 2D
assert torch.allclose(A @ B, torch.matmul(A, B)) # matmul 支持 broadcast + batch
print(f"矩阵乘 ({A.shape}) @ ({B.shape}) = {(A@B).shape}")

# bmm：批量矩阵乘，常见于 Multi-Head Attention 的 QKᵀ 计算
# batch_A: [B, seq, d_model/h] × batch_B: [B, d_model/h, seq] → [B, seq, seq]
batch_A = torch.randn(8, 3, 4)
batch_B = torch.randn(8, 4, 5)
print(f"bmm: {batch_A.shape} × {batch_B.shape} = {torch.bmm(batch_A, batch_B).shape}")

# 聚合运算
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f"sum(dim=0) = {x.sum(dim=0)}  ← 沿行方向折叠（按列聚合）")
print(f"sum(dim=1) = {x.sum(dim=1)}  ← 沿列方向折叠（按行聚合）")
print()

# ========== 广播机制（Broadcasting）==========
print("=== 广播机制 ===")
#
# 广播规则（与 NumPy 完全一致）：
#   从尾部维度对齐，逐维比较：
#     相同：无变化；
#     其中一个为 1：沿该维复制；
#     不同且都非 1：报错。
# 广播不产生数据拷贝，仅通过 stride=0 模拟（零拷贝虚复制）。
# 因此 a(3,1) + b(1,3) → (3,3) 的内存占用仍分别只有 3 个元素。

a = torch.tensor([[1.0], [2.0], [3.0]])     # (3, 1)
b = torch.tensor([10.0, 20.0, 30.0])        # (3,) → 广播为 (1, 3) → (3, 3)
c = a + b
print(f"a({a.shape}) + b({b.shape}) → c({c.shape})")
print(f"广播结果:\n{c}")
print()

# ========== GPU 加速 ==========
print("=== GPU 加速 ===")
#
# GPU（CUDA）相对于 CPU 的优势根源：
#   SIMD 宽度：GPU 拥有数千个流处理器（CUDA cores），
#   针对矩阵乘等"embarrassingly parallel"操作有数量级的吞吐优势。
# 注意事项：
#   ① 传输开销（PCIe bandwidth ≈ 16 GB/s）远低于 GPU 内部带宽（≈ 1 TB/s），
#      小计算任务传输开销可能大于计算收益；
#   ② torch.cuda.synchronize() 是精确计时的必要步骤（GPU 为异步执行）；
#   ③ CPU/GPU tensor 不可直接参与同一运算，需显式 .to(device) 迁移。

if torch.cuda.is_available():
    device = torch.device('cuda')
    props  = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"显存: {props.total_memory / 1024**3:.1f} GB")
    print(f"SM 数量: {props.multi_processor_count}")

    x_cpu = torch.randn(1000, 1000)
    x_gpu = x_cpu.to(device)    # .to(device) 优于 .cuda()，设备无关

    print(f"CPU tensor device: {x_cpu.device}")
    print(f"GPU tensor device: {x_gpu.device}")

    # ---- CPU vs GPU 矩阵乘速度对比 ----
    SIZE = 4096
    a_cpu = torch.randn(SIZE, SIZE)
    b_cpu = torch.randn(SIZE, SIZE)
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # GPU 预热（消除 JIT 编译和上下文初始化的一次性开销）
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        c_cpu = a_cpu @ b_cpu
    cpu_time = time.perf_counter() - start

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()   # 等待 GPU 异步操作完成后再计时结束
    gpu_time = time.perf_counter() - start

    print(f"\n{SIZE}×{SIZE} 矩阵乘 ×10：CPU={cpu_time:.3f}s，GPU={gpu_time:.3f}s，"
          f"加速 {cpu_time/gpu_time:.1f}×")

    # 结果搬回 CPU（用于后处理、numpy 互转等）
    result = c_gpu.cpu()
    print(f"结果迁回 CPU: {result.device}")
else:
    print("未检测到 CUDA GPU，跳过 GPU 部分。")
    print("安装 CUDA 版 PyTorch：")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")

print()

# ========== Tensor ↔ NumPy 互转 ==========
print("=== Tensor ↔ NumPy ===")
#
# torch.from_numpy(arr)：零拷贝（共享底层内存），修改其一另一个同步变化。
#   限制：① 仅支持 CPU tensor；② numpy array 必须是 C-contiguous。
# torch.tensor(arr)  ：数据拷贝，产生独立 tensor，不共享内存。
#
# tensor.numpy()：同样零拷贝，返回共享内存的 ndarray。
#   前提：tensor 在 CPU 上且 requires_grad=False（否则需 .detach()）。
#
# 有梯度 tensor 转 numpy 的标准写法：t.detach().cpu().numpy()
#   ① detach()：脱离计算图，消除 requires_grad
#   ② .cpu()   ：确保在 CPU 上
#   ③ .numpy() ：零拷贝转 ndarray

np_arr = np.array([1, 2, 3])
t_shared = torch.from_numpy(np_arr)    # 共享内存
t_copy   = torch.tensor(np_arr)        # 拷贝

# 演示共享内存的影响
np_arr[0] = 999
print(f"修改 numpy 原数组后：")
print(f"  from_numpy（共享）: {t_shared}  ← 同步变化")
print(f"  torch.tensor（拷贝）: {t_copy}  ← 独立，不变")

# 有梯度 tensor → numpy 的安全写法
t_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
n = t_grad.detach().cpu().numpy()
print(f"\n有梯度 tensor → numpy: {n}")

print()
print("=== 总结 ===")
print("""
1. strides 决定内存布局；view 要求连续，reshape 不要求（必要时触发拷贝）
2. permute/transpose 不拷贝数据，仅调整 strides（结果通常非连续）
3. @ 是矩阵乘，* 是 Hadamard 积；bmm 用于批量矩阵乘（如 Attention）
4. 广播通过 stride=0 实现零拷贝虚复制，理解它可避免意外的形状错误
5. .to(device) 优于 .cuda()，前者设备无关；GPU 计时必须 synchronize()
6. from_numpy 共享内存（高效但需注意副作用）；torch.tensor 产生独立拷贝
""")

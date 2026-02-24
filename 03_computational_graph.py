# 03_computational_graph.py
# 深入理解 PyTorch 动态计算图（Dynamic Computational Graph）
#
# 计算图本质是一个有向无环图（DAG）：
#   节点（Node）—— 运算（op），如加、乘、ReLU
#   边（Edge）   —— 数据流（tensor），携带形状与 dtype 信息
#
# PyTorch 采用"define-by-run"（动态图/eager 模式）范式：
# 计算图在前向传播执行时即时构建，backward() 结束后默认释放。
# 这与 TensorFlow 1.x / Theano 的"define-and-run"（静态图）相对立。
# 动态图的核心优势：Python 原生控制流（if/for/while）可直接参与图结构，
# 使变长序列、递归网络、元学习（MAML）等场景的实现大幅简化。

import torch

# ========== grad_fn：每个节点记录其"来源运算" ==========
#
# 每当一个 requires_grad 参与的运算产生新 tensor，
# PyTorch 就在该 tensor 上挂一个 grad_fn 对象，
# 内部持有指向输入 tensor 的弱引用（weak reference）和局部梯度函数。
# backward() 本质是一次拓扑排序后的深度优先遍历，
# 对每个 grad_fn 调用 accumulate_grad 将结果写入叶子节点的 .grad。

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a + b        # grad_fn = AddBackward0
d = a * b        # grad_fn = MulBackward0
e = c + d        # grad_fn = AddBackward0（依赖 c, d）
f = e.mean()     # grad_fn = MeanBackward0

print("=== grad_fn（每个非叶子 tensor 的来源运算）===")
print(f"a.grad_fn = {a.grad_fn}")   # None：叶子节点无来源
print(f"c.grad_fn = {c.grad_fn}")   # AddBackward0
print(f"d.grad_fn = {d.grad_fn}")   # MulBackward0
print(f"e.grad_fn = {e.grad_fn}")   # AddBackward0
print(f"f.grad_fn = {f.grad_fn}")   # MeanBackward0
print()

# ---- 叶子节点 vs 中间节点 ----
# "叶子节点"（is_leaf=True）：由用户直接创建（如模型参数 w, b）。
# "中间节点"（is_leaf=False）：由运算产生，其 .grad 默认在 backward 后被释放，
# 以避免 O(N) 的额外显存开销。如需保留，见 retain_grad() 和 04_hooks.py。
print("=== 叶子节点判断 ===")
print(f"a 是叶子节点: {a.is_leaf}")   # True
print(f"c 是叶子节点: {c.is_leaf}")   # False
print()

f.backward()
print("=== backward() 后叶子节点的梯度 ===")
# f = (a+b + a*b) = e，mean() 对标量无效果
# ∂f/∂a = ∂(a+b)/∂a + ∂(a*b)/∂a = 1 + b = 1 + 3 = 4
# ∂f/∂b = ∂(a+b)/∂b + ∂(a*b)/∂b = 1 + a = 1 + 2 = 3
print(f"a.grad = {a.grad}  （期望 4 = 1 + b）")
print(f"b.grad = {b.grad}  （期望 3 = 1 + a）")

# 中间节点 c 的 .grad 为 None，访问时触发 UserWarning（这是正常行为）
print(f"c.grad = {c.grad}   ← 非叶子节点梯度默认不保留（UserWarning 属预期行为）")
print()

# ========== 动态图：每次前向传播独立构建 ==========
#
# 每次执行 Python 代码，PyTorch 都从零建立一张新图。
# 这意味着：图的拓扑结构本身可以是输入数据的函数。
# 典型场景：① NLP 中不同长度的句子走不同展开深度的 RNN；
#           ② 强化学习中依据当前状态决定计算路径；
#           ③ Neural ODE / 递归网络（树结构）。

print("=== 动态图：图结构随运行时数据分支变化 ===")

x = torch.tensor(1.5, requires_grad=True)

for i in range(3):
    # 计算图的结构取决于当前 x 的值——静态图框架无法直接表达此逻辑
    if x.item() > 1.0:
        y = x ** 2          # 此轮图：x → (pow) → y，局部梯度 = 2x
    else:
        y = x * 3           # 此轮图：x → (mul) → y，局部梯度 = 3

    y.backward()
    print(f"轮次 {i}: x={x.data:.4f}, y={y.data:.4f}, "
          f"grad={x.grad.item():.4f}, 分支={'x²' if x.item() > 1.0 else '3x'}")

    with torch.no_grad():
        x -= 0.5 * x.grad
        x.grad.zero_()

print()

# ========== retain_graph：让图在一次 backward 后存活 ==========
#
# 默认行为：backward() 遍历 DAG 的同时释放中间激活值（节省显存）。
# retain_graph=True 阻止释放，用于：
#   ① 对同一 loss 分别关于不同参数子集求梯度；
#   ② 高阶梯度计算（需配合 create_graph=True）；
#   ③ 某些 GAN 训练中的双路反向传播。
# 注意：多次 backward 时梯度会累加，需在合适位置手动清零。

print("=== retain_graph 演示 ===")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3   # y = x³，∂y/∂x = 3x² = 12

y.backward(retain_graph=True)
print(f"第 1 次 backward: x.grad = {x.grad}  （期望 12）")

y.backward(retain_graph=True)
print(f"第 2 次 backward（梯度累加）: x.grad = {x.grad}  （12 + 12 = 24）")

x.grad.zero_()
y.backward()  # 此次不保留图
print(f"清零后第 3 次 backward: x.grad = {x.grad}  （期望 12）")
print()

# ========== detach()：从图中剪断节点 ==========
#
# detach() 返回与原 tensor 共享底层存储但不属于任何计算图的新 tensor。
# 经典用例：
#   GAN —— 训练判别器时将生成器输出 detach，避免梯度流入生成器；
#   Target Network（DQN）—— 用 detach 的网络输出计算 Bellman 目标；
#   可视化 / 日志 —— 不希望 .item() 之外的操作污染计算图。
#
# detach() vs .data：
#   detach() 受版本计数器（version counter）保护，对 detach 后的 tensor
#   进行 in-place 修改再访问原 tensor 时会正确报错；
#   .data 绕过版本计数器，静默地破坏图的一致性，风险更高。

print("=== detach()：从计算图剪断 ===")

x = torch.tensor(3.0, requires_grad=True)
y = x * 2        # y ∈ 计算图，requires_grad=True
z = y.detach()   # z 与 y 共享存储，但 requires_grad=False

print(f"y = {y},  y.requires_grad = {y.requires_grad}")
print(f"z = {z},  z.requires_grad = {z.requires_grad}")
print(f"y 和 z 指向相同底层数据: {y.data_ptr() == z.data_ptr()}")

# 经由 y 的后续运算梯度可传回 x；经由 z 则不能
w = y * 5
w.backward()
print(f"经由 y: x.grad = {x.grad}  （期望 2×5 = 10）")

print()
print("=== 核心要点总结 ===")
print("""
1. 计算图是 DAG：边为 tensor，节点为运算（grad_fn）
2. define-by-run：每次前向传播重建图，图结构可随数据分支变化
3. 仅叶子节点的 .grad 默认保留；中间节点梯度在 backward 后释放
4. retain_graph=True 阻止图被释放，但多次 backward 会累加梯度
5. detach() 是安全地从图中"剪断"的首选方式，优于直接访问 .data
""")

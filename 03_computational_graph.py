# 03_computational_graph.py
# 用途：深入理解 PyTorch 动态计算图（Dynamic Computational Graph）
# 
# 很多人用 PyTorch 写了一年代码，都没真正搞明白计算图到底是什么。
# 这个文件带你把计算图"看"出来。

import torch

# ========== 什么是计算图？ ==========
# 
# 你每次写 c = a + b，PyTorch 不仅算出 c 的值，
# 还偷偷在后台画了一条线：a → (+) → c, b → (+) → c
# 这条线就是计算图。backward() 就是沿着这条线往回走，算梯度。

# --- 例1：看看 grad_fn ---
# grad_fn 就是计算图上每个节点的"来源"信息

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a + b        # c 是怎么来的？加法
d = a * b        # d 是怎么来的？乘法
e = c + d        # e 是怎么来的？加法
f = e.mean()     # f 是怎么来的？求均值（虽然只有一个元素，但还是会记录）

print("=== 计算图的 grad_fn（每个节点是怎么来的）===")
print(f"a.grad_fn = {a.grad_fn}")   # None，因为 a 是叶子节点（用户创建的）
print(f"c.grad_fn = {c.grad_fn}")   # AddBackward0
print(f"d.grad_fn = {d.grad_fn}")   # MulBackward0
print(f"e.grad_fn = {e.grad_fn}")   # AddBackward0
print(f"f.grad_fn = {f.grad_fn}")   # MeanBackward0
print()

# --- 叶子节点 vs 非叶子节点 ---
# "叶子节点"就是你自己创建的 tensor（比如模型参数 w, b）
# "非叶子节点"是通过运算得到的中间结果
print("=== 叶子节点判断 ===")
print(f"a 是叶子节点: {a.is_leaf}")   # True
print(f"c 是叶子节点: {c.is_leaf}")   # False，c 是 a+b 算出来的
print(f"e 是叶子节点: {e.is_leaf}")   # False
print()

# 重要：默认只有叶子节点的 .grad 会被保留！
# 中间节点的梯度算完就扔了（省内存）
# 后面的 04_hooks.py 会教你怎么拿到中间节点的梯度

f.backward()
print("=== backward() 之后 ===")
print(f"a.grad = {a.grad}")   # ∂f/∂a = ∂(a+b+a*b)/∂a = 1 + b = 1 + 3 = 4
print(f"b.grad = {b.grad}")   # ∂f/∂b = 1 + a = 1 + 2 = 3

# c 是非叶子节点，梯度默认不保留
# 下面这行会触发一个 UserWarning —— 这不是报错，只是 PyTorch 在提醒你：
# "c 不是叶子节点，它的 .grad 不会被 backward() 填充"
# 这正是我们要演示的：非叶子节点拿不到梯度（想拿的话看 04_hooks.py）
print(f"c.grad = {c.grad}")   # None！有警告是正常的

print()

# ========== 动态图 vs 静态图 ==========
#
# TensorFlow 1.x 用静态图：先画好整张图，然后 session.run()
# PyTorch 用动态图：每次前向传播都重新建图，backward() 后图就没了
#
# 动态图的好处：
# 1. 图的结构可以随数据变化（比如 NLP 中句子长度不同）
# 2. 可以用 Python 的 if/for 控制流，非常灵活
# 3. debug 方便，跟普通 Python 代码一样单步调试

# --- 例2：动态图的意思 —— 每次 forward 图可以不一样 ---

print("=== 动态图演示：图的结构随数据变化 ===")

x = torch.tensor(1.5, requires_grad=True)

for i in range(3):
    # 每轮循环，计算图的结构都可以不同
    if x.item() > 1.0:
        y = x ** 2        # x > 1 走这条路
    else:
        y = x * 3         # x <= 1 走这条路
    
    y.backward()
    print(f"轮次 {i}: x={x.data:.4f}, y={y.data:.4f}, grad={x.grad.item():.4f}, "
          f"走的分支: {'x²' if x.item() > 1.0 else '3x'}")
    
    # 更新 x（让它变小，看看会不会走不同分支）
    with torch.no_grad():
        x -= 0.5 * x.grad
        x.grad.zero_()

print()

# ========== retain_graph 的用法 ==========
# 
# 默认 backward() 之后计算图就被销毁了。
# 如果你需要对同一个 loss 做两次 backward（比较少见），
# 就需要 retain_graph=True

print("=== retain_graph 演示 ===")

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3   # y = x³

# 第一次 backward，保留图
y.backward(retain_graph=True)
print(f"第一次 backward: x.grad = {x.grad}")  # 3 * x² = 12

# 注意！梯度会累加！所以 x.grad 现在是 12
# 如果再来一次 backward...
y.backward(retain_graph=True)
print(f"第二次 backward（梯度累加了）: x.grad = {x.grad}")  # 12 + 12 = 24

# 清零后再来
x.grad.zero_()
y.backward()  # 这次不保留图了
print(f"清零后第三次 backward: x.grad = {x.grad}")  # 12

print()

# ========== detach() —— 从计算图上"剪断" ==========
#
# 有时候你想拿到某个中间结果的数值，但不想它影响梯度计算
# 比如做 GAN 训练时，生成器的输出传给判别器，
# 但更新判别器时不想梯度传回生成器

print("=== detach() 演示 ===")

x = torch.tensor(3.0, requires_grad=True)
y = x * 2        # y 在计算图上
z = y.detach()    # z 就是 y 的值（6.0），但跟计算图无关了

print(f"y = {y}, y.requires_grad = {y.requires_grad}")
print(f"z = {z}, z.requires_grad = {z.requires_grad}")
print(f"y 和 z 的值相同: {y.item() == z.item()}")
print(f"但 z 不在计算图中，对 z 做任何操作都不会影响 x 的梯度")

# 用 z 算出来的东西，梯度传不回 x
w = z * 5
# w.backward()  # 如果取消注释，会报错，因为 z 没有 grad_fn

# 而用 y 算出来的东西，梯度能传回 x
w = y * 5
w.backward()
print(f"通过 y 计算的 w 对 x 的梯度: x.grad = {x.grad}")  # 2*5 = 10

print()
print("=== 总结 ===")
print("""
1. 每次前向传播 PyTorch 建图，backward() 后图销毁 —— 这就是"动态图"
2. 只有叶子节点的 .grad 会保留（中间结果的梯度默认丢弃）
3. grad_fn 告诉你每个 tensor 是怎么算出来的
4. detach() 是从图上剪断的手术刀
5. retain_graph=True 让图活过一次 backward()，但一般用不到
""")

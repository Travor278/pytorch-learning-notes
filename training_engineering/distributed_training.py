# distributed_training.py
# 分布式训练：DDP 原理与单机多卡实践
#
# 主题：
#   1. DataParallel vs DistributedDataParallel 的本质区别
#   2. DDP 的 all-reduce 梯度同步原理（Ring-AllReduce）
#   3. 单机多卡 DDP 启动范式
#   4. 混合精度 + DDP 的组合写法
#   5. 常见坑与调试技巧
#
# 参考：Pytorch DDP 论文: Li et al., "PyTorch Distributed: Experiences on
#       Accelerating Data Parallel Training", VLDB 2020.

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# ========== 1. DataParallel vs DDP ==========
print("=== DataParallel vs DistributedDataParallel ===")
print("""
DataParallel（DP）— 简单但低效：
  - 单进程多线程，受 Python GIL 限制
  - 主 GPU 做梯度聚合，存在显存/计算不均衡（主卡热点）
  - 通信瓶颈：所有梯度先汇聚到 GPU 0，再广播更新后的参数

DistributedDataParallel（DDP）— 生产推荐：
  - 每个 GPU 一个独立进程，无 GIL 争用
  - 梯度通过 Ring-AllReduce 对等同步（无主卡瓶颈）
  - 通信与反向传播重叠（overlap）：每层梯度算完立刻开始 all-reduce，
    不等整个 backward 结束，大幅减少通信等待时间
  - 支持多节点（Multi-node）扩展
""")

# ========== 2. Ring-AllReduce 简述 ==========
print("=== Ring-AllReduce 梯度同步原理 ===")
print("""
N 张 GPU 排成一个环，共进行 2(N-1) 步：

  Reduce-Scatter（前 N-1 步）：
    每张卡将自己的梯度分成 N 份，
    每步向下一张卡发送一份，并将接收到的份累加到自己对应的段。
    结束后，每张卡上有一段"完整求和"的梯度。

  AllGather（后 N-1 步）：
    每步将自己持有的完整段发送给下一张卡，
    最终每张卡都拥有全部参数的完整梯度均值。

时间复杂度：O(L) 与 GPU 数量 N 无关（L = 参数量），
带宽利用率趋近于 (N-1)/N —— N 越大越接近 100%。
这使得 DDP 在数十乃至数百卡时仍保持良好的线性扩展性。
""")

# ========== 3. DDP 训练函数（每个进程执行）==========

def train_ddp(rank, world_size):
    """
    rank      : 当前进程的 GPU 编号（0 到 world_size-1）
    world_size: 总 GPU 数
    """
    # 初始化进程组：nccl 是 GPU 通信后端（CPU 调试时用 gloo）
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(
        backend    = backend,
        rank       = rank,
        world_size = world_size
    )

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # 每个进程在自己的 GPU 上创建模型，然后用 DDP 包装
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ).to(device)

    # DDP 包装：自动在 backward 中插入梯度 all-reduce hook
    # find_unused_parameters=False（默认）：前向中所有参数都必须参与计算图
    # find_unused_parameters=True：允许部分参数在某些 forward 中不使用（开销更大）
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    # DistributedSampler：确保每张卡看到不同的数据子集
    # 完整数据集 → 每张卡看到 1/world_size 的数据（无重叠）
    # 这里用随机数据模拟
    torch.manual_seed(rank)  # 不同 rank 用不同种子，确保数据不一致

    for step in range(5):
        x = torch.randn(8, 32, device=device)
        y = torch.randint(0, 10, (8,), device=device)

        optimizer.zero_grad()

        # AMP + DDP 组合：autocast 套在 DDP 模型调用外层
        with torch.amp.autocast(device.type,
                                 enabled=torch.cuda.is_available()):
            logits = ddp_model(x)
            loss   = criterion(logits, y)

        loss.backward()
        # 注意：DDP 的梯度 all-reduce 在 backward 内部自动触发，
        #       optimizer.step() 时各卡梯度已经同步
        optimizer.step()

        if rank == 0:  # 只有 rank 0 打印，避免重复输出
            print(f"  [DDP] step={step+1}, loss={loss.item():.4f}")

    # 销毁进程组
    dist.destroy_process_group()

# ========== 4. 单机多卡启动（mp.spawn）==========
print("=== DDP 训练启动 ===")

def run_ddp():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 2:
        print(f"检测到 {world_size} 张 GPU，以单卡 DDP 模式运行（模拟分布式环境）")

    # 设置 rendezvous 地址（单机用 localhost + 任意空闲端口）
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # mp.spawn 会 fork world_size 个进程，每个进程执行 train_ddp(rank, world_size)
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

# 仅在直接运行时启动（避免 import 时触发）
if __name__ == '__main__':
    run_ddp()
    print()
    print("=== DDP 使用规范 ===")
    print("""
1. 启动方式：mp.spawn（代码内） 或 torchrun（命令行，推荐生产使用）
   torchrun --nproc_per_node=4 distributed_training.py

2. DistributedSampler 必须配合使用，每个 epoch 调用 sampler.set_epoch(epoch)
   确保各卡间数据随机打乱但不重叠

3. 模型保存：只在 rank 0 保存，torch.save(ddp_model.module.state_dict(), ...)
   DDP 包装后访问原始模型用 .module 属性

4. 梯度 all-reduce 默认在 backward 中自动完成；若需手动控制同步边界，
   用 ddp_model.no_sync() 上下文管理器跳过中间步的同步（梯度累积场景）

5. find_unused_parameters=True 会扫描参数图，开销大；
   模型结构固定时保持 False
""")
else:
    # 非主程序时，仅展示概念，不实际启动多进程
    print("（DDP 需要以 __main__ 直接运行，当前作为模块 import，跳过 spawn）")
    print()
    print("=== 要点速览 ===")
    print("""
- DDP vs DP：DDP 多进程 + Ring-AllReduce，DP 单进程 + 主卡聚合，生产环境用 DDP
- Ring-AllReduce：O(L) 通信，带宽利用率 (N-1)/N，N 越大越高效
- 反向传播中每层梯度算完即触发 all-reduce（与下一层的 backward 重叠）
- AMP + DDP：autocast 套在 ddp_model(...) 外层；GradScaler.unscale_ 在 step 前
- torchrun 是推荐的多进程启动方式（PyTorch 1.9+ 替代 torch.distributed.launch）
""")

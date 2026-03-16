#%% # 嵌套 Logit vs Luce/Softmax — FashionMNIST
# FashionMNIST 的天然类别层次：
#   Nest 0 上装 Tops     : T-shirt(0), Pullover(2), Dress(3), Coat(4), Shirt(6)
#   Nest 1 鞋履 Footwear : Sandal(5), Sneaker(7), Ankle boot(9)
#   Nest 2 其他 Others   : Trouser(1), Bag(8)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

torch.manual_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% 数据
BATCH = 256
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
train_set = torchvision.datasets.FashionMNIST('./data', train=True,  transform=trans, download=True)
test_set  = torchvision.datasets.FashionMNIST('./data', train=False, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0)

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',  'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
NESTS = {0: [0, 2, 3, 4, 6], 1: [5, 7, 9], 2: [1, 8]}
NEST_NAMES = {0: 'Tops', 1: 'Footwear', 2: 'Others'}
N_NESTS, N_CLS = 3, 10

#%% 特征提取器（两个模型共用同一结构）
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x):
        return self.fc(self.conv(x))

#%% Luce / Softmax 模型（MNL）
class LuceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = CNN()
        self.head = nn.Linear(256, N_CLS)

    def forward(self, x):
        return self.head(self.feat(x))   # 返回 logits

#%% 嵌套 Logit 模型
# P(i) = P(i|nest_k) × P(nest_k)
# P(i|nest_k) = softmax(V/λ_k) 在巢内
# IV_k = λ_k × log Σ_{j∈k} exp(V_j/λ_k)
# P(nest_k) = softmax(α_k + IV_k)
class NestedLogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat       = CNN()
        self.item_head  = nn.Linear(256, N_CLS)
        self.nest_bias  = nn.Parameter(torch.zeros(N_NESTS))
        # λ_k 可学习，sigmoid 映射到 (0.1, 1.0]
        self._lam_raw   = nn.Parameter(torch.ones(N_NESTS) * 1.5)
        for k, members in NESTS.items():
            self.register_buffer(f'nest_{k}', torch.tensor(members))

    @property
    def lambdas(self):
        return 0.1 + 0.9 * torch.sigmoid(self._lam_raw)

    def forward(self, x):
        V   = self.item_head(self.feat(x))   # (B, 10)
        lam = self.lambdas
        B   = x.size(0)

        log_p_cond = torch.zeros(B, N_CLS,   device=x.device)
        IV         = torch.zeros(B, N_NESTS, device=x.device)

        for k, members in NESTS.items():
            idx   = getattr(self, f'nest_{k}')
            V_k   = V[:, idx] / lam[k]
            lse_k = torch.logsumexp(V_k, dim=1)
            log_p_cond[:, idx] = V_k - lse_k.unsqueeze(1)
            IV[:, k] = lam[k] * lse_k

        nest_logits = IV + self.nest_bias.unsqueeze(0)
        log_p_nest  = nest_logits - torch.logsumexp(nest_logits, dim=1, keepdim=True)

        log_p = torch.zeros(B, N_CLS, device=x.device)
        for k, members in NESTS.items():
            idx = getattr(self, f'nest_{k}')
            log_p[:, idx] = log_p_cond[:, idx] + log_p_nest[:, k:k+1]

        return log_p   # 返回 log-概率，配合 NLLLoss

#%% 训练与评估
def train_epoch(model, loader, optimizer, loss_fn, is_nested):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += y.size(0)
    return total_loss / n, correct / n

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    cls_correct = [0] * N_CLS
    cls_total   = [0] * N_CLS
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            total_loss += loss_fn(out, y).item() * y.size(0)
            pred        = out.argmax(1)
            correct    += (pred == y).sum().item()
            n          += y.size(0)
            for c in range(N_CLS):
                m = (y == c)
                cls_correct[c] += (pred[m] == y[m]).sum().item()
                cls_total[c]   += m.sum().item()
    per_cls = [cls_correct[c] / max(cls_total[c], 1) for c in range(N_CLS)]
    return total_loss / n, correct / n, per_cls

def run_training(model, loss_fn, is_nested, num_epochs=8):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    hist = defaultdict(list)
    for ep in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn, is_nested)
        te_loss, te_acc, per_cls = evaluate(model, test_loader, loss_fn)
        sch.step()
        hist['tr_loss'].append(tr_loss)
        hist['tr_acc'].append(tr_acc)
        hist['te_loss'].append(te_loss)
        hist['te_acc'].append(te_acc)
        hist['per_cls'].append(per_cls)
        print(f'[{ep}/{num_epochs}] train {tr_loss:.4f}/{tr_acc:.3f}  test {te_loss:.4f}/{te_acc:.3f}')
    return hist

#%% 训练 Luce/Softmax
print('=== Luce / Softmax (MNL) ===')
luce_model = LuceModel()
luce_hist  = run_training(luce_model, nn.CrossEntropyLoss(), is_nested=False)

#%% 训练嵌套 Logit
print('\n=== Nested Logit ===')
nested_model = NestedLogitModel()
nested_hist  = run_training(nested_model, nn.NLLLoss(), is_nested=True)

#%% 打印习得的 λ 参数
print('\n习得的尺度参数 λ_k：')
with torch.no_grad():
    lam_vals = nested_model.lambdas.cpu().numpy()
for k, name in NEST_NAMES.items():
    members = [CLASS_NAMES[c] for c in NESTS[k]]
    print(f'  Nest {k} ({name:10s})  λ={lam_vals[k]:.4f}  '
          f'巢内相关 ρ≈{1 - lam_vals[k]**2:.4f}  {members}')
print('λ < 1 → 巢内备选项存在正相关（违反 IIA）；λ = 1 → 退化为 MNL')

#%% IIA 违反实验
# 模拟"红色巴士/蓝色巴士"：移除一个竞争者后，观察目标类别概率的变化
# IIA 成立 → 移除任意对手，目标概率提升比例相同
# 嵌套 Logit  → 移除同巢对手时提升更大（巢内竞争减弱），移除异巢时基本不变
def iia_experiment(model, is_nested, target=0, remove=2):
    """返回 (目标基准概率, 被移除类基准概率, 移除后目标概率)"""
    model.eval()
    p_target, p_remove, p_after = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            mask = (y == target)
            if not mask.any():
                continue
            X_t = X[mask].to(DEVICE)
            out = model(X_t)
            probs = torch.exp(out) if is_nested else torch.softmax(out, dim=1)
            p_target.append(probs[:, target].mean().item())
            p_remove.append(probs[:, remove].mean().item())

            out_mod = out.clone()
            out_mod[:, remove] = -1e9
            if is_nested:
                p_mod = torch.exp(out_mod - torch.logsumexp(out_mod, dim=1, keepdim=True))
            else:
                p_mod = torch.softmax(out_mod, dim=1)
            p_after.append(p_mod[:, target].mean().item())
    return np.mean(p_target), np.mean(p_remove), np.mean(p_after)

print('\n=== IIA 违反实验（T-shirt 为目标类） ===')
print('归一化增量 = (移除后目标概率 - 移除前) / 被移除类概率  →  IIA 成立时两列应相等')
for mname, mdl, is_n in [('Luce/Softmax', luce_model, False), ('Nested Logit', nested_model, True)]:
    pt_s, pr_s, pa_s = iia_experiment(mdl, is_n, target=0, remove=2)
    pt_d, pr_d, pa_d = iia_experiment(mdl, is_n, target=0, remove=7)
    print(f'\n  {mname}：')
    print(f'    移除同巢 Pullover (p={pr_s:.3f}) → T-shirt: {pt_s:.4f}→{pa_s:.4f}  归一化增量={( pa_s-pt_s)/pr_s:+.3f}')
    print(f'    移除异巢 Sneaker  (p={pr_d:.3f}) → T-shirt: {pt_d:.4f}→{pa_d:.4f}  归一化增量={(pa_d-pt_d)/pr_d:+.3f}')

#%% 可视化
EPOCHS = range(1, len(luce_hist['te_acc']) + 1)
C = {'luce': '#4C72B0', 'nested': '#DD8452'}
NCOL = [['#2196F3'] * 5, ['#FF5722'] * 3, ['#4CAF50'] * 2]
NCOL_FLAT = [c for sub in NCOL for c in sub]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Luce/Softmax vs Nested Logit — FashionMNIST', fontsize=13, fontweight='bold')

# 准确率曲线
ax = axes[0, 0]
ax.plot(EPOCHS, [a*100 for a in luce_hist['tr_acc']],   '--', color=C['luce'],   label='Luce train')
ax.plot(EPOCHS, [a*100 for a in luce_hist['te_acc']],   '-',  color=C['luce'],   label='Luce test', lw=2)
ax.plot(EPOCHS, [a*100 for a in nested_hist['tr_acc']], '--', color=C['nested'], label='Nested train')
ax.plot(EPOCHS, [a*100 for a in nested_hist['te_acc']], '-',  color=C['nested'], label='Nested test', lw=2)
ax.set_title('Accuracy (%)')
ax.set_xlabel('Epoch')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 测试损失
ax = axes[0, 1]
ax.plot(EPOCHS, luce_hist['te_loss'],   color=C['luce'],   label='Luce', lw=2)
ax.plot(EPOCHS, nested_hist['te_loss'], color=C['nested'], label='Nested', lw=2)
ax.set_title('Test Loss')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(alpha=0.3)

# 习得的 λ
ax = axes[0, 2]
nc = ['#2196F3', '#FF5722', '#4CAF50']
with torch.no_grad():
    lam_np = nested_model.lambdas.cpu().numpy()
bars = ax.bar([NEST_NAMES[k] for k in range(N_NESTS)], lam_np, color=nc, edgecolor='black')
ax.axhline(1.0, color='red', ls='--', lw=1.5, label='λ=1 (IIA)')
ax.set_ylim(0, 1.25)
ax.set_title('Learned λ_k')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')
for bar, v in zip(bars, lam_np):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.3f}', ha='center', fontsize=9)

# 逐类准确率
ax = axes[1, 0]
luce_cls   = luce_hist['per_cls'][-1]
nested_cls = nested_hist['per_cls'][-1]
x = np.arange(N_CLS)
w = 0.38
ax.bar(x - w/2, [v*100 for v in luce_cls],   w, color=C['luce'],   alpha=0.85, label='Luce')
ax.bar(x + w/2, [v*100 for v in nested_cls], w, color=C['nested'], alpha=0.85, label='Nested')
ax.set_xticks(x)
ax.set_xticklabels([n[:6] for n in CLASS_NAMES], rotation=45, ha='right', fontsize=7)
ax.set_title('Per-Class Accuracy (%)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

# 嵌套结构树
ax = axes[1, 1]
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Nested Logit Structure', fontsize=10)
ax.text(5, 9.2, 'IMAGE', ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFECB3', edgecolor='#FFC107'))
nest_xs = [2, 5, 8]
for k, nx in enumerate(nest_xs):
    ax.annotate('', xy=(nx, 7.0), xytext=(5, 8.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(nx, 6.6, f'Nest {k}\n{NEST_NAMES[k]}\nλ={lam_np[k]:.3f}',
            ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=nc[k], alpha=0.5, edgecolor='black'))
leaf_info = [
    (0.5, 3.8, 'T-shirt', 0), (1.5, 3.8, 'Pullover', 0), (2.5, 3.8, 'Dress', 0),
    (3.5, 3.8, 'Coat', 0),    (4.5, 3.8, 'Shirt', 0),
    (5.0, 1.8, 'Sandal', 1),  (6.0, 1.8, 'Sneaker', 1), (7.0, 1.8, 'A.boot', 1),
    (7.8, 3.8, 'Trouser', 2), (8.8, 3.8, 'Bag', 2),
]
parent_xy = [(2,6.2)]*5 + [(5,6.2)]*3 + [(8,6.2)]*2
for (lx, ly, lname, nk), (px, py) in zip(leaf_info, parent_xy):
    ax.annotate('', xy=(lx, ly+0.35), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='lightgray', lw=0.9))
    ax.text(lx, ly, lname, ha='center', va='center', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=nc[nk], alpha=0.35, edgecolor='black'))

# IIA 实验结果（归一化增量：Δp_target / p_removed）
# IIA 成立 → 两组柱高相等；嵌套 Logit → 同巢柱更高
ax = axes[1, 2]
ax.set_title('IIA Violation\n(normalized: Δp_target / p_removed)')
luce_norm, nested_norm = [], []
for rem in [2, 7]:
    pt_l, pr_l, pa_l = iia_experiment(luce_model, False, 0, rem)
    pt_n, pr_n, pa_n = iia_experiment(nested_model, True, 0, rem)
    luce_norm.append((pa_l - pt_l) / pr_l)
    nested_norm.append((pa_n - pt_n) / pr_n)
x = np.arange(2)
w = 0.35
bars_l = ax.bar(x - w/2, luce_norm,   w, color=C['luce'],   alpha=0.85, label='Luce')
bars_n = ax.bar(x + w/2, nested_norm, w, color=C['nested'], alpha=0.85, label='Nested')
ax.set_xticks(x)
ax.set_xticklabels(['Remove Pullover\n(same nest)', 'Remove Sneaker\n(diff nest)'], fontsize=8)
ax.set_ylabel('Δp(T-shirt) / p(removed)')
ax.axhline(0, color='black', lw=0.8)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')
for bar, v in zip(list(bars_l) + list(bars_n), luce_norm + nested_norm):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.3f}',
            ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig('Luce_select_model/nested_logit_results.png', dpi=150, bbox_inches='tight')
plt.show()

#%% 最终结果汇总
_, luce_acc,   luce_cls   = evaluate(luce_model,   test_loader, nn.CrossEntropyLoss())
_, nested_acc, nested_cls = evaluate(nested_model, test_loader, nn.NLLLoss())
print(f'\n{"模型":<20} {"测试准确率":>10}')
print(f'{"Luce/Softmax":<20} {luce_acc*100:>9.2f}%')
print(f'{"Nested Logit":<20} {nested_acc*100:>9.2f}%')
print(f'\n{"类别":<15} {"Luce":>8} {"Nested":>8} {"Δ":>8}')
print('-' * 42)
for c in range(N_CLS):
    d = (nested_cls[c] - luce_cls[c]) * 100
    print(f'  {CLASS_NAMES[c]:<13} {luce_cls[c]*100:>7.1f}% {nested_cls[c]*100:>7.1f}% {d:>+7.1f}%')

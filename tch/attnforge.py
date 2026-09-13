import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SEBlock(nn.Module):
    # squeeze-and-excitation: перевзвешивает каналы по глобальному контексту
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        # x: (B, T, D) -> усредняю по T, получаю (B, D)
        s = x.mean(dim=1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(1)


class GatedMLPBlock(nn.Module):
    # gated mlp: две параллельные ветки, одна модулирует другую
    # это swiglu-подобная схема, работает лучше обычного mlp
    def __init__(self, dim, expansion=4, dropout=0.15):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc_gate = nn.Linear(dim, hidden)
        self.fc_val = nn.Linear(dim, hidden)
        self.fc_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        gate = F.silu(self.fc_gate(h))
        val = self.fc_val(h)
        out = self.fc_out(self.drop(gate * val))
        return identity + out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, expansion=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = GatedMLPBlock(dim, expansion, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = self.mlp(x)
        return x


class ConvStem(nn.Module):
    # свёрточный stem: локальные паттерны по оси T выучиваются раньше,
    # чем их увидит attention
    def __init__(self, in_dim, hidden_dim, kernel=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel, padding=kernel // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) для conv1d
        x = x.transpose(1, 2)
        x = self.drop(F.gelu(self.bn1(self.conv1(x))))
        x = self.drop(F.gelu(self.bn2(self.conv2(x))))
        return x.transpose(1, 2)


class MyModel(nn.Module):
    def __init__(
        self,
        in_dim=2,
        out_dim=3,
        hidden_dim=256,
        num_blocks=12,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.stem = ConvStem(in_dim, hidden_dim, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, expansion=4, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.se = SEBlock(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, out_dim)

        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

        # инициализация residual-веток по глубине (как в gpt-2):
        # масштабирую выходные проекции на 1/sqrt(2N), чтобы дисперсия
        # активаций не росла с глубиной
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * num_blocks))

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.se(x)
        x = self.norm_out(x)
        x = self.head_drop(x)
        return self.head(x)


class MyDataset(Dataset):
    def __init__(self, x, true):
        super().__init__()
        self.x = x
        self.true = true

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.true[index]


# на 12-блочной модели 4096 сэмплов мало -- она их просто запомнит,
# беру 65536, чтобы регуляризация и attention имели смысл
x = torch.randn(65536, 3, 2, device=device)
true = torch.randn(65536, 3, 3, device=device)

dataset = MyDataset(x, true)
dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
)

print(len(dataset))
print(dataset[1])
print(x.shape)
print(true.shape)
print("\ntrain")

model = MyModel(
    in_dim=2,
    out_dim=3,
    hidden_dim=256,
    num_blocks=12,
    num_heads=8,
    dropout=0.1,
).to(device)

# adamw + weight_decay: регуляризация, отделённая от адаптивного шага
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.95),
    weight_decay=0.05,
)

# warmup + cosine: warmup спасает от расхождения на первых шагах,
# cosine плавно сажает lr к концу
epochs = 600
steps_per_epoch = len(dataloader)
total_steps = epochs * steps_per_epoch
warmup_steps = steps_per_epoch * 10

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

mse = nn.MSELoss(reduction="mean")
model.train()

log_every = max(1, epochs // 30)

for e in range(epochs):
    epoch_loss = 0.0
    n_batches = 0

    for x_train, y_train in dataloader:
        pred = model(x_train)
        loss = mse(pred, y_train)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # clipping после warmup: на ранних шагах градиент и так мал
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.detach()
        n_batches += 1

    if e % log_every == 0:
        avg_loss = (epoch_loss / n_batches).item()
        lr_now = scheduler.get_last_lr()[0]
        print(f"{e}:\tloss = {avg_loss:.6f}\tlr = {lr_now:.2e}")

model.eval()
torch.save(model.state_dict(), "model30.pth")

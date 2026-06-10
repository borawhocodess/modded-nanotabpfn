import torch
import torch.nn.functional as F
from torch import nn

VERBOSE = True

L = 2
A = 4
E = 256
H = 768
O = 2

TABLE = [
    [
        [1, 0, 4, 1, 6, 5, 7, 8, 9, 10],
        [3, 0, 0, 1, 8, 4, 7, 8, 9, 11],
        [5, 0, 2, 1, 5, 5, 7, 8, 9, 10],
        [5, 0, 2, 1, 5, 5, 7, 8, 9, 11],
        [7, 0, 2, 1, 3, 1, 7, 8, 9, 10],
        [9, 0, 2, 1, 7, 9, 7, 8, 9, 11],
    ],
]
TRAIN_LABELS = [
    [
        [0],
        [0],
        [1],
        [1],
    ],
]

BATCH = len(TABLE)
ROWS = len(TABLE[0])
COLS = len(TABLE[0][0])
SEP = len(TRAIN_LABELS[0])

x = torch.tensor(TABLE, dtype=torch.float32)
y = torch.tensor(TRAIN_LABELS, dtype=torch.float32)

x_tbl = x[0]
y_tbl = y[0]
col_headers = "  ".join(f"{f'f{c}':>3}" for c in range(COLS))
print(f" row   {col_headers}  | label")
print("-" * 60)
for r in range(ROWS):
    row_vals = "  ".join(f"{int(x_tbl[r, c].item()):>3d}" for c in range(COLS))
    if r < SEP:
        label_str = f"{y_tbl[r].item()} (train)"
    else:
        label_str = "- (test)"
    print(f"  {r}    {row_vals}  | {label_str}")
print("-" * 60)

class NanoTabPFNModel(nn.Module):
    def __init__(self, l, a, e, h, o, residual_decay=1.0):
        super().__init__()
        self.l = l
        self.a = a
        self.e = e
        self.h = h
        self.o = o
        self.feature_encoder = FeatureEncoder(e)
        self.target_encoder = TargetEncoder(e)
        self.transformer_encoder = TransformerEncoderStack(l, a, e, h, residual_decay=residual_decay)
        self.decoder = Decoder(e, h, o)

        self.register_buffer("borders", None, persistent=True)

    def forward(self, *args, **kwargs):
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), sep=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            if VERBOSE: print("forward len(args) == 1 and isinstance(args[0], tuple)")
            return self._forward(*args, **kwargs)

    def _forward(self, src, sep):
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            if VERBOSE: print(f"y_src: {tuple(y_src.shape)} < x_src: {tuple(x_src.shape)}")
            y_src = y_src.unsqueeze(-1)
        if VERBOSE: print(f"x_src: {tuple(x_src.shape)}")
        if VERBOSE: print("-" * 80)
        x_src = self.feature_encoder(x_src, sep)
        if VERBOSE: print("-" * 80)
        if VERBOSE: print(f"x_src: {tuple(x_src.shape)}")
        num_rows = x_src.shape[1]
        if VERBOSE: print(f"num_rows: {num_rows}")
        if VERBOSE: print(f"y_src: {tuple(y_src.shape)}")
        if VERBOSE: print("-" * 80)
        y_src = self.target_encoder(y_src, num_rows)
        if VERBOSE: print("-" * 80)
        if VERBOSE: print(f"y_src: {tuple(y_src.shape)}")
        src = torch.cat([x_src, y_src], 2)
        if VERBOSE: print(f"src: {tuple(src.shape)}")
        if VERBOSE: print("-" * 80)
        output = self.transformer_encoder(src, sep)
        if VERBOSE: print("-" * 80)
        if VERBOSE: print(f"output: {tuple(output.shape)}")
        output = output[:, sep:, -1, :]
        if VERBOSE: print(f"output: {tuple(output.shape)}")
        if VERBOSE: print("-" * 80)
        output = self.decoder(output)
        if VERBOSE: print("-" * 80)
        if VERBOSE: print(f"output: {tuple(output.shape)}")
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, x, sep):
        if VERBOSE: print("FeatureEncoder forward")
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = x.unsqueeze(-1)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        mean = x[:, :sep].mean(dim=1, keepdim=True)
        std = x[:, :sep].std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        if VERBOSE: print(f" - x: {tuple(x.shape)}, mean: {tuple(mean.shape)}, std: {tuple(std.shape)}")
        x = self.linear_layer(x)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        return x


class TargetEncoder(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, y_train, num_rows):
        if VERBOSE: print("TargetEncoder forward")
        if VERBOSE: print(f" - y_train: {tuple(y_train.shape)}, num_rows: {num_rows}")
        mean = y_train.mean(dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        if VERBOSE: print(f" - y: {tuple(y.shape)}, mean: {tuple(mean.shape)}, padding: {tuple(padding.shape)}")
        y = y.unsqueeze(-1)
        if VERBOSE: print(f" - y: {tuple(y.shape)}")
        y = self.linear_layer(y)
        if VERBOSE: print(f" - y: {tuple(y.shape)}")
        return y


class TransformerEncoderStack(nn.Module):
    def __init__(self, l, a, e, h, residual_decay=1.0):
        super().__init__()
        self.residual_decay = residual_decay
        self.transformer_blocks = nn.ModuleList()
        for _ in range(l):
            self.transformer_blocks.append(TransformerEncoderLayer(a, e, h))

    def forward(self, x, sep):
        if VERBOSE: print("TransformerEncoderStack forward")
        for i, block in enumerate(self.transformer_blocks):
            if VERBOSE: print("-" * 40 + f" layer {i}")
            if VERBOSE: print(f" - x: {tuple(x.shape)}")
            x = x * (self.residual_decay ** i)
            if VERBOSE: print(f" - x: {tuple(x.shape)} * {self.residual_decay ** i}")
            x = block(x, sep=sep)
            if VERBOSE: print(f" - x: {tuple(x.shape)}")
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, a, e, h, eps=1e-5):
        super().__init__()
        self.num_heads = a
        self.head_dim = e // a
        assert e % a == 0, "Embedding size must be divisible by heads"

        self.qkv_datapoints = nn.Linear(e, 3 * e)
        self.qkv_features = nn.Linear(e, 3 * e)

        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, e)

        self.norm1 = nn.LayerNorm(e, eps=eps)
        self.norm2 = nn.LayerNorm(e, eps=eps)
        self.norm3 = nn.LayerNorm(e, eps=eps)

    def forward(self, src, sep):
        if VERBOSE: print("TransformerEncoderLayer forward")
        b, r, c, e = src.shape

        x = src.reshape(b * r, c, e)
        res = x
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = self.norm1(x)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        qkv = self.qkv_features(x)
        if VERBOSE: print(f" - qkv: {tuple(qkv.shape)}")
        qkv = qkv.reshape(b * r, c, 3, self.num_heads, self.head_dim)
        if VERBOSE: print(f" - qkv: {tuple(qkv.shape)}")
        qkv = qkv.permute(2, 0, 3, 1, 4)
        if VERBOSE: print(f" - qkv: {tuple(qkv.shape)}")
        q, k, v = qkv[0], qkv[1], qkv[2]
        if VERBOSE: print(f" - q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}")

        x = F.scaled_dot_product_attention(q, k, v)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = x.transpose(1, 2).reshape(b * r, c, e)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")

        src = res + x
        src = src.reshape(b, r, c, e)
        if VERBOSE: print(f" - src: {tuple(src.shape)}")
        x = src.transpose(1, 2).reshape(b * c, r, e)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        res = x
        x = self.norm2(x)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        qkv = self.qkv_datapoints(x)
        if VERBOSE: print(f" - qkv: {tuple(qkv.shape)}")
        qkv = qkv.reshape(b * c, r, 3, self.num_heads, self.head_dim)
        if VERBOSE: print(f" - qkv: {tuple(qkv.shape)}")
        qkv = qkv.permute(2, 0, 3, 1, 4)
        if VERBOSE: print(f" - qkv: {tuple(qkv.shape)}")
        q, k, v = qkv[0], qkv[1], qkv[2]
        if VERBOSE: print(f" - q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}")
        q_left, q_right = q.split([sep, r - sep], dim=2)
        if VERBOSE: print(f" - q_left: {tuple(q_left.shape)}")
        if VERBOSE: print(f" - q_right: {tuple(q_right.shape)}")
        k_train = k[:, :, :sep, :]
        v_train = v[:, :, :sep, :]
        if VERBOSE: print(f" - k_train: {tuple(k_train.shape)}")
        if VERBOSE: print(f" - v_train: {tuple(v_train.shape)}")
        x_left = F.scaled_dot_product_attention(q_left, k_train, v_train)
        if VERBOSE: print(f" - x_left: {tuple(x_left.shape)}")
        x_right = F.scaled_dot_product_attention(q_right, k_train, v_train)
        if VERBOSE: print(f" - x_right: {tuple(x_right.shape)}")
        x = torch.cat([x_left, x_right], dim=2)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = x.transpose(1, 2).reshape(b * c, r, e)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        src = res + x
        src = src.reshape(b, c, r, e).transpose(2, 1)
        if VERBOSE: print(f" - src: {tuple(src.shape)}")
        x = self.norm3(src)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = self.linear2(F.gelu(self.linear1(x)))
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        src = src + x
        if VERBOSE: print(f" - src: {tuple(src.shape)}")
        return src


class Decoder(nn.Module):
    def __init__(self, e, h, o):
        super().__init__()
        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, o)

    def forward(self, x):
        if VERBOSE: print("Decoder forward")
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = self.linear1(x)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = F.gelu(x)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        x = self.linear2(x)
        if VERBOSE: print(f" - x: {tuple(x.shape)}")
        return x


model = NanoTabPFNModel(l=L, a=A, e=E, h=H, o=O, residual_decay=0.95)
out = model((x, y), sep=SEP)

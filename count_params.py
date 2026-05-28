"""
精確計算 DEIMv2 Baseline vs 改後版本的參數量
包含：總參數量、每層參數量、實際有效參數量（考慮 Stochastic Depth 跳過率）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import copy
from engine.deim.deim_utils import RMSNorm, SwiGLUFFN, Gate, MLP
from engine.deim.dfine_decoder import MSDeformableAttention, LQE, Integral

# ── 設定（與 config 一致）────────────────────────────────────────
D_MODEL       = 192
N_HEAD        = 8
DIM_FF        = 512
N_LEVELS      = 3
N_POINTS      = 4
NUM_LAYERS    = 6
DROP_PATH_RATE = 0.0   # config 裡設定的值

# ── 計算每層 drop rate（與程式碼邏輯完全一致）──────────────────────
drop_rates = [0.0] * NUM_LAYERS
if DROP_PATH_RATE > 0.0 and NUM_LAYERS > 2:
    for idx in range(1, NUM_LAYERS - 1):
        drop_rates[idx] = (idx / (NUM_LAYERS - 2)) * DROP_PATH_RATE

print("=" * 65)
print("  Stochastic Depth 各層跳過率")
print("=" * 65)
for i, r in enumerate(drop_rates):
    label = "（第一層，永不跳過）" if i == 0 else \
            "（最後層，永不跳過）" if i == NUM_LAYERS-1 else ""
    print(f"  Layer {i}: drop_rate = {r:.4f}  ({r*100:.1f}% 機率跳過) {label}")

# ── 建立 Baseline 單層 ─────────────────────────────────────────────
class BaselineLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(D_MODEL, N_HEAD, batch_first=True)
        self.dropout1   = nn.Dropout(0.)
        self.norm1      = RMSNorm(D_MODEL)
        self.cross_attn = MSDeformableAttention(D_MODEL, N_HEAD, N_LEVELS, N_POINTS, method='default')
        self.dropout2   = nn.Dropout(0.)
        self.gateway    = Gate(D_MODEL, use_rmsnorm=True)
        self.swish_ffn  = SwiGLUFFN(D_MODEL, DIM_FF // 2, D_MODEL)
        self.dropout4   = nn.Dropout(0.)
        self.norm3      = RMSNorm(D_MODEL)

# ── 建立 Modified 單層 ────────────────────────────────────────────
class ModifiedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(D_MODEL, N_HEAD, batch_first=True)
        self.dropout1   = nn.Dropout(0.)
        self.norm1      = RMSNorm(D_MODEL)
        self.cross_attn = MSDeformableAttention(D_MODEL, N_HEAD, N_LEVELS, N_POINTS, method='default', decoupled=False)
        self.dropout2   = nn.Dropout(0.)
        self.gateway    = Gate(D_MODEL, use_rmsnorm=True)
        self.amr_gamma  = nn.Parameter(torch.zeros(1))   # ← 新增
        self.sa_gamma   = nn.Parameter(torch.zeros(1))   # ← 新增
        self.swish_ffn  = SwiGLUFFN(D_MODEL, DIM_FF // 2, D_MODEL)
        self.dropout4   = nn.Dropout(0.)
        self.norm3      = RMSNorm(D_MODEL)

def count_params(module):
    return sum(p.numel() for p in module.parameters())

# ── 建立各層並統計 ────────────────────────────────────────────────
baseline_layer = BaselineLayer()
modified_layer = ModifiedLayer()

bl_params = count_params(baseline_layer)
mo_params  = count_params(modified_layer)

print()
print("=" * 65)
print("  單層參數量對比")
print("=" * 65)
print(f"  Baseline  Layer params : {bl_params:>10,}")
print(f"  Modified  Layer params : {mo_params:>10,}")
print(f"  差異 (Modified - Baseline): +{mo_params - bl_params:,} params")

# ── 逐子模組細分 ──────────────────────────────────────────────────
print()
print("  Baseline 子模組細分：")
for name, mod in baseline_layer.named_children():
    p = count_params(mod)
    print(f"    {name:<20s}: {p:>8,}")

print()
print("  Modified 子模組細分：")
for name, mod in modified_layer.named_children():
    p = count_params(mod)
    flag = " ← 新增" if name in ("amr_gamma", "sa_gamma") else ""
    print(f"    {name:<20s}: {p:>8,}{flag}")
# amr_gamma / sa_gamma 是 Parameter，不是 Module，需要單獨列出
print(f"    {'amr_gamma':<20s}: {modified_layer.amr_gamma.numel():>8,}  ← 新增")
print(f"    {'sa_gamma':<20s}: {modified_layer.sa_gamma.numel():>8,}  ← 新增")

# ── 計算全 Decoder（6 層）總參數量 ────────────────────────────────
total_bl = bl_params * NUM_LAYERS
total_mo = mo_params  * NUM_LAYERS

print()
print("=" * 65)
print("  全 Decoder（6 層）參數量")
print("=" * 65)
print(f"  Baseline  total : {total_bl:>10,}")
print(f"  Modified  total : {total_mo:>10,}")
print(f"  新增差異        : +{total_mo - total_bl:,} params (+{(total_mo-total_bl)/total_bl*100:.4f}%)")

# ── 實際有效參數量（期望值）──────────────────────────────────────
# 被跳過的層不貢獻參數；期望有效參數 = Σ (1 - drop_rate_i) * layer_params
print()
print("=" * 65)
print("  實際有效參數量（期望值，考慮 Stochastic Depth 跳過率）")
print("=" * 65)

eff_bl = sum((1.0 - drop_rates[i]) * bl_params for i in range(NUM_LAYERS))
eff_mo = sum((1.0 - drop_rates[i]) * mo_params  for i in range(NUM_LAYERS))

print(f"  Stochastic Depth drop_path_rate = {DROP_PATH_RATE}")
print()
for i in range(NUM_LAYERS):
    eff_b = (1.0 - drop_rates[i]) * bl_params
    eff_m = (1.0 - drop_rates[i]) * mo_params
    note = ""
    if drop_rates[i] > 0:
        note = f"  （期望跳過 {drop_rates[i]*100:.1f}%）"
    print(f"  Layer {i}: Baseline eff={eff_b:>9,.0f}  Modified eff={eff_m:>9,.0f}{note}")

print()
print(f"  Baseline 期望有效參數總量: {eff_bl:>12,.1f}")
print(f"  Modified 期望有效參數總量: {eff_mo:>12,.1f}")
print(f"  有效參數差異              : +{eff_mo - eff_bl:,.1f} params")

if DROP_PATH_RATE == 0.0:
    print()
    print("  ⚠️  注意：drop_path_rate=0.0，Stochastic Depth 完全關閉")
    print("     → 所有層 100% 執行，有效參數量 = 靜態參數量（無跳過）")
    print()
    print("  若啟用 drop_path_rate（例如 0.1），效果如下試算：")
    sim_drop = 0.1
    sim_rates = [0.0] * NUM_LAYERS
    for idx in range(1, NUM_LAYERS - 1):
        sim_rates[idx] = (idx / (NUM_LAYERS - 2)) * sim_drop
    eff_bl_sim = sum((1.0 - sim_rates[i]) * bl_params for i in range(NUM_LAYERS))
    eff_mo_sim = sum((1.0 - sim_rates[i]) * mo_params  for i in range(NUM_LAYERS))
    reduction = (1 - eff_bl_sim / total_bl) * 100
    print(f"  若 drop_path_rate=0.1：")
    for i, r in enumerate(sim_rates):
        print(f"    Layer {i}: drop={r:.3f} ({r*100:.1f}%)")
    print(f"  Baseline 有效參數: {eff_bl_sim:>10,.1f} (靜態的 {100-reduction:.2f}%，省 {reduction:.2f}%)")
    print(f"  Modified 有效參數: {eff_mo_sim:>10,.1f}")

print()
print("=" * 65)
print("  結論")
print("=" * 65)
print(f"  1. 靜態參數差異：每層 +2 params (amr_gamma + sa_gamma)")
print(f"     6 層合計 +12 params，佔比極微（<0.01%）")
print(f"  2. 目前 drop_path_rate=0.0 → 實際有效參數 = 靜態參數")
print(f"     等同於 Baseline 的架構完整執行")
print(f"  3. 即便啟用 Stochastic Depth，")
print(f"     「被跳過的層」其參數仍然存在、仍會被更新")
print(f"     只是本次 forward 不計算它的輸出（不貢獻 FLOPs）")
print(f"  4. 推理時（eval mode）Stochastic Depth 不啟動，")
print(f"     所有層 100% 執行，無任何跳過")

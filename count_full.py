import sys
sys.path.insert(0, '.')
import torch
from engine.core import YAMLConfig

cfg = YAMLConfig('configs/deimv2/deimv2_dinov3_s_coco.yml')
model = cfg.model

def count(m):
    return sum(p.numel() for p in m.parameters())

total = count(model)

print("=" * 60)
print("  Full Model Parameter Breakdown")
print("=" * 60)
for name, module in model.named_children():
    p = count(module)
    pct = p / total * 100
    print(f"  {name:<28s}: {p:>10,}  ({pct:.1f}%)")

print("-" * 60)
print(f"  {'TOTAL MODEL':<28s}: {total:>10,}  (100.0%)")
print("=" * 60)
print()

# Backbone blocks specifically
dino = None
for name, module in model.named_children():
    if 'dino' in name.lower() or 'backbone' in name.lower():
        dino = module
        break

if dino is not None:
    bb_total = count(dino)
    # Find blocks
    try:
        blocks = dino._model.blocks
        block_total = sum(count(b) for b in blocks)
        other = bb_total - block_total
        print(f"  Backbone total     : {bb_total:>10,}")
        print(f"    Transformer blocks (12): {block_total:>10,}  ({block_total/bb_total*100:.1f}% of backbone)")
        print(f"    Other (patch embed etc): {other:>10,}  ({other/bb_total*100:.1f}% of backbone)")
    except:
        print(f"  Backbone total: {bb_total:,}")

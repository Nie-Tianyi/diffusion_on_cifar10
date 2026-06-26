"""Visualize the U-Net architecture with ASCII art and parameter counts.

Usage:
    uv run python visualize_unet.py          # full diagram
    uv run python visualize_unet.py --params # parameter breakdown only
    uv run python visualize_unet.py --table  # tabular summary
"""

import argparse
import torch.nn as nn

from config import cifar10_config
from unet import UNet, AttentionBlock


def fmt_params(n: int) -> str:
    """Format parameter count in human-readable form."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# -- data for the diagram ----------------------------------------------------

def build_diagram_data(config):
    """Compute all shapes and attention flags."""
    base = config.model.base_channels
    mul = config.model.channel_multipliers
    chs = [base * m for m in mul]
    img = config.model.image_size
    res_list = [img // (2 ** i) for i in range(len(chs))]
    attn_res = config.model.attention_resolutions
    n_res = config.model.num_res_blocks

    down_stages = []
    in_ch = base
    for i, (out_ch, res) in enumerate(zip(chs, res_list)):
        attn = res in attn_res
        ds = i < len(chs) - 1
        down_stages.append({
            "res": res,
            "in_ch": in_ch,
            "out_ch": out_ch,
            "attn": attn,
            "downsample": ds,
            "n_res": n_res,
        })
        in_ch = out_ch

    mid_ch = chs[-1]
    mid_res = res_list[-1]

    up_stages = []
    rev_chs = list(reversed(chs))
    in_ch = mid_ch
    for i, (out_ch, res, skip_ch) in enumerate(
        zip(rev_chs, reversed(res_list), reversed(chs))
    ):
        attn = res in attn_res
        us = i < len(rev_chs) - 1
        up_stages.append({
            "res": res,
            "in_ch": in_ch,
            "out_ch": out_ch,
            "skip_ch": skip_ch,
            "attn": attn,
            "upsample": us,
            "n_res": n_res,
        })
        in_ch = out_ch

    return {
        "in_channels": config.model.in_channels,
        "out_channels": config.model.out_channels,
        "base_ch": base,
        "time_emb_dim": base * 4,
        "down_stages": down_stages,
        "mid_ch": mid_ch,
        "mid_res": mid_res,
        "up_stages": up_stages,
        "final_ch": rev_chs[-1],
    }


# -- ASCII diagram -----------------------------------------------------------

def draw_ascii(data):
    """Print an ASCII tree of the U-Net architecture with tensor shapes."""

    B = 8  # dummy batch size for display

    def shape(c, h, w, label=""):
        suffix = f" ({label})" if label else ""
        return f"({B}, {c}, {h}, {w}){suffix}"

    print()
    print("=" * 78)
    print("  DDPM U-Net  --  Architecture Diagram")
    print("=" * 78)
    print()

    # Input
    print(f"  Input image                 {shape(3, 32, 32)}")
    print(f"  Timestep t                  ({B},) -> SinusoidalEmbed(128)")
    print(f"                              -> Linear+SiLU+Linear")
    print(f"                              -> ({B}, {data['time_emb_dim']})")
    print(f"  init_conv  (Conv2d 3->128, k3)  -> {shape(128, 32, 32)}")
    print(f"  |")

    # Encoder
    down = data["down_stages"]
    for i, s in enumerate(down):
        attn_tag = " + SelfAttn" if s["attn"] else ""
        ds_tag = " + downsample" if s["downsample"] else ""
        res_in = s["res"]
        res_out = res_in // 2 if s["downsample"] else res_in
        n_res = s["n_res"]

        box_top = "+" if i < len(down) - 1 else "\\"
        print(f"  {box_top}-- Down[{i}]  {'-' * 52}")
        print(f"  |   {n_res}x ResBlock({s['in_ch']}->{s['out_ch']})  @ {res_in}x{res_in}{attn_tag}{ds_tag}")

        if s["downsample"]:
            print(f"  |   |-> skip connection  ->  {shape(s['out_ch'], res_in, res_in, 'skip')}")
            print(f"  |   \\-> Conv2d({s['out_ch']},{s['out_ch']},k3,s2)  ->  {shape(s['out_ch'], res_out, res_out)}")
        else:
            print(f"  |   \\-> skip connection  ->  {shape(s['out_ch'], res_in, res_in, 'skip')}")
        if i < len(down) - 1:
            print(f"  |")

    # Bottleneck
    mid = data["mid_ch"]
    mr = data["mid_res"]
    print(f"  |")
    print(f"  +-- Bottleneck  {'-' * 47}")
    print(f"  |   ResBlock({mid}->{mid})                @ {mr}x{mr}")
    print(f"  |   SelfAttn({mid}, heads=4)             @ {mr}x{mr}")
    print(f"  |   ResBlock({mid}->{mid})                @ {mr}x{mr}")
    print(f"  |")

    # Decoder
    up = data["up_stages"]
    for i, s in enumerate(up):
        attn_tag = " + SelfAttn" if s["attn"] else ""
        us_tag = " + upsample" if s["upsample"] else ""
        concat_ch = s["in_ch"] + s["skip_ch"]

        print(f"  +-- Up[{i}]  {'-' * 54}")
        print(f"  |   concat(x={s['in_ch']}ch, skip={s['skip_ch']}ch)  ->  {shape(concat_ch, s['res'], s['res'])}")

        if s["n_res"] == 2:
            print(f"  |   ResBlock({concat_ch}->{s['out_ch']})     @ {s['res']}x{s['res']}{attn_tag}{us_tag}")
            print(f"  |   ResBlock({s['out_ch']}->{s['out_ch']})          @ {s['res']}x{s['res']}")
        else:
            print(f"  |   ResBlock({concat_ch}->{s['out_ch']})     @ {s['res']}x{s['res']}{attn_tag}{us_tag}")

        next_res = s["res"] * 2 if s["upsample"] else s["res"]
        if s["upsample"]:
            print(f"  |   \\-> Upsample(nearest) + Conv2d({s['out_ch']},{s['out_ch']},k3)")
            print(f"         ->  {shape(s['out_ch'], next_res, next_res)}")
        else:
            print(f"  |   \\-> (no upsample)  ->  {shape(s['out_ch'], next_res, next_res)}")

    # Output
    fc = data["final_ch"]
    out_c = data["out_channels"]
    print(f"  |")
    print(f"  \\-- Output  {'-' * 53}")
    print(f"       GroupNorm + SiLU")
    print(f"       Conv2d({fc}->{out_c}, k3, p1)  ->  {shape(out_c, 32, 32, 'predicted noise')}")
    print()
    print("-" * 78)
    print("  ResBlock:   GN -> SiLU -> Conv3x3 -> +time_proj -> GN -> SiLU -> Drop -> Conv3x3  + residual")
    print("  SelfAttn:   GN -> reshape -> MultiheadAttention(4 heads) -> reshape  + residual")
    print("-" * 78)


# -- Parameter breakdown -----------------------------------------------------

def draw_params(model):
    """Print a parameter-count breakdown by component."""
    print()
    print("=" * 60)
    print("  Parameter Breakdown")
    print("=" * 60)
    header = f"  {'Component':<40} {'Params':>10}  {'%':>6}"
    print(header)
    print("  " + "-" * 58)

    def row(name, n, total):
        pct = n / total * 100 if total else 0
        print(f"  {name:<40} {fmt_params(n):>10}  {pct:5.1f}%")

    total = count_params(model)

    # Time embedding
    n = count_params(model.time_embed)
    row("Time embedding (full pipeline)", n, total)

    # Init conv
    row("init_conv", count_params(model.init_conv), total)

    # Encoder
    for i, block in enumerate(model.down_blocks):
        for j, rb in enumerate(block.res_blocks):
            row(f"  Down[{i}] ResBlock[{j}]", count_params(rb), total)
        for j, attn in enumerate(block.attentions):
            if isinstance(attn, AttentionBlock):
                row(f"  Down[{i}] SelfAttn[{j}]", count_params(attn), total)
        if hasattr(block, 'downsample') and not isinstance(block.downsample, nn.Identity):
            row(f"  Down[{i}] downsample", count_params(block.downsample), total)

    # Bottleneck
    row("Mid ResBlock1", count_params(model.mid_block1), total)
    row("Mid SelfAttn", count_params(model.mid_attn), total)
    row("Mid ResBlock2", count_params(model.mid_block2), total)

    # Decoder
    for i, block in enumerate(model.up_blocks):
        for j, rb in enumerate(block.res_blocks):
            row(f"  Up[{i}] ResBlock[{j}]", count_params(rb), total)
        for j, attn in enumerate(block.attentions):
            if isinstance(attn, AttentionBlock):
                row(f"  Up[{i}] SelfAttn[{j}]", count_params(attn), total)
        if hasattr(block, 'upsample') and not isinstance(block.upsample, nn.Identity):
            row(f"  Up[{i}] upsample", count_params(block.upsample), total)

    # Output
    row("out_norm", count_params(model.out_norm), total)
    row("out_conv", count_params(model.out_conv), total)

    print("  " + "-" * 58)
    row("TOTAL", total, total)
    row("  of which trainable", count_trainable(model), total)
    print("=" * 60)
    print()


# -- Tabular summary ---------------------------------------------------------

def draw_table(data):
    """Compact tabular summary matching the ASCII sketch in CLAUDE.md."""
    print()
    print("=" * 76)
    print("  Compact Architecture Table  (matches CLAUDE.md)")
    print("=" * 76)
    header = f"  {'Stage':<12} {'Block':<40} {'Resolution':<10} {'Channels':<14}"
    print(header)
    print("  " + "-" * 74)

    init_label = "Init conv"
    init_block = "Conv2d(3->128, k3)"
    init_ch = "3 -> 128"
    print(f"  {init_label:<12} {init_block:<40} {'32x32':<10} {init_ch:<14}")

    for i, s in enumerate(data["down_stages"]):
        attn = " + SelfAttn" if s["attn"] else ""
        ds = " + Downsample" if s["downsample"] else ""
        n = s["n_res"]
        blk = f"{n}xResBlock + skip{attn}{ds}"
        res_str = f"{s['res']}x{s['res']}"
        ch_str = f"{s['in_ch']} -> {s['out_ch']}"
        stage_label = f"Down[{i}]"
        print(f"  {stage_label:<12} {blk:<40} {res_str:<10} {ch_str:<14}")

    mid = data["mid_ch"]
    mr = data["mid_res"]
    mid_res_str = f"{mr}x{mr}"
    print(f"  {'Mid':<12} {'ResBlock + SelfAttn + ResBlock':<40} {mid_res_str:<10} {str(mid):<14}")

    for i, s in enumerate(data["up_stages"]):
        attn = " + SelfAttn" if s["attn"] else ""
        us = " + Upsample" if s["upsample"] else ""
        n = s["n_res"]
        blk = f"concat(skip) + {n}xResBlock{attn}{us}"
        res_str = f"{s['res']}x{s['res']}"
        ch_str = f"{s['in_ch']}+skip -> {s['out_ch']}"
        stage_label = f"Up[{i}]"
        print(f"  {stage_label:<12} {blk:<40} {res_str:<10} {ch_str:<14}")

    fc = data["final_ch"]
    oc = data["out_channels"]
    out_ch_str = f"{fc} -> {oc}"
    print(f"  {'Output':<12} {'GN + SiLU + Conv2d(k3)':<40} {'32x32':<10} {out_ch_str:<14}")
    print("=" * 76)
    print()


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize the DDPM U-Net architecture")
    parser.add_argument("--params", action="store_true", help="Parameter breakdown only")
    parser.add_argument("--table", action="store_true", help="Tabular summary only")
    args = parser.parse_args()

    config = cifar10_config()
    model = UNet(config.model)

    data = build_diagram_data(config)

    if args.params:
        draw_params(model)
        return

    if args.table:
        draw_table(data)
        draw_params(model)
        return

    # Full view
    draw_ascii(data)
    draw_table(data)
    draw_params(model)

    total = count_params(model)
    print(f"  GPU memory estimate (BF16, no grad):  {total * 2 / 1e9:.2f} GB  (params only)")
    print(f"  GPU memory estimate (BF16, training): ~{total * 6 / 1e9:.1f} GB  (params + grads + optimiser)")
    print(f"  Peak VRAM with batch=256:              ~4-6 GB  (activations dominate)")
    print()


if __name__ == "__main__":
    main()

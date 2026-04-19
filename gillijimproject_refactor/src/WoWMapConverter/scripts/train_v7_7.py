#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from train_v7 import build_arg_parser, train


DEFAULT_V77_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "output" / "ml-training" / "v7_7"


def build_v77_arg_parser():
    parser = build_arg_parser()
    parser.description = "Train the unified V7.7 terrain regressor with the auxiliary detail head enabled by default."
    parser.set_defaults(
        model_family="v7_7",
        output_dir=str(DEFAULT_V77_OUTPUT_DIR),
        norm_type="group",
        global_residual_scale=0.35,
        detail_head_weight=0.10,
        detail_head_start_epoch=1,
        bounds_loss_scale=0.40,
        bounds_loss_start_epoch=8,
        multiscale_global_weight=0.04,
        multiscale_local_weight=0.03,
        wdl_coarse_weight=0.02,
        wdl_coarse_size=17,
        output_head_mode="linear_unclamped_train",
    )
    return parser


if __name__ == "__main__":
    train(build_v77_arg_parser().parse_args())
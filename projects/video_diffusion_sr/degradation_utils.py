# Copyright (c) 2022 BasicSR: Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache License, Version 2.0 (the "License")
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 1st June 2025
#
# Original file was released under Apache License, Version 2.0 (the "License"), with the full license text
# available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.

import io
import math
import random
from typing import Dict
import av
import numpy as np
import torch
from basicsr.data.degradations import (
    circular_lowpass_kernel,
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
    random_mixed_kernels,
)
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from einops import rearrange
from torch import nn
from torch.nn import functional as F


def remove_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for k in list(state_dict.keys()):
        if k.startswith("_flops_wrap_module."):
            v = state_dict.pop(k)
            state_dict[k.replace("_flops_wrap_module.", "")] = v
        if k.startswith("module."):
            v = state_dict.pop(k)
            state_dict[k.replace("module.", "")] = v
    return state_dict


def clean_memory_bank(module: nn.Module):
    if hasattr(module, "padding_bank"):
        module.padding_bank = None
    for child in module.children():
        clean_memory_bank(child)


para_dic = {
    "kernel_list": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob": 0.1,
    "blur_sigma": [0.2, 1.5],
    "betag_range": [0.5, 2.0],
    "betap_range": [1, 1.5],
    "kernel_list2": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob2": 0.1,
    "blur_sigma2": [0.2, 1.0],
    "betag_range2": [0.5, 2.0],
    "betap_range2": [1, 1.5],
    "final_sinc_prob": 0.5,
}

degrade_dic = {
    # "gt_usm": True,  # USM the ground-truth
    # the first degradation process
    "resize_prob": [0.2, 0.7, 0.1],  # up, down, keep
    "resize_range": [0.3, 1.5],
    "gaussian_noise_prob": 0.5,
    "noise_range": [1, 15],
    "poisson_scale_range": [0.05, 2],
    "gray_noise_prob": 0.4,
    "jpeg_range": [60, 95],
    # the second degradation process
    "second_blur_prob": 0.5,
    "resize_prob2": [0.3, 0.4, 0.3],  # up, down, keep
    "resize_range2": [0.6, 1.2],
    "gaussian_noise_prob2": 0.5,
    "noise_range2": [1, 12],
    "poisson_scale_range2": [0.05, 1.0],
    "gray_noise_prob2": 0.4,
    "jpeg_range2": [60, 95],
    "queue_size": 180,
    "scale": 4,  # output size: ori_h // scale
    "sharpen": False,
}


def set_para(para_dic):
    # blur settings for the first degradation
    # blur_kernel_size = opt['blur_kernel_size']
    kernel_list = para_dic["kernel_list"]
    kernel_prob = para_dic["kernel_prob"]
    blur_sigma = para_dic["blur_sigma"]
    betag_range = para_dic["betag_range"]
    betap_range = para_dic["betap_range"]
    sinc_prob = para_dic["sinc_prob"]

    # blur settings for the second degradation
    # blur_kernel_size2 = opt['blur_kernel_size2']
    kernel_list2 = para_dic["kernel_list2"]
    kernel_prob2 = para_dic["kernel_prob2"]
    blur_sigma2 = para_dic["blur_sigma2"]
    betag_range2 = para_dic["betag_range2"]
    betap_range2 = para_dic["betap_range2"]
    sinc_prob2 = para_dic["sinc_prob2"]

    # a final sinc filter
    final_sinc_prob = para_dic["final_sinc_prob"]

    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    pulse_tensor = torch.zeros(
        21, 21
    ).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None,
        )
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------ Generate kernels (used in the second degradation) -------------- #
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob2:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel2 = random_mixed_kernels(
            kernel_list2,
            kernel_prob2,
            kernel_size,
            blur_sigma2,
            blur_sigma2,
            [-math.pi, math.pi],
            betag_range2,
            betap_range2,
            noise_range=None,
        )
    pad_size = (21 - kernel_size) // 2
    kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

    # ------------------------------------- sinc kernel ------------------------------------- #
    if np.random.uniform() < final_sinc_prob:
        kernel_size = random.choice(kernel_range)
        omega_c = np.random.uniform(np.pi / 3, np.pi)
        sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        sinc_kernel = torch.FloatTensor(sinc_kernel)
    else:
        sinc_kernel = pulse_tensor
    kernel = torch.FloatTensor(kernel)
    kernel2 = torch.FloatTensor(kernel2)
    return_d = {"kernel1": kernel, "kernel2": kernel2, "sinc_kernel": sinc_kernel}
    return return_d


def print_stat(a):
    print(
        f"shape={a.shape}, min={a.min():.2f}, \
        max={a.max():.2f}, var={a.var():.2f}, {a.flatten()[0]}"
    )


@torch.no_grad()
def esr_blur_gpu(image, paras, usm_sharpener, jpeger, device="cpu"):
    """
    input and output: image is a tensor with shape: b f c h w, range (-1, 1)
    """
    video_length = image.shape[1]
    image = rearrange(image, "b f c h w -> (b f) c h w").to(device)
    image = (image + 1) * 0.5
    if degrade_dic["sharpen"]:
        gt_usm = usm_sharpener(image)
    else:
        gt_usm = image
    ori_h, ori_w = image.size()[2:4]
    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, paras["kernel1"].unsqueeze(0).to(device))
    # random resize
    updown_type = random.choices(["up", "down", "keep"], degrade_dic["resize_prob"])[0]
    if updown_type == "up":
        scale = np.random.uniform(1, degrade_dic["resize_range"][1])
    elif updown_type == "down":
        scale = np.random.uniform(degrade_dic["resize_range"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # noise
    gray_noise_prob = degrade_dic["gray_noise_prob"]
    out = out.to(torch.float32)
    if np.random.uniform() < degrade_dic["gaussian_noise_prob"]:
        out = random_add_gaussian_noise_pt(
            out,
            # video_length=video_length,
            sigma_range=degrade_dic["noise_range"],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            # video_length=video_length,
            scale_range=degrade_dic["poisson_scale_range"],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
        )
    # out = out.to(torch.bfloat16)

    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*degrade_dic["jpeg_range"])
    out = torch.clamp(out, 0, 1)
    out = jpeger(out, quality=jpeg_p)

    # Video compression 1
    # print('Video compression 1')

    # print_stat(out)
    if video_length > 1:
        out = video_compression(out, device=device)
    # print('After video compression 1')

    # print_stat(out)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < degrade_dic["second_blur_prob"]:
        out = filter2D(out, paras["kernel2"].unsqueeze(0).to(device))
    # random resize
    updown_type = random.choices(["up", "down", "keep"], degrade_dic["resize_prob2"])[0]
    if updown_type == "up":
        scale = np.random.uniform(1, degrade_dic["resize_range2"][1])
    elif updown_type == "down":
        scale = np.random.uniform(degrade_dic["resize_range2"][0], 1)
    else:
        scale = 1
    mode = random.choice(["area", "bilinear", "bicubic"])
    out = F.interpolate(
        out,
        size=(
            int(ori_h / degrade_dic["scale"] * scale),
            int(ori_w / degrade_dic["scale"] * scale),
        ),
        mode=mode,
    )
    # noise
    gray_noise_prob = degrade_dic["gray_noise_prob2"]
    out = out.to(torch.float32)
    if np.random.uniform() < degrade_dic["gaussian_noise_prob2"]:
        out = random_add_gaussian_noise_pt(
            out,
            # video_length=video_length,
            sigma_range=degrade_dic["noise_range2"],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            # video_length=video_length,
            scale_range=degrade_dic["poisson_scale_range2"],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
        )
    # out = out.to(torch.bfloat16)

    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(ori_h // degrade_dic["scale"], ori_w // degrade_dic["scale"]), mode=mode
        )
        out = filter2D(out, paras["sinc_kernel"].unsqueeze(0).to(device))
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*degrade_dic["jpeg_range2"])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*degrade_dic["jpeg_range2"])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(ori_h // degrade_dic["scale"], ori_w // degrade_dic["scale"]), mode=mode
        )
        out = filter2D(out, paras["sinc_kernel"].unsqueeze(0).to(device))

    # print('Video compression 2')

    # print_stat(out)
    if video_length > 1:
        out = video_compression(out, device=device)
    # print('After video compression 2')

    # print_stat(out)

    out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
    blur_image = torch.clamp(out, 0, 1)
    # blur_image = ColorJitter(0.1, 0.1, 0.1, 0.05)(blur_image)  # 颜色数据增广
    # (-1, 1)
    blur_image = 2.0 * blur_image - 1
    blur_image = rearrange(blur_image, "(b f) c h w->b f c h w", f=video_length)
    return blur_image


def video_compression(video_in, device="cpu"):
    # Shape: (t, c, h, w); channel order: RGB; image range: [0, 1], float32.

    video_in = torch.clamp(video_in, 0, 1)
    params = dict(
        codec=["libx264", "h264", "mpeg4"],
        codec_prob=[1 / 3.0, 1 / 3.0, 1 / 3.0],
        bitrate=[1e4, 1e5],
    )  # 1e4, 1e5
    codec = random.choices(params["codec"], params["codec_prob"])[0]
    # print(f"use codec {codec}")

    bitrate = params["bitrate"]
    bitrate = np.random.randint(bitrate[0], bitrate[1] + 1)

    h, w = video_in.shape[-2:]
    video_in = F.interpolate(video_in, (h // 2 * 2, w // 2 * 2), mode="bilinear")

    buf = io.BytesIO()
    with av.open(buf, "w", "mp4") as container:
        stream = container.add_stream(codec, rate=1)
        stream.height = video_in.shape[-2]
        stream.width = video_in.shape[-1]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = bitrate

        for img in video_in:  # img: C H W; 0-1
            img_np = img.permute(1, 2, 0).contiguous() * 255.0
            # 1 reference_np = reference.detach(). to (torch.float) .cpu() .numpy ()
            img_np = img_np.detach().to(torch.float).cpu().numpy().astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(img_np, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    outputs = []
    with av.open(buf, "r", "mp4") as container:
        if container.streams.video:
            for frame in container.decode(**{"video": 0}):
                outputs.append(frame.to_rgb().to_ndarray().astype(np.float32))

    video_in = torch.Tensor(np.array(outputs)).permute(0, 3, 1, 2).contiguous()  # T C H W
    video_in = torch.clamp(video_in / 255.0, 0, 1).to(device)  # 0-1
    return video_in


@torch.no_grad()
def my_esr_blur(images, device="cpu"):
    """
    images is a list of tensor with shape: b f c h w, range (-1, 1)
    """
    jpeger = DiffJPEG(differentiable=False).to(device)
    usm_sharpener = USMSharp()
    if degrade_dic["sharpen"]:
        usm_sharpener = usm_sharpener.to(device)
    paras = set_para(para_dic)
    blur_image = [
        esr_blur_gpu(image, paras, usm_sharpener, jpeger, device=device) for image in images
    ]

    return blur_image


para_dic_latent = {
    "kernel_list": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob": 0.1,
    "blur_sigma": [0.2, 1.5],
    "betag_range": [0.5, 2.0],
    "betap_range": [1, 1.5],
    "kernel_list2": [
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    ],
    "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
    "sinc_prob2": 0.1,
    "blur_sigma2": [0.2, 1.0],
    "betag_range2": [0.5, 2.0],
    "betap_range2": [1, 1.5],
    "final_sinc_prob": 0.5,
}


def set_para_latent(para_dic):
    # blur settings for the first degradation
    # blur_kernel_size = opt['blur_kernel_size']
    kernel_list = para_dic["kernel_list"]
    kernel_prob = para_dic["kernel_prob"]
    blur_sigma = para_dic["blur_sigma"]
    betag_range = para_dic["betag_range"]
    betap_range = para_dic["betap_range"]
    sinc_prob = para_dic["sinc_prob"]

    # a final sinc filter

    kernel_range = [2 * v + 1 for v in range(1, 11)]  # kernel size ranges from 7 to 21
    pulse_tensor = torch.zeros(
        21, 21
    ).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1
    kernel_size = random.choice(kernel_range)
    if np.random.uniform() < sinc_prob:
        # this sinc filter setting is for kernels ranging from [7, 21]
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
    else:
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            betag_range,
            betap_range,
            noise_range=None,
        )
    # pad kernel
    pad_size = (21 - kernel_size) // 2
    kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    kernel = torch.FloatTensor(kernel)
    return_d = {"kernel1": kernel}
    return return_d


@torch.no_grad()
def latent_blur_gpu(image, paras, device="cpu"):
    """
    input and output: image is a tensor with shape: b f c h w, range (-1, 1)
    """
    video_length = image.shape[1]
    image = rearrange(image, "b f c h w -> (b f) c h w").to(device)
    image = (image + 1) * 0.5
    gt_usm = image
    ori_h, ori_w = image.size()[2:4]
    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(gt_usm, paras["kernel1"].unsqueeze(0).to(device))
    blur_image = torch.clamp(out, 0, 1)
    # blur_image = ColorJitter(0.1, 0.1, 0.1, 0.05)(blur_image)  # 颜色数据增广
    # (-1, 1)
    blur_image = 2.0 * blur_image - 1
    blur_image = rearrange(blur_image, "(b f) c h w->b f c h w", f=video_length)
    return blur_image


@torch.no_grad()
def add_latent_blur(images, device="cpu"):
    """
    images is a list of tensor with shape: b f c h w, range (-1, 1)
    """
    paras = set_para_latent(para_dic_latent)
    blur_image = [latent_blur_gpu(image, paras, device=device) for image in images]
    print("apply blur to the latents")

    return blur_image

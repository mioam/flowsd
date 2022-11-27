import sys
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from configs.submission import get_cfg
from configs.small_things_eval import get_cfg
from core.utils.misc import process_cfg
from utils import flow_viz
from utils import frame_utils
import cv2
import math
import os.path as osp

from core.FlowFormer import build_flowformer

from utils.utils import InputPadder, forward_interpolate

from flow import flow_forward
import webuiapi


TRAIN_SIZE = [432, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

@torch.no_grad()
def compute_flow(model, image1, image2, weights=None, device='cuda'):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].to(device), image2[None].to(device)

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(root_dir, fn, img_size):
    print(f"preparing image...")
    print(f"root dir = {root_dir}, fn = {fn}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    if img_size is not None:
        # dsize = compute_adaptive_image_size(image1.shape[0:2])
        # dsize = (1024, 576)
        image1 = cv2.resize(image1, dsize=img_size, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()


    # dirname = osp.dirname(fn)
    # filename = osp.splitext(osp.basename(fn))[0]

    # viz_dir = osp.join(viz_root_dir, dirname)
    # if not osp.exists(viz_dir):
    #     os.makedirs(viz_dir)

    # viz_fn = osp.join(viz_dir, filename + '.png')

    return image1

def build_model(device):
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model, map_location=device))

    model.to(device)
    model.eval()

    return model

def gen_flow(root_dir, flow_dir, model, img_list, img_size, device):
    if not osp.exists(flow_dir):
        os.makedirs(flow_dir)
    image_pre = None
    for fn in img_list:
        print(f"processing {fn}...")
        image = prepare_image(root_dir, fn, img_size)
        flow_fn = osp.join(flow_dir, osp.splitext(osp.basename(fn))[0] + '.npy')
        if image_pre is not None:
            flow = compute_flow(model, image_pre, image, None, device=device)
            np.save(flow_fn, flow)
        image_pre = image

def sd_flow(root_dir, flow_dir, viz_dir, model, img_list, alpha, img_size, device):
    if not osp.exists(viz_dir):
        os.makedirs(viz_dir)
    api = webuiapi.WebUIApi()
    image_sd = None
    image_pre = None
    for fn in img_list:
        print(f"processing {fn}...")
        image = prepare_image(root_dir, fn, img_size)
        viz_fn = osp.join(viz_dir, osp.splitext(osp.basename(fn))[0] + '.png')

        image_np = image.permute(1,2,0).numpy() / 255
        if image_pre is not None:
            if flow_dir is None:
                flow = compute_flow(model, image_pre, image, None, device=device)
            else:
                flow_fn = osp.join(flow_dir, osp.splitext(osp.basename(fn))[0] + '.npy')
                flow = np.load(flow_fn)
            # flow_img = flow_viz.flow_to_image(flow)
            # cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])
            w, s = flow_forward(image_sd, flow)
            w += alpha
            s += image_np * alpha
            image_sd = s / (w[:,:,None] + 1e-6)
            image_sd = np.clip(image_sd, 0, 1)
        else:
            image_sd = image_np

        res = api.img2img([Image.fromarray(np.uint8(image_sd*255))],
                denoising_strength=0.3,
                width=img_size[0],
                height=img_size[1],)
        image_sd = np.asarray(res.images[0]) / 255.
        plt.imsave(viz_fn, image_sd)
        image_pre = image

def generate_list(start_idx, end_idx, step=1):
    img_list = []
    for i in range(start_idx, end_idx+1, step):
        img1 = f'{i:06}.png'
        img_list.append(img1)
    return img_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_dir', default='../frames')
    parser.add_argument('--start_idx', type=int, default=1100)     # starting index of the image sequence
    parser.add_argument('--end_idx', type=int, default=1120)    # ending index of the image sequence
    parser.add_argument('--flow_dir', type=str, default='../flow_results')
    parser.add_argument('--viz_dir', type=str, default='../viz_results')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gen_flow', action='store_true')
    parser.add_argument('--H', type=int, default=576)
    parser.add_argument('--W', type=int, default=1024)

    args = parser.parse_args()
    args.device = 'cpu' if args.cpu else 'cuda'
    args.alpha = 0.3

    root_dir = args.seq_dir
    flow_dir = args.flow_dir
    viz_dir = args.viz_dir
    img_size = (args.W, args.H)

    img_list = generate_list(args.start_idx, args.end_idx, 1)
    
    if args.gen_flow:
        model = build_model(args.device)
        gen_flow(root_dir, flow_dir, model, img_list, img_size, args.device)
    else:
        if flow_dir is not None:
            model = None
        else:
            model = build_model(args.device)
        sd_flow(root_dir, flow_dir, viz_dir, model, img_list, args.alpha, img_size, args.device)
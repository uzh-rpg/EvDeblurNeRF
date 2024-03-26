import os
from options import config_parser

import cv2
import imageio
import numpy as np
from glob import glob

import torch
from torch import nn
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from data.sampler_image_batch import ImageBatchSampler

from networks.renderer import NeRFAll
from networks.pdrf.blurmodel import BlurModel
from networks.dpnerf.awp import AdaptiveWeightProposal
from networks.tonemapping import TonemappingTransform
from networks.dpnerf.blurmodel import RigidBlurringModel
from networks.embedding import ViewEmbedding, ViewEmbeddingMLP

from utils.logger import Logger
from utils.grads import grads_norm
from utils.metrics import compute_img_metric, img2mse, mse2psnr

from utils.events import egm_loss
from data.loader import LLFFDataset, endless
from data.loader_events import LLFFEventsDataset
from utils.misc import seed_everything, to8b, smart_load_state_dict, \
    exponential_scale_fine_loss_weight, annealing_interpolator


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.events_threshold_pos is None or args.events_threshold_neg is None:
        print(f"WARNING: overriding events_threshold_pos and events_threshold_neg "
              f"to events_threshold={args.events_threshold}")
        args.events_threshold_pos = args.events_threshold
        args.events_threshold_neg = args.events_threshold

    if len(args.torch_hub_dir) > 0:
        print(f"Change torch hub cache to {args.torch_hub_dir}")
        torch.hub.set_dir(args.torch_hub_dir)

    # Load data
    print(args)
    print('RANDOM SEED', args.seed)
    seed_everything(args.seed, deterministic=True, warn_only=True)

    if args.dataset_type == 'llff':
        llff_dataset = LLFFDataset(args, args.datadir, args.factor,
                                   recenter=True, bd_factor=args.bd_factor,
                                   spherify=args.spherify,
                                   path_epi=args.render_epi,
                                   exp_data_size=args.kernel_ptnum,
                                   pose_transform_allknown=args.pose_transform_allknown,
                                   device="cpu")

        if args.ray_sampling_mode == "random":
            sampler = BatchSampler(RandomSampler(llff_dataset, generator=torch.Generator(device='cuda')),
                                   batch_size=args.N_rand, drop_last=True)
        elif args.ray_sampling_mode == "images":
            sampler = ImageBatchSampler(llff_dataset, same_imgs_size=args.ray_sampling_images_num,
                                        batch_size=args.N_rand, num_imgs=llff_dataset.n_imgs,
                                        image_resolution=(llff_dataset.w, llff_dataset.h),
                                        generator=torch.Generator(device='cpu'))
        else:
            raise ValueError(f"Unknown ray_sampling_mode: {args.ray_sampling_mode}")

        if args.use_events:
            llffev_dataset = LLFFEventsDataset(args, args.datadir, llff_dataset.h, llff_dataset.w, llff_dataset.K,
                                               args.factor, recenter=True,
                                               bd_factor=args.bd_factor,
                                               bd_scale=llff_dataset.scale,
                                               closest_bds=llff_dataset.closest_bds,
                                               furthest_bds=llff_dataset.furthest_bds,
                                               spherify=args.spherify,
                                               recenter_partial=llff_dataset.recenter_partial,
                                               spherify_partial=llff_dataset.spherify_partial,
                                               events_tms_unit=args.events_tms_unit,
                                               events_tms_files_unit=args.events_tms_files_unit,
                                               color_events=args.event_egm_use_colorevents,
                                               device="cpu")
            train_ev_loader = DataLoader(
                llffev_dataset,
                # Use a batch sampler as the sampler so that __getitem__ is called with a list of indices
                sampler=BatchSampler(RandomSampler(llffev_dataset, generator=torch.Generator(device='cuda')),
                                     batch_size=args.events_N_rand, drop_last=True),
                # Use batch size None to disable auto-batching, but still use multiple workers to prefetch
                batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=16)

            events_threshold_negpos = torch.tensor([[args.events_threshold_neg, args.events_threshold_pos]],
                                                   dtype=torch.float32, device="cuda")

            if args.use_pts0_prior == "edi":
                llff_dataset.set_pts0_prior(llffev_dataset.compute_edi_prior(
                    llff_dataset.i_train, llff_dataset.images, args.pts0_edi_steps,
                    args.events_threshold_pos, args.events_threshold_neg))
        else:
            llffev_dataset, train_ev_loader = None, None
            events_threshold_negpos = None

        train_loader = DataLoader(
            llff_dataset, sampler=sampler,
            # Use batch size None to disable auto-batching, but still use multiple workers to prefetch
            batch_size=None, num_workers=8, pin_memory=True, prefetch_factor=8)

        train_iterator = iter(endless(train_loader))
        train_ev_iterator = iter(endless(train_ev_loader))

        args.bounding_box = llff_dataset.bounding_box
        near, far = llff_dataset.near, llff_dataset.far
        H, W = int(llff_dataset.h), int(llff_dataset.w)
        K = llff_dataset.K
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    w_events_egm = lambda x: None
    if args.use_events:
        w_events_egm = annealing_interpolator(args.event_egm_weight,
                                              args.event_egm_weight_end,
                                              args.event_egm_weight_steps,
                                              args.event_egm_weight_scheduler)

    w_pts0_target = lambda x: None
    if args.use_pts0_prior:
        w_pts0_target = annealing_interpolator(args.pts0_target_weight,
                                               args.pts0_target_weight_end,
                                               args.pts0_target_weight_steps,
                                               args.pts0_target_weight_scheduler)

    w_kernel = lambda x: 1.0
    kernel_end_warmup_iter = -1
    if args.kernel_start_warmup_mode != "step":
        kernel_end_warmup_iter = args.kernel_start_iter + args.kernel_start_warmup_iters
        w_kernel = annealing_interpolator(0.0, 1.0,
                                          kernel_end_warmup_iter,
                                          args.kernel_start_warmup_mode,
                                          start_step=args.kernel_start_iter)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    wandb_id = None
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n"
                       "||\n"
                       "\\/\n")

    if args.kernel_type != 'none':
        if args.kernel_img_embed_type == 'param':
            view_embed = ViewEmbedding(num_embed=llff_dataset.n_imgs, embed_dim=args.kernel_img_embed,
                                       init_params=args.kernel_img_embed_init)
        elif args.kernel_img_embed_type == 'param_mlp':
            view_embed = ViewEmbeddingMLP(num_embed=llff_dataset.n_imgs, embed_dim=args.kernel_img_embed,
                                          init_params=args.kernel_img_embed_init,
                                          D=args.kernel_img_mlp_depth, W=args.kernel_img_mlp_embed,
                                          skips=[args.kernel_img_mlp_skips])
        else:
            raise ValueError(f"Unknown kernel_img_embed_type: {args.kernel_img_embed_type}")
        view_embed_cnl = view_embed.out_channels
    else:
        view_embed, view_embed_cnl = None, 0

    # The DSK module
    if args.kernel_type == 'PBE' or args.kernel_type == 'DSK':
        kernelnet = BlurModel(
            llff_dataset.n_imgs,
            args.kernel_ptnum, args.kernel_hwindow, args.kernel_type,
            img_wh=[W, H],
            random_hwindow=args.kernel_random_hwindow,
            in_embed=args.kernel_rand_embed,
            random_mode=args.kernel_random_mode,
            spatial_embed=args.kernel_spatial_embed,
            depth_embed=args.kernel_depth_embed,
            num_hidden=args.kernel_num_hidden,
            num_wide=args.kernel_num_wide,
            feat_cnl=args.kernel_feat_cnl,
            short_cut=args.kernel_shortcut,
            pattern_init_radius=args.kernel_pattern_init_radius,
            isglobal=args.kernel_isglobal,
            optim_trans=args.kernel_global_trans,
            optim_spatialvariant_trans=args.kernel_spatialvariant_trans,
            view_embed_cnl=view_embed_cnl,
            view_embed=view_embed
        )
    elif args.kernel_type == 'RBK':
        kernelnet = RigidBlurringModel(
            feat_ch=args.kernel_rbk_extra_feat_ch, num_motion=args.kernel_ptnum - 1,
            D_r=args.kernel_rbk_se_r_depth, W_r=args.kernel_rbk_se_r_width,
            D_v=args.kernel_rbk_se_v_depth, W_v=args.kernel_rbk_se_v_width,
            D_w=args.kernel_rbk_ccw_depth, W_w=args.kernel_rbk_ccw_width,
            output_ch_r=args.kernel_rbk_se_r_output_ch,
            output_ch_v=args.kernel_rbk_se_v_output_ch,
            rv_window=args.kernel_rbk_se_rv_window,
            use_origin=args.kernel_rbk_use_origin,
            view_embed=view_embed, W=view_embed_cnl,
        )
    elif args.kernel_type == 'none':
        kernelnet = None
    else:
        raise RuntimeError(f"kernel_type {args.kernel_type} not recognized")

    if args.kernel_use_awp:
        awpnet = AdaptiveWeightProposal(
            input_ch=args.fine_geo_feat_dim if args.mode == 'c2f' else args.netwidth,
            num_motion=args.kernel_ptnum - 1, use_origin=True,
            D_sam=args.kernel_awp_sam_emb_depth, W_sam=args.kernel_awp_sam_emb_width,
            D_mot=args.kernel_awp_mot_emb_depth, W_mot=args.kernel_awp_mot_emb_width,
            dir_freq=args.kernel_awp_dir_freq, rgb_freq=args.kernel_awp_rgb_freq,
            depth_freq=args.kernel_awp_depth_freq, ray_dir_freq=args.kernel_awp_ray_dir_freq,
            view_feature_ch=view_embed_cnl)
    else:
        awpnet = None

    # Create camera(s) response function
    extra_features_event = 0 if args.tone_mapping_events_add_bii == "none" else 2
    crf = TonemappingTransform(map_type_rgb=args.tone_mapping_type,
                               map_type_event=args.tone_mapping_events_type,
                               extra_features_event=extra_features_event,
                               gamma=args.tone_mapping_gamma,
                               init_learn_identity=args.tone_mapping_learn_init_identity)

    # Create nerf model
    nerf = NeRFAll(args, kernelnet, awpnet)
    if args.mode == 'c2f':
        if args.colornet_weightdecay:
            optim_params = [
                {'params': nerf.get_parameters("net", match_re=r"\.color_net\.[0-9]+\.weight"),
                 'lr': args.lrate, 'weight_decay': args.colornet_weightdecay},
                {'params': nerf.get_parameters("net", not_match_re=r"\.color_net\.[0-9]+\.weight"),
                 'lr': args.lrate},
                {'params': nerf.grad_vars_vol, 'lr': args.lrate}]
        else:
            optim_params = [
                {'params': nerf.grad_vars, 'lr': args.lrate},
                {'params': nerf.grad_vars_vol, 'lr': args.lrate}]
    elif args.mode == 'nerf':
        optim_params = [
            {'params': nerf.parameters(), 'lr': args.lrate}]
    else:
        raise NotImplementedError(f"{args.mode} for rendering network is not implemented")

    optim_params += [{'params': crf.parameters(), 'lr': args.lrate}]

    # Stores the initial lr to remember it for later
    for group in optim_params:
        group.setdefault('initial_lr', group['lr'])

    # Scales the lr by the warmup factor
    if args.lrate_warmup_iters > 0:
        for group in optim_params:
            group['lr'] = group['lr'] * args.lrate_warmup_factor

    optimizer = torch.optim.Adam(params=optim_params,
                                 lr=args.lrate,
                                 betas=(0.9, 0.999))

    start = 0

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 '.tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        if llffev_dataset is not None:
            llffev_dataset.global_step = start
        wandb_id = ckpt['wandb_id'] if 'wandb_id' in ckpt else None

        # Load model
        smart_load_state_dict(nerf, ckpt, network_key="network_state_dict")
        smart_load_state_dict(crf, ckpt, network_key="crf_state_dict")
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    logger = Logger(log_dir=args.tbdir, expname=args.expname,
                    use_wandb=not args.no_wandb and not args.render_only,
                    use_tensorboard=args.use_tensorboard,
                    wandb_id=wandb_id,
                    args=args)

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'inference': False,
    }
    # NDC only good for LLFF-style forward facing data
    if args.no_ndc:  # args.dataset_type != 'llff' or
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['inference'] = True
    render_kwargs_test['raw_noise_std'] = 0.

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    global_step = start
    # Move testing data to GPU
    nerf = nerf.cuda()
    crf = crf.cuda()
    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            render_poses = llff_dataset.poses if args.render_test else llff_dataset.render_poses
            testsavedir = os.path.join(basedir, expname,
                                       f"renderonly"
                                       f"_{'test' if args.render_test else 'path'}"
                                       f"_{start:06d}")

            if os.path.exists(testsavedir):
                all_versions = sorted(glob(testsavedir + "_ver*"))
                if len(all_versions) == 0:
                    ver = 0
                else:
                    ver = max([int(p.split("_ver")[1]) for p in all_versions]) + 1
                testsavedir = testsavedir + f"_ver{ver}"

            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            dummy_num = ((len(render_poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(render_poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            torch.cuda.empty_cache()
            # Turn on testing mode
            with torch.no_grad():
                nerf.eval()
                crf.eval()
                rgbshdr, disps = nerf(
                    H, W, K, args.chunk // 2,
                    poses=torch.cat([render_poses, dummy_poses], dim=0),
                    render_kwargs=render_kwargs_test,
                    render_factor=args.render_factor,
                             )
            rgbshdr = crf(rgbshdr, mode="encode_rgb", chunk=8)

            rgbshdr = rgbshdr[:len(rgbshdr) - dummy_num]
            disps = (1. - disps)
            disps = disps[:len(disps) - dummy_num].cpu().numpy()
            rgbs = rgbshdr
            rgbs = rgbs.cpu().numpy()

            for rgb_idx, rgb in enumerate(rgbs):
                rgb8 = to8b(rgb)
                np.save(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.npy'), disps[rgb_idx])
                curr_disp = to8b(disps[rgb_idx] / disps[rgb_idx].max())
                imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}.png'), rgb8)
                imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_disp.png'),
                                cv2.applyColorMap(255 - curr_disp, cv2.COLORMAP_TWILIGHT_SHIFTED))

            prefix = 'epi_' if args.render_epi else ''
            imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video.mp4'), rgbs, fps=30, quality=9)
            disps = to8b(disps / disps.max())
            imageio.mimwrite(os.path.join(testsavedir, f'{prefix}video_disp.mp4'), disps, fps=30, quality=9)

            if args.render_test and args.render_multipoints:
                for pti in range(args.kernel_ptnum):
                    nerf.eval()
                    crf.eval()
                    poses_num = len(render_poses) + dummy_num
                    imgidx = torch.arange(poses_num, dtype=torch.long).to(render_poses.device).reshape(poses_num, 1)
                    rgbs, weights = nerf(
                        H, W, K, args.chunk // 2,
                        poses=torch.cat([render_poses, dummy_poses], dim=0),
                        render_kwargs=render_kwargs_test,
                        render_factor=args.render_factor,
                                 )
                    rgbs = crf(rgbs, mode="encode_rgb", chunk=8)

                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    weights = weights[:len(weights) - dummy_num]
                    rgbs = rgbs.cpu().numpy()
                    weights = to8b(weights.cpu().numpy())
                    for rgb_idx, rgb in enumerate(rgbs):
                        rgb8 = to8b(rgb)
                        imageio.imwrite(os.path.join(testsavedir, f'{rgb_idx:03d}_pt{pti}.png'), rgb8)
                        imageio.imwrite(os.path.join(testsavedir, f'w_{rgb_idx:03d}_pt{pti}.png'), weights[rgb_idx])
            return

    num_pts = args.kernel_ptnum
    fine_loss_weight = args.kernel_awp_fine_loss_start_ratio

    N_iters = args.N_iters + 1
    print('Begin')

    start = start + 1
    for i in trange(start, N_iters):
        is_last_iter = i == N_iters - 1

        #####  Core optimization loop  #####
        nerf.train()
        crf.train()

        if i == args.kernel_start_iter:
            torch.cuda.empty_cache()

        batch_data = next(train_iterator)
        batch_data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                      for k, v in batch_data.items()}

        use_pts0_loss = args.use_pts0_prior is not None and args.pts0_target_start_iter <= i < args.pts0_target_end_iter
        rgb, rgb0, extra_loss, extra_tensor = nerf(H, W, K, chunk=args.chunk, rays=batch_data["rays"],
                                                   rays_info=batch_data, retraw=True,
                                                   force_naive=i < args.kernel_start_iter,
                                                   return_pts0_rgb=global_step<kernel_end_warmup_iter or use_pts0_loss,
                                                   **render_kwargs_train)
        rgb = crf(rgb, mode="encode_rgb", skip_learn_crf=i<args.tone_mapping_start_learn_iter)
        rgb0 = crf(rgb0, mode="encode_rgb", skip_learn_crf=i<args.tone_mapping_start_learn_iter)

        # Compute Losses
        # =====================
        loss = 0.0
        target_rgb = batch_data['rgbsf'].squeeze(-2)

        if i > args.blur_loss_after:
            img_loss = img2mse(rgb, target_rgb)
            psnr = mse2psnr(img_loss)

            if rgb0 is not None:
                img_loss0 = img2mse(rgb0, target_rgb)
                img_loss = img_loss + img_loss0
            loss += img_loss
        else:
            img_loss = torch.tensor(0.0)
            psnr = torch.tensor(0.0)

        if 'rgb_awp' in extra_tensor and extra_tensor['rgb_awp'] is not None:
            img_fine_loss = img2mse(crf(extra_tensor['rgb_awp'], mode="encode_rgb",
                                        skip_learn_crf=i<args.tone_mapping_start_learn_iter), target_rgb)
            if args.kernel_awp_use_coarse_to_fine_opt:
                if i % 10000 == 0:
                    fine_loss_weight = exponential_scale_fine_loss_weight(
                        N_iters=N_iters, kernel_start_iter=args.kernel_start_iter,
                        start_ratio=0.1, end_ratio=0.9, iter=i)
                loss = loss * (1 - fine_loss_weight) + img_fine_loss * fine_loss_weight
            else:
                loss = loss + img_fine_loss

        if (args.kernel_start_warmup_mode != "step" and
            args.kernel_start_iter <= global_step < kernel_end_warmup_iter) or use_pts0_loss:
            pts0_loss = 0.0
            target_rgb_pts0 = target_rgb if not use_pts0_loss else batch_data['rgbsf_pts0'].squeeze(-2)
            # Directly apply the loss between the mid-exposure ray and the blur color, as done before kernel start
            for outname in ["stage0_rgb_pts0", "stage1_rgb_pts0", "stage1_rgb1_pts0"]:
                if outname in extra_tensor:
                    pts0_loss += img2mse(crf(extra_tensor[outname], mode="encode_rgb",
                                             skip_learn_crf=i<args.tone_mapping_start_learn_iter),
                                         target_rgb_pts0)

            extra_loss[f"pts0_{args.use_pts0_prior}_target"] = pts0_loss
            w_pts0_override = None
            if i <= args.blur_loss_after:  # print this psnr
                psnr = mse2psnr(extra_loss[f"pts0_{args.use_pts0_prior}_target"])
                w_pts0_override = 1.0

            if use_pts0_loss:
                w_pts0 = w_pts0_override if w_pts0_override is not None else w_pts0_target(global_step)
                loss = loss + extra_loss[f"pts0_{args.use_pts0_prior}_target"] * w_pts0
            else:
                # Interpolate between before-kernel-start mode and after-kernel-start mode
                loss = w_kernel(global_step) * loss + (1 - w_kernel(global_step)) * pts0_loss

        extra_loss.update({k: torch.mean(v) for k, v in extra_loss.items()})
        if "TV" in extra_loss:
            loss = loss + extra_loss["TV"] * args.kernel_tv_loss_weight
        if "align" in extra_loss:
            if args.align_start_iter <= i <= args.align_end_iter:
                loss = loss + extra_loss["align"] * args.kernel_align_weight

        ##############
        if args.add_event_egm and (args.add_event_egm_startiter is None or i >= args.add_event_egm_startiter):
            ev_batch_data = next(train_ev_iterator)
            ev_batch_data = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v
                             for k, v in ev_batch_data.items()}

            events_rays_start = ev_batch_data["events_rays_start"]
            events_rays_end = ev_batch_data["events_rays_end"]
            events_neg_pol_cumsum = ev_batch_data["events_neg_pol_cumsum"]
            events_pos_pol_cumsum = ev_batch_data["events_pos_pol_cumsum"]
            events_color_map = ev_batch_data["events_color_map"]

            cumsum_pols = torch.stack([events_neg_pol_cumsum, events_pos_pol_cumsum], dim=-1)
            bii = (events_threshold_negpos * cumsum_pols).sum(-1)  # [N,2] -> [N]

            ev_crf_kwargs = {"tonemap_only": True} if args.event_egm_use_colorevents else {}
            if args.tone_mapping_events_add_bii == 'pos-neg':
                ev_crf_extra_feat = torch.stack([events_neg_pol_cumsum, events_pos_pol_cumsum], dim=-1)
            elif args.tone_mapping_events_add_bii == 'color-pos-neg':
                color_events_neg_pol_cumsum = events_neg_pol_cumsum.new_zeros([events_color_map.shape[0], 3])
                color_events_pos_pol_cumsum = events_pos_pol_cumsum.new_zeros([events_color_map.shape[0], 3])
                color_events_neg_pol_cumsum[events_color_map] = events_neg_pol_cumsum
                color_events_pos_pol_cumsum[events_color_map] = events_pos_pol_cumsum
                ev_crf_extra_feat = torch.stack([color_events_neg_pol_cumsum, color_events_pos_pol_cumsum],
                                                dim=-1)
            else:
                ev_crf_extra_feat = None

            ev_start_rgb, ev_start_rgb0, start_extra_loss, start_extra_tensor = nerf(
                H, W, K, chunk=args.chunk,
                rays=events_rays_start, rays_info=None,
                retraw=True, force_naive=True,  # Does not use the kernel network
                **render_kwargs_train)
            ev_start_luma = crf(ev_start_rgb, mode="encode_luma",
                                skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                ev_extra_feat=ev_crf_extra_feat, **ev_crf_kwargs)
            ev_start_luma0 = crf(ev_start_rgb0, mode="encode_luma",
                                 skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                 ev_extra_feat=ev_crf_extra_feat,
                                 **ev_crf_kwargs)

            ev_end_rgb, ev_end_rgb0, end_extra_loss, end_extra_tensor = nerf(
                H, W, K, chunk=args.chunk,
                rays=events_rays_end, rays_info=None,
                retraw=True, force_naive=True,  # Does not use the kernel network
                **render_kwargs_train)
            ev_end_luma = crf(ev_end_rgb, mode="encode_luma",
                              skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                              ev_extra_feat=ev_crf_extra_feat, **ev_crf_kwargs)
            ev_end_luma0 = crf(ev_end_rgb0, mode="encode_luma",
                               skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                               ev_extra_feat=ev_crf_extra_feat, **ev_crf_kwargs)

            event_egm_parts = []
            if ev_start_rgb0 is not None and ev_end_rgb0 is not None:
                if "stage0" in args.add_event_egm_stages:
                    event_egm_parts.append(egm_loss(ev_start_luma0, ev_end_luma0, bii, color_mask=events_color_map,
                                                    color_weight=args.event_egm_use_color_weights
                                                    if i > args.event_egm_color_weights_start_iter else None))
            if "stage1" in args.add_event_egm_stages:
                event_egm_parts.append(egm_loss(ev_start_luma, ev_end_luma, bii, color_mask=events_color_map,
                                                color_weight=args.event_egm_use_color_weights
                                                if i > args.event_egm_color_weights_start_iter else None))

            extra_loss["event_egm"] = sum(event_egm_parts)

            if args.event_egm_use_awp and 'rgb_awp' in start_extra_tensor and 'rgb_awp' in end_extra_tensor:
                awp_start_luma = crf(start_extra_tensor['rgb_awp'], mode="encode_luma",
                                     skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                     ev_extra_feat=ev_crf_extra_feat,
                                     **ev_crf_kwargs)
                awp_end_luma = crf(end_extra_tensor['rgb_awp'], mode="encode_luma",
                                   skip_learn_crf=i<args.tone_mapping_start_learn_iter,
                                   ev_extra_feat=ev_crf_extra_feat,
                                   **ev_crf_kwargs)

                awp_egm = egm_loss(awp_start_luma, awp_end_luma, bii, color_mask=events_color_map,
                                   color_weight=args.event_egm_use_color_weights
                                   if i > args.event_egm_color_weights_start_iter else None)
                if args.event_egm_awp_use_coarse_to_fine_opt:
                    extra_loss["event_egm"] = extra_loss["event_egm"] * (1 - fine_loss_weight) + \
                                              awp_egm * fine_loss_weight
                else:
                    extra_loss["event_egm"] = extra_loss["event_egm"] + awp_egm

            loss += extra_loss["event_egm"] * w_events_egm(global_step)

        optimizer.zero_grad()
        loss.backward()

        if args.clip_grads_norm is not None:
            nn.utils.clip_grad_norm_(nerf.parameters(),
                                     max_norm=args.clip_grads_norm,
                                     norm_type=2)

        optimizer.step()

        ###   update learning rate   ###
        if args.lrate_warmup_iters > 0 and global_step < args.lrate_warmup_iters:
            scale = (1 - args.lrate_warmup_factor) * global_step / args.lrate_warmup_iters + args.lrate_warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * scale
        else:
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            for param_group in optimizer.param_groups:
                new_lrate = param_group['initial_lr'] * (decay_rate ** (global_step / decay_steps))
                param_group['lr'] = new_lrate
        ################################

        # Rest is logging
        if (i % args.i_weights == 0 and i > 0) or is_last_iter:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if os.path.exists(path):
                # Encapsulates '[' in brackets to escape, otherwise it will be interpreted as a character set
                ver_path = sorted(glob(os.path.join(basedir, expname, '{:06d}_ver*.tar'.format(i))
                                       .replace('[', '[[]')))
                latest_ver = max([int(os.path.basename(p).split('_ver')[-1].split('.')[0]) for p in ver_path]) \
                    if len(ver_path) > 0 else 0
                path = os.path.join(basedir, expname, '{:06d}_ver{:02d}.tar'.format(i, latest_ver + 1))

            if not os.path.exists(path):
                torch.save({
                    'wandb_id': wandb_id,
                    'global_step': global_step,
                    'crf_state_dict': crf.state_dict(),
                    'network_state_dict': nerf.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
            else:
                # Versioning did not work for some reason, we avoid overwriting the checkpoint
                print('Checkpoint already exists at', path)

        ######################################

        if (i % args.i_testset == 0 and i > 0) or is_last_iter:
            torch.cuda.empty_cache()
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            poses = llff_dataset.test_poses
            target_rgb_ldr = llff_dataset.test_images

            print('test poses shape', poses.shape)
            dummy_num = ((len(poses) - 1) // args.num_gpu + 1) * args.num_gpu - len(poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(poses)
            print(f"Append {dummy_num} # of poses to fill all the GPUs")
            with torch.no_grad():
                nerf.eval()
                crf.eval()

                rgbs, disps = nerf(H, W, K, args.chunk // 2, poses=torch.cat([poses, dummy_poses], dim=0),
                                   render_kwargs=render_kwargs_test)
                rgbs = crf(rgbs, mode="encode_rgb", chunk=8)
                rgbs = rgbs[:len(rgbs) - dummy_num]
                rgbs_save = rgbs  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
                disps = (1. - disps)

                for j, (rgb, gtrgb, disp) in enumerate(zip(rgbs, target_rgb_ldr, disps)):
                    assert rgb.shape == gtrgb.shape and len(rgb.shape) == 3
                    rgb = rgb.cpu().numpy()
                    disp = disp.cpu().numpy()
                    gtrgb = gtrgb.cpu().numpy()
                    pixmse = ((rgb - gtrgb) ** 2).mean(-1)
                    if i == args.i_testset:  # Only save at the first validation
                        logger.image(f"images/test_groundtruth_{j}", to8b(gtrgb), step=global_step)
                    logger.image(f"images/test_prediction_{j}", to8b(rgb), step=global_step)
                    logger.image(f"images/test_depth_{j}",
                                 cv2.applyColorMap(255 - to8b(disp / float(disps.max())),
                                                   cv2.COLORMAP_TWILIGHT_SHIFTED),
                                 step=global_step)
                    logger.image(f"images/test_errmap_{j}",
                                 cv2.applyColorMap(255 - to8b(pixmse / float(pixmse.max())),
                                                   cv2.COLORMAP_TWILIGHT_SHIFTED),
                                 step=global_step)

                metrics_str = ""
                # evaluation
                test_mse = compute_img_metric(rgbs, target_rgb_ldr, 'mse')
                test_psnr = compute_img_metric(rgbs, target_rgb_ldr, 'psnr')
                test_ssim = compute_img_metric(rgbs, target_rgb_ldr, 'ssim')
                test_lpips = compute_img_metric(rgbs, target_rgb_ldr, 'lpips')
                if isinstance(test_lpips, torch.Tensor):
                    test_lpips = test_lpips.item()

                logger.scalar("test/mse", test_mse, step=global_step)
                logger.scalar("test/psnr", test_psnr, step=global_step)
                logger.scalar("test/ssim", test_ssim, step=global_step)
                logger.scalar("test/lpips", test_lpips, step=global_step)
                metrics_str += f"MSE:{test_mse:.8f} PSNR:{test_psnr:.8f} " \
                               f"SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}"

                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"iter{i}/globalstep{global_step}: {metrics_str}\n")
                print(f"[TEST]  Iter: {i} {metrics_str}")

                for rgb_idx, rgb in enumerate(rgbs_save):
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, f'{rgb_idx:03d}.png')
                    imageio.imwrite(filename, rgb8)

            torch.cuda.empty_cache()
            print('Saved test set')

        if (i % args.i_video == 0 and i > 0) or is_last_iter:
            torch.cuda.empty_cache()
            # Turn on testing mode
            torch.cuda.empty_cache()
            # Turn on testing mode
            with torch.no_grad():
                nerf.eval()
                crf.eval()
                render_poses = llff_dataset.poses if args.render_test else llff_dataset.render_poses
                rgbs, disps = nerf(H, W, K, args.chunk // 2, poses=render_poses, render_kwargs=render_kwargs_test)
                lumas = crf(rgbs, mode="encode_luma", chunk=8)  # Zero-pad the CRF if learned with extra bii features
                rgbs = crf(rgbs, mode="encode_rgb", chunk=8)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))

            rgbs = (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
            rgbs = rgbs.cpu().numpy()
            disps = disps.cpu().numpy()

            logger.video(f"test/spiral_rgb", moviebase + 'rgb.mp4', to8b(rgbs),
                         fps=30, step=global_step)
            logger.video(f"test/spiral_disp", moviebase + 'disp.mp4', to8b(disps / disps.max()),
                         fps=30, step=global_step)
            torch.cuda.empty_cache()

        if i % args.i_tensorboard == 0 or is_last_iter:
            if not args.no_log_grads_norm:
                for k, v in grads_norm(nerf).items():
                    logger.scalar(f"gradients/{k}", float(v), global_step)

            logger.scalar("train/loss", loss.item(), global_step)
            logger.scalar("train/loss_img", img_loss.item(), global_step)
            for k, v in extra_loss.items():
                logger.scalar(f"train/{k}", v.item(), global_step)

            if args.kernel_start_warmup_mode != "step":
                logger.scalar(f"train/w_kernel", w_kernel(global_step), global_step)

            if args.use_pts0_prior:
                logger.scalar(f"train/dataset_global_step", w_pts0_target(global_step), global_step)

            if args.use_events:
                if args.event_accumulate_step_scheduler != "constant":
                    # Reads the internal dataset global step to make sure
                    # the value is the one actually applied by the loader
                    dataset_global_step = llffev_dataset.global_step
                    logger.scalar(f"train/dataset_global_step", dataset_global_step, global_step)
                    logger.scalar(f"train/event_accum_min", llffev_dataset.event_accum_min_step(
                        dataset_global_step), global_step)
                    logger.scalar(f"train/event_accum_max", llffev_dataset.event_accum_max_step(
                        dataset_global_step), global_step)
                if w_events_egm is not None:
                    logger.scalar(f"train/w_events_egm", w_events_egm(global_step), global_step)
                if events_threshold_negpos is not None:
                    events_threshold_negpos_neg = events_threshold_negpos[..., 0].mean()
                    events_threshold_negpos_pos = events_threshold_negpos[..., 1].mean()
                    logger.scalar(f"train/events_threshold_negpos_neg",
                                  events_threshold_negpos_neg.float().item(), global_step)
                    logger.scalar(f"train/events_threshold_negpos_pos",
                                  events_threshold_negpos_pos.float().item(), global_step)

        if i % args.i_print == 0 or is_last_iter:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

import os
import configargparse

# Limits the number of threads to avoid using all available CPU cores
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', required=True,
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, required=True,
                        help='input data directory')

    parser.add_argument("--datadownsample", type=float, default=-1,
                        help='if downsample > 0, means downsample the image to scale=datadownsample')
    parser.add_argument("--tbdir", type=str, required=True,
                        help="tensorboard log directory")
    parser.add_argument("--no_wandb", action="store_true",
                        help="whether to disable wandb")
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="whether enable tensorboard logging, disabled by default")

    parser.add_argument("--num_gpu", type=int, default=1,
                        help=">1 will use DataParallel")
    parser.add_argument("--torch_hub_dir", type=str, default='',
                        help=">1 will use DataParallel")
    parser.add_argument("--no_log_grads_norm", action="store_true",
                        help="whether to disable logging of the gradient's norm")
    parser.add_argument("--clip_grads_norm", type=float, default=None,
                        help="The maximum value of the total L2 norm of the gradients")

    # ===============================
    # Training options
    # ===============================
    parser.add_argument("--seed", type=int, default=0,
                        help='random seed')
    parser.add_argument("--mode", type=str, default='c2f', required=True,
                        help='choose bewteen c2f (CRR+FVR) or nerf (2 MLPs) for rendering networks')
    parser.add_argument("--ray_sampling_mode", choices=["random", "images"], default="random",
                        help="controls if, during training, rays are sampled from all available images "
                             "(random), or from a set of random images (images)")
    parser.add_argument("--ray_sampling_images_num", type=int, default=32,
                        help="when sampling mode is 'images', controls from how many images rays are sample each time")
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_warmup_factor", type=float, default=0.1,
                        help='learning rate warmup from lrate * lrate_warmup_factor to lrate')
    parser.add_argument("--lrate_warmup_iters", type=float, default=-1,
                        help='learning rate warmup in lrate_warmup_iters iterations')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--colornet_weightdecay", type=float, default=None,  # 0.0002
                        help='L2 weight decay to apply on color net\'s weights')

    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--N_iters", type=int, default=50000,
                        help='number of iteration')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--rgb_add_bias", action="store_true",
                        help='whether to use bias in color net linear layers')
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplus"')

    parser.add_argument("--dataset_type", type=str, default='llff', choices=['llff'],
                        help='options: llff')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--factor", type=int, default=None,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--pose_transform_allknown", action='store_true',
                        help='whether to compute pose transformation only from image data, or from all known poses')
    parser.add_argument("--bd_factor", type=float, default=0.75,
                        help='factor to rescale pose bounds (default: 0.75)')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--llffhold_end", action="store_true",
                        help='modifies llffhold to take the last N images as test set rather than one every N')

    # ===============================
    # CRR/FVR options
    # ===============================
    parser.add_argument("--coarse_num_layers", type=int, default=2,
                        help='CRR layer for estimating sigma + feature')
    parser.add_argument("--coarse_num_layers_color", type=int, default=3,
                        help='CRR layer for estimating color')
    parser.add_argument("--coarse_hidden_dim", type=int, default=64,
                        help='coarse_hidden_dim')
    parser.add_argument("--coarse_hidden_dim_color", type=int, default=64,
                        help='coarse_hidden_dim_color')
    parser.add_argument("--coarse_app_dim", type=int, default=32,
                        help='coarse_app_dim')
    parser.add_argument("--coarse_app_n_comp", type=int, action="append")
    parser.add_argument("--coarse_n_voxels", type=int, default=16777248,
                        help='coarse_n_voxels')
    parser.add_argument("--coarse_app_actfn", type=str, default="none")

    parser.add_argument("--fine_num_layers", type=int, default=2,
                        help='FVR layer for estimating sigma + feature')
    parser.add_argument("--fine_num_layers_color", type=int, default=3,
                        help='FVR layer for estimating color')
    parser.add_argument("--fine_hidden_dim", type=int, default=256,
                        help='fine_hidden_dim')
    parser.add_argument("--fine_hidden_dim_color", type=int, default=256,
                        help='fine_hidden_dim_color')
    parser.add_argument("--fine_app_dim", type=int, default=32,
                        help='fine_app_dim')
    parser.add_argument("--fine_geo_feat_dim", type=int, default=128,
                        help='fine_geo_feat_dim')
    parser.add_argument("--fine_app_n_comp", type=int, action="append")
    parser.add_argument("--fine_app_actfn", type=str, default="none")
    parser.add_argument("--fine_n_voxels", type=int, default=134217984,
                        help='fine_n_voxels')

    # ===============================
    # Events optimizing
    # ===============================
    parser.add_argument("--use_pts0_prior", choices=["edi"], default=None,
                        help="whether to add a loss between the color predicted by the mid-rays and a target image")
    parser.add_argument("--pts0_edi_steps", type=int, default=9,
                        help="number of steps to use in the EDI computation")
    parser.add_argument("--pts0_target_weight", type=float, default=0.1,
                        help="weight of the pts0_target loss")
    parser.add_argument("--pts0_target_weight_end", type=float, default=1.0,
                        help="weight of the pts0_target loss at the end of the training")
    parser.add_argument("--pts0_target_weight_steps", type=int, default=None,
                        help="number of steps to linearly increase the pts0_target loss weight")
    parser.add_argument("--pts0_target_weight_scheduler", choices=["constant", "linear", "cosine"],
                        default="constant", help="scheduler to use for the pts0_target loss weight")
    parser.add_argument("--pts0_target_start_iter", type=int, default=-1,
                        help="Iterations after which applying the pts0_target loss")
    parser.add_argument("--pts0_target_end_iter", type=int, default=9999999,
                        help="Iterations after which the pts0_target loss is not used anymore")

    parser.add_argument("--use_events", action="store_true")
    parser.add_argument("--tone_mapping_events_type", choices=['gamma', 'learn', 'none'], default='none')
    parser.add_argument("--tone_mapping_events_add_bii", choices=['none', 'pos-neg', 'color-pos-neg'],
                        default='none')
    parser.add_argument("--events_tms_unit", default="ns", choices=["ns", "us"])
    parser.add_argument("--events_tms_files_unit", default="us", choices=["ns", "us"])

    parser.add_argument("--events_N_rand", type=int, default=32 * 32 * 4 // 2)
    parser.add_argument("--events_threshold", type=float, default=0.2)
    parser.add_argument("--events_threshold_pos", type=float, default=None)
    parser.add_argument("--events_threshold_neg", type=float, default=None)

    parser.add_argument("--add_event_egm", action="store_true")
    parser.add_argument("--event_egm_use_colorevents", action="store_true")
    parser.add_argument("--event_egm_use_color_weights", type=float, nargs=3, default=None)
    parser.add_argument("--event_egm_color_weights_start_iter", type=int, default=-1)
    parser.add_argument("--event_egm_use_awp", action="store_true")
    parser.add_argument("--event_egm_awp_use_coarse_to_fine_opt", action="store_true")
    parser.add_argument("--add_event_egm_stages", nargs="+",
                        choices=["stage0", "stage1"], default=["stage0"])
    parser.add_argument("--add_event_egm_startiter", type=int, default=None)

    parser.add_argument("--event_accumulate_step_range", nargs=2, type=int, default=[0, 0])
    parser.add_argument("--event_accumulate_step_range_end", nargs=2, type=int, default=[0, 0])
    parser.add_argument("--event_accumulate_step_scheduler", choices=["constant", "linear", "cosine"],
                        default="constant")
    parser.add_argument("--event_accumulate_step_end", type=int, default=0)
    parser.add_argument("--event_egm_weight", type=float, default=1.0)
    parser.add_argument("--event_egm_weight_end", type=float, default=1.0)
    parser.add_argument("--event_egm_weight_steps", type=int, default=None)
    parser.add_argument("--event_egm_weight_scheduler", choices=["constant", "linear", "cosine"],
                        default="constant")

    # ===============================
    # Kernel optimizing
    # ===============================
    parser.add_argument("--blur_loss_after", type=int, default=-1,
                        help="Iterations after which applying the blur loss")
    parser.add_argument("--kernel_type", type=str, default='kernel',
                        help='choose among <none>, <DSK>, <PBE>, <RBK>')
    parser.add_argument("--kernel_isglobal", action='store_true',
                        help='if specified, the canonical kernel position is global')
    parser.add_argument("--kernel_start_iter", type=int, default=0,
                        help='start training kernel after # iteration')
    parser.add_argument("--kernel_start_warmup_mode", choices=["step", "cosine", "linear"], default="step",
                        help='whether there is a scheduling to add the loss (from 0 to 1 weight), and which type')
    parser.add_argument("--kernel_start_warmup_iters", type=int, default=1,
                        help="if scheduling is selected, how many iterations it'll take to fully introduce the kernel")
    parser.add_argument("--kernel_ptnum", type=int, default=5,
                        help='the number of sparse locations in the kernels '
                             'that involves computing the final color of ray')
    parser.add_argument("--kernel_random_hwindow", type=float, default=0.25,
                        help='randomly displace the predicted ray position')
    parser.add_argument("--kernel_img_embed_type", choices=["param", "param_mlp"], default="param",
                        help='whether the image embedding is purely parametric or also modulated by an MLP')
    parser.add_argument("--kernel_img_embed_init", choices=["zero", "normal", "linspace"], default="zero",
                        help='init function used to initialize the parametric image latent code')
    parser.add_argument("--kernel_img_embed", type=int, default=32,
                        help='the dim of parametric image latent code (before MLP if kernel_img_embed_type=param_mlp)')
    parser.add_argument("--kernel_img_mlp_embed", type=int, default=32,
                        help='the out and hidden dim of image latent mlp, if kernel_img_embed_type=param_mlp')
    parser.add_argument("--kernel_img_mlp_depth", type=int, default=4,
                        help='the depth of image latent mlp, if kernel_img_embed_type=param_mlp')
    parser.add_argument("--kernel_img_mlp_skips", type=int, default=4,
                        help='the image latent mlp network skip connection')
    parser.add_argument("--kernel_feat_cnl", type=int, default=15,
                        help='the dim of radiance field latent code')
    parser.add_argument("--kernel_rand_dim", type=int, default=2,
                        help='dimensions of input random number which uniformly sample from (0, 1)')
    parser.add_argument("--kernel_rand_embed", type=int, default=3,
                        help='embed frequency of input kernel coordinate')
    parser.add_argument("--kernel_random_mode", type=str, default='input', choices=['input', 'output'],
                        help='<input>, <output>')
    parser.add_argument("--kernel_spatial_embed", type=int, default=0,
                        help='the dim of spatial coordinate embedding')
    parser.add_argument("--kernel_depth_embed", type=int, default=0,
                        help='the dim of depth coordinate embedding')
    parser.add_argument("--kernel_hwindow", type=int, default=10,
                        help='the max window of the kernel (sparse location will lie inside the window')
    parser.add_argument("--kernel_pattern_init_radius", type=float, default=0.1,
                        help='the initialize radius of init pattern')
    parser.add_argument("--kernel_num_hidden", type=int, default=3,
                        help='the number of hidden layer')
    parser.add_argument("--kernel_num_wide", type=int, default=64,
                        help='the wide of hidden layer')
    parser.add_argument("--kernel_shortcut", action='store_true',
                        help='if yes, add a short cut to the network')
    parser.add_argument("--align_start_iter", type=int, default=0,
                        help='start iteration of the align loss')
    parser.add_argument("--align_end_iter", type=int, default=1e10,
                        help='end iteration of the align loss')
    parser.add_argument("--kernel_align_weight", type=float, default=0,
                        help='align term weight')
    parser.add_argument("--kernel_tv_loss_weight", type=float, default=1.0,
                        help="weight for total variation loss")
    parser.add_argument("--kernel_spatialvariant_trans", action='store_true',
                        help='if true, optimize spatial variant 3D translation of each sampling point')
    parser.add_argument("--kernel_global_trans", action='store_true',
                        help='if true, optimize global 3D translation of each sampling point')
    parser.add_argument("--kernel_rbk_extra_feat_ch", type=int, default=15,
                        help='additional features ch')
    parser.add_argument("--kernel_rbk_use_viewdirs", action='store_true',
                        help='use viewdirs in rbk')
    parser.add_argument("--kernel_rbk_enc_brc_skips", type=int, default=4,
                        help='rbk encoding  network skip connection')
    parser.add_argument("--kernel_rbk_se_r_depth", type=int, default=1,
                        help='rbk se3 r network depth')
    parser.add_argument("--kernel_rbk_se_r_width", type=int, default=32,
                        help='rbk se3 r network width')
    parser.add_argument("--kernel_rbk_se_r_output_ch", type=int, default=3,
                        help='rbk se3 r network output channel')
    parser.add_argument("--kernel_rbk_se_v_depth", type=int, default=1,
                        help='rbk se3 v network depth')
    parser.add_argument("--kernel_rbk_se_v_width", type=int, default=32,
                        help='rbk se3 v network width')
    parser.add_argument("--kernel_rbk_se_v_output_ch", type=int, default=3,
                        help='rbk se3 v network output channel')
    parser.add_argument("--kernel_rbk_ccw_depth", type=int, default=1,
                        help='rbk ccw network depth')
    parser.add_argument("--kernel_rbk_ccw_width", type=int, default=32,
                        help='rbk ccw network width')
    parser.add_argument("--kernel_rbk_se_rv_window", type=float, default=0.2,
                        help='rbk se3 rv network output scale window')
    parser.add_argument("--kernel_rbk_use_origin", action='store_true',
                        help='use original ray in rbk module')
    parser.add_argument("--kernel_rbk_feature_extractor_type", choices=["resnet18", "resnet34"],
                        help="which feature extractor to use for the additional features")
    parser.add_argument("--kernel_rbk_feature_extractor_pretrained", action='store_true',
                        help="whether to use a pretrained feature extractor")
    parser.add_argument("--kernel_rbk_feature_extractor_process_views_separately", action='store_true',
                        help="whether to process each view separately in the feature extractor")

    parser.add_argument("--kernel_use_awp", action='store_true',
                        help='use awp module')
    parser.add_argument("--kernel_awp_use_coarse_to_fine_opt", action='store_true',
                        help='use_coarse_to_fine_optimization')
    parser.add_argument("--kernel_awp_fine_loss_start_ratio", type=float, default=0.1,
                        help='start weight of the coarse to fine loss')
    parser.add_argument("--kernel_awp_fine_loss_end_ratio", type=float, default=0.9,
                        help='end weight of the coarse to fine loss')

    parser.add_argument("--kernel_awp_sam_emb_depth", type=int, default=4,
                        help='awp sample feature embedding layer depth')
    parser.add_argument("--kernel_awp_sam_emb_width", type=int, default=32,
                        help='awp sample feature embedding layer width')
    parser.add_argument("--kernel_awp_dir_freq", type=int, default=2,
                        help='awp dir fourier embedding freq')
    parser.add_argument("--kernel_awp_mot_emb_depth", type=int, default=1,
                        help='awp motion feature embedding layer depth')
    parser.add_argument("--kernel_awp_mot_emb_width", type=int, default=32,
                        help='awp motion feature embedding layer depth')
    parser.add_argument("--kernel_awp_rgb_freq", type=int, default=2,
                        help='awp rgb freq')
    parser.add_argument("--kernel_awp_depth_freq", type=int, default=2,
                        help='awp depth freq')
    parser.add_argument("--kernel_awp_ray_dir_freq", type=int, default=2,
                        help='awp network ray dir freq')

    parser.add_argument("--tone_mapping_type", type=str, choices=['none', 'gamma'], default='none',
                        help='the tone mapping of linear to LDR color space, <none>, <gamma>')
    parser.add_argument("--tone_mapping_start_learn_iter", type=int, default=0,
                        help='start iteration of the tone mapping learn loss')
    parser.add_argument("--tone_mapping_learn_init_identity", action='store_true',
                        help='init the learnable tone mapping with identity')
    parser.add_argument("--tone_mapping_gamma", type=float, default=2.2,
                        help='the gamma encoding to be applied if \'gamma\' in tone_mapping_type')

    # ===============================
    # Render options
    # ===============================
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_multipoints", action='store_true',
                        help='render sub image that reconstruct the blur image')
    parser.add_argument("--render_rmnearplane", type=int, default=0,
                        help='when render, set the density of nearest plane to 0')
    parser.add_argument("--render_focuspoint_scale", type=float, default=1.,
                        help='scale the focal point when render')
    parser.add_argument("--render_radius_scale", type=float, default=1.,
                        help='scale the radius of the camera path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_epi", action='store_true',
                        help='render the video with epi path')

    # ===============================
    #
    # Logging/saving options
    # ===============================
    parser.add_argument("--i_print", type=int, default=200,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_tensorboard", type=int, default=200,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=25000,
                        help='frequency of render_poses video saving')

    return parser

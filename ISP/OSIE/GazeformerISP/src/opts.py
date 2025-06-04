from __future__ import print_function
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")

    parser.add_argument("--log_root", type=str, help="Log root")
    parser.add_argument("--user_emb_path", default="", type=str, help="Log root")
    parser.add_argument("--ex_subject", nargs='+', type=int, default=[-1], help='A list of exclude subjects')
    parser.add_argument("--fewshot_subject", nargs='+', type=int, default=[-1], help="subject ids for few-shot learning")
    parser.add_argument("--num_fewshot", type=int, default=20, help="number of images from the new subject in few-shot learning")
    parser.add_argument("--subject_num", type=int, default=15, help="The number of the subject in OSIE")
    parser.add_argument("--random_support", type=int, default=100, help="The number of the subject in OSIE")
    parser.add_argument("--mode", type=str, default="train", help="Selecting running mode (default: train)")
    parser.add_argument("--img_dir", type=str, default="../../saliency/data/OSIE/train", help="Directory to the image data (stimuli)")
    parser.add_argument("--feat_dir", type=str, default="../../../ISP_back/OSIE/GazeformerISP/src/data/OSIE/image_features",
                        help="Directory to the image feature data (stimuli)")
    parser.add_argument("--emb_dir", type=str, default="src/data/embeddings.npy",
                        help="Directory to the image feature data (stimuli)")
    parser.add_argument("--fix_dir", type=str, default="src/data/fixations.json", help="Directory to the raw fixation file")
    parser.add_argument("--width", type=int, default=512, help="Width of input data")
    parser.add_argument("--height", type=int, default=384, help="Height of input data")
    parser.add_argument("--origin_width", type=int, default=800, help="original Width of input data")
    parser.add_argument("--origin_height", type=int, default=600, help="original Height of input data")
    parser.add_argument('--im_h', default=24, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
    parser.add_argument("--clip", type=float, default=-1, help="Gradient clipping")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--test_batch", type=int, default=1, help="Batch size")
    parser.add_argument("--epoch", type=int, default=40, help="Number of epochs")
    parser.add_argument("--warmup_epoch", type=int, default=1, help="Epoch when finishing warn up strategy")
    parser.add_argument("--start_rl_epoch", type=int, default=25, help="Epoch when starting reinforcement learning")
    parser.add_argument("--rl_sample_number", type=int, default=5,
                        help="Number of samples used in policy gradient update")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate for MidOpt")
    parser.add_argument("--rl_lr_initial_decay", type=float, default=0.1, help="Initial decay of learning rate of rl")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")
    parser.add_argument("--gpu_ids", type=list, default=[3], help="Used gpu ids")
    # parser.add_argument("--gpu_ids", type=list, default=[0], help="Used gpu ids")
    
    parser.add_argument("--resume_dir", type=str, default="", help="Resume from a specific directory")
    parser.add_argument("--eval_repeat_num", type=int, default=1, help="Repeat number for evaluation")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")

    parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
    parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
    parser.add_argument('--lm_hidden_dim', default=768, type=int,
                        help="Dimensionality of target embeddings from language model")
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help="Encoder dropout rate")
    parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
    parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")

    parser.add_argument('--cuda', default=3, type=int, help="CUDA core to load models and data")
    parser.add_argument("--subject_feature_dim", type=int, default=384, help="The dim of the subject feature")
    parser.add_argument("--action_map_num", type=int, default=4, help="The dim of action map")
    parser.add_argument("--no_eval_epoch", type=int, default=1, help="The number of no evaluation epoch")

    parser.add_argument("--lambda_1", type=float, default=1.0, help="Hyper-parameter for duration loss term")
    parser.add_argument("--supervised_save", type=bool, default=False,
                        help="Copy the files before start the policy gradient update")

    # config
    parser.add_argument('--cfg', type=str, default=None,
                        help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from utils.config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    return args

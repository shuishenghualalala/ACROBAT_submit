import argparse

def ACROBAT_args():
    parser = argparse.ArgumentParser('ACROBAT')
    parser.add_argument('--exp', type=str, default='0')

    # Data
    parser.add_argument('--Pair_path_validate', default=None,
                        help='Dic file of image pairs')
    parser.add_argument('--Pair_path', default=None,
                        help='Dic file of image pairs')
    parser.add_argument('--Pair_path_test', default=None,
                        help='Dic file of image pairs')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for dataloader(default=32)')
    parser.add_argument('--batch_size_eval', type=int, default=32,
                        help='batch size for dataloader(default=32)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number workers for dataloader(default=8)')
    # model
    parser.add_argument('--modelname', type=str, default='Affine',
                        help='model name for train(default = Affine/Deformation)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for load model and data(default=cuda:0)')


    # loss
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='weight for curvature_regularization loss')
    parser.add_argument('--alpha', type=float, default=1.0,
                            help='weight for curv loss')
    # log
    parser.add_argument('--cpt', type=str, default=None,help='path for checkpoints')

    # train
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--purge_step', type=int, default=0)
    args = parser.parse_args()
    return args

import argparse


def form_expr_name(args):
    if args.pointcloud:
        type_str = 'PC' + str(args.pc_samples)
    else:
        type_str = 'Voxels'

    dist_str = ''.join(str(e) + '_' for e in args.sample_distribution)
    sigmas_str = ''.join(str(e) + '_' for e in args.sample_sigmas)

    exp_name = f'i{type_str}_dist-{dist_str}sigmas-{sigmas_str}v{args.res}_m{args.model}'
    return exp_name


def prepare_and_parse_args():
    # python train.py -posed -dist 0.5 0.5 -std_dev 0.15 0.05 -res 32 -batch_size 40 -m
    parser = argparse.ArgumentParser(
        description='Run Model'
    )

    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-pc_samples', default=3000, type=int)
    parser.add_argument('-dist', '--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
    parser.add_argument('-std_dev', '--sample_sigmas', default=[0.15, 0.015], nargs='+', type=float)
    parser.add_argument('-batch_size', default=30, type=int)
    parser.add_argument('-res', default=32, type=int)
    parser.add_argument('-m', '--model', default='LocNet', type=str)
    parser.add_argument('-o', '--optimizer', default='Adam', type=str)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]

    return args

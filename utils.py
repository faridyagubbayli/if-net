

def form_expr_name(args):
    if args.pointcloud:
        type_str = 'PC' + str(args.pc_samples)
    else:
        type_str = 'Voxels'

    dist_str = ''.join(str(e) + '_' for e in args.sample_distribution)
    sigmas_str = ''.join(str(e) + '_' for e in args.sample_sigmas)

    exp_name = f'i{type_str}_dist-{dist_str}sigmas-{sigmas_str}v{args.res}_m{args.model}'
    return exp_name

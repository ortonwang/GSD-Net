
def data_path(args):
    if args.datasets == 'kvasir':
        path_img = './data/Kvasir/images/'
        path_gt_clean = './data/Kvasir/masks/'
        if args.noise_type == 'clean': path_gt = './data/Kvasir/masks/'
        if args.noise_type == '02': path_gt = './data/Kvasir/masks_200_0.2_0.05_0.2_mask_channel0/'
        if args.noise_type == '08': path_gt = './data/Kvasir/masks_200_0.8_0.05_0.2_mask_channel0/'
        if args.noise_type == 'DE': path_gt = './data/Kvasir/masks_DE/'
        path_img_test = './data/Kvasir/images_test/'
        path_gt_test = './data/Kvasir/masks_test/'
        path_SLIC = './data/Kvasir/images_SLIC/'
    elif args.datasets == 'shenzhen':
        path_img = './data/shenzhen/img/'
        path_gt_clean = './data/shenzhen/mask/'
        if args.noise_type == 'clean': path_gt = './data/shenzhen/mask/'
        if args.noise_type == '02': path_gt = './data/shenzhen/mask_200_0.2_0.05_0.2/'
        if args.noise_type == '08': path_gt = './data/shenzhen/mask_200_0.8_0.05_0.2/'
        if args.noise_type == 'DE': path_gt = './data/shenzhen/mask_DE/'
        path_img_test = './data/shenzhen/test_img/'
        path_gt_test = './data/shenzhen/test_mask/'
        path_SLIC = './data/shenzhen/images_SLIC/'
    elif args.datasets == 'BUSUC':
        path_img = './data/BUSUC/images/'
        path_gt_clean = './data/BUSUC/masks/'
        if args.noise_type == 'clean': path_gt = './data/BUSUC/masks'
        if args.noise_type == '02': path_gt = './data/BUSUC/masks_200_0.2_0.05_0.2/'
        if args.noise_type == '08': path_gt = './data/BUSUC/masks_200_0.8_0.05_0.2/'
        if args.noise_type == 'DE': path_gt = './data/BUSUC/masks_DE/'
        path_img_test = './data/BUSUC/images_test/'
        path_gt_test = './data/BUSUC/masks_test/'
        path_SLIC = './data/BUSUC/images_SLIC/'
    else:
        raise 'dataset configure false'

    return path_img , path_gt_clean, path_gt,path_img_test, path_gt_test,path_SLIC
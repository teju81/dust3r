from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.viz import SceneViz
from dust3r.utils.device import to_numpy
from dust3r.viz import SceneViz, segment_sky, auto_cam_size

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, get_imshapes, signed_expm1, signed_log1p,
                                      cosine_schedule, linear_schedule, get_conf_trf)
import numpy as np
import os

def add_ptc_to_scene(viz, scene, show_pw_cams=False, show_pw_pts3d=False, cam_size=None):
    if scene.imgs is None:
        colors = np.random.randint(0, 256, size=(scene.n_imgs, 3))
        colors = list(map(tuple, colors.tolist()))
        for n in range(scene.n_imgs):
            viz.add_pointcloud(scene.get_pts3d()[n], colors[n], scene.get_masks()[n])
    else:
        viz.add_pointcloud(scene.get_pts3d(), scene.imgs, scene.get_masks())
        colors = np.random.randint(256, size=(scene.n_imgs, 3))

    # camera poses
    im_poses = to_numpy(scene.get_im_poses())
    if cam_size is None:
        cam_size = auto_cam_size(im_poses)
    viz.add_cameras(im_poses, scene.get_focals(), colors=colors,
                    images=scene.imgs, imsizes=scene.imsizes, cam_size=cam_size)
    if show_pw_cams:
        pw_poses = scene.get_pw_poses()
        viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

        if show_pw_pts3d:
            pts = [geotrf(pw_poses[e], scene.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(scene.edges)]
            viz.add_pointcloud(pts, (128, 0, 128))

def main():
    device = 'cuda'
    chunk_size = 5
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    viz = SceneViz()

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    #images = load_images(['croco/assets/Chateau1.png', 'croco/assets/Chateau2.png'], size=512)
    folder_path = '/root/code/datasets/artgarage/coe_dust3r/images'
    img_filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(img_filenames)
    img_list = [img_filenames[i:i + chunk_size] for i in range(0, len(img_filenames), chunk_size)]
    for file_list in img_list:
        print(len(file_list))
        images = load_images(file_list, size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
        #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
        # in each view you have:
        # an integer image identifier: view1['idx'] and view2['idx']
        # the img: view1['img'] and view2['img']
        # the image shape: view1['true_shape'] and view2['true_shape']
        # an instance string output by the dataloader: view1['instance'] and view2['instance']
        # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
        # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
        # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

        # next we'll use the global_aligner to align the predictions
        # depending on your task, you may be fine with the raw output and not need it
        # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
        # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        add_ptc_to_scene(viz, scene)

        # visualize reconstruction
        viz.show()

        # # find 2D-2D matches between the two images
        # from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        # pts2d_list, pts3d_list = [], []
        # for i in range(2):
        #     conf_i = confidence_masks[i].cpu().numpy()
        #     pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        # reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        # print(f'found {num_matches} matches')
        # matches_im1 = pts2d_list[1][reciprocal_in_P2]
        # matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # # visualize a few matches
        # import numpy as np
        # from matplotlib import pyplot as pl
        # n_viz = 10
        # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img = np.concatenate((img0, img1), axis=1)
        # pl.figure()
        # pl.imshow(img)
        # cmap = pl.get_cmap('jet')
        # for i in range(n_viz):
        #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        # pl.show(block=True)

if __name__ == '__main__':
    main()

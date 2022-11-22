#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import open3d as o3d
import numpy as np


def eval(in_file, scene, dataset_dir, eval_dir, suffix):

    in_mesh = o3d.io.read_triangle_mesh(str(in_file))

    sample = int(1e6)
    thresh = 0.8

    stl_pcd_large = o3d.io.read_point_cloud(f"{dataset_dir}/{scene}_dense/gt_full.ply")
    stl_pcd_centered = o3d.io.read_point_cloud(f"{dataset_dir}/{scene}_dense/gt_center.ply")

    in_pcd_large = in_mesh.sample_points_uniformly(sample, seed=0)

    bb_np = np.load(f"{dataset_dir}/{scene}_dense/bbox.npy")
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bb_np))

    idx_pts = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(np.asarray(in_mesh.vertices)))
    mask_pts = np.zeros(len(in_mesh.vertices), dtype=bool)
    mask_pts[idx_pts] = True

    np_faces = np.asarray(in_mesh.triangles)

    valid_triangles = mask_pts[np_faces].all(axis=1)
    in_mesh.triangles = o3d.utility.Vector3iVector(np_faces[valid_triangles])
    in_pcd_centered = in_mesh.sample_points_uniformly(sample, seed=0)

    thresh_color = [0.8, 0.8, 0.05, 0.05]
    from_l = [in_pcd_large, stl_pcd_large, in_pcd_centered, stl_pcd_centered]
    to_l = [stl_pcd_large, in_pcd_large, stl_pcd_centered, in_pcd_centered]
    names = ["pred2stl", "stl2pred", "pred2stl_centered", "stl2pred_centered"]
    result = {}
    for src, dst, name, thresh_c in zip(from_l, to_l, names, thresh_color):
        res = np.asarray(src.compute_point_cloud_distance(dst))
        print(res.min(), res.mean(), res.max(), res[res < thresh].mean())
        result[name] = (res, res[res < thresh].mean())

    with open(f'{eval_dir}/result{suffix}.txt', 'w') as f:
        f.write(f'{result["pred2stl"][1]} {result["stl2pred"][1]} {(result["pred2stl"][1]+result["stl2pred"][1])/2}\n')
        f.write(
            f'{result["pred2stl_centered"][1]} {result["stl2pred_centered"][1]} {(result["pred2stl_centered"][1] + result["stl2pred_centered"][1]) / 2}')
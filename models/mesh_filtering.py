#  NeuralWarp  All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#    * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#    * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

import torch
from tqdm import tqdm
from torch.nn import functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterizer
from pytorch3d.renderer.cameras import PerspectiveCameras



def normalize(flow, h, w, clamp=None):
    # either h and w are simple float or N torch.tensor where N batch size
    try:
        h.device

    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    # for grid_sample with align_corners=True
    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res

def visible_mask(mesh, cams, nb_visible=1):
    # returns a mask of triangles that reprojects on at least nb_visible images
    num_faces = len(mesh.faces)
    count = torch.zeros(num_faces, device="cuda")

    K, R, t, sizes = cams[:4]

    n = len(K)
    with torch.no_grad():
        for i in tqdm(range(n), desc="Rasterization"):
            intr = torch.zeros((1, 4, 4), device="cuda")
            intr[:, :3, :3] = K[i:i + 1]
            intr[:, 3, 3] = 1
            vertices = torch.from_numpy(mesh.vertices).cuda().float()
            faces = torch.from_numpy(mesh.faces).cuda().long()
            meshes = Meshes(verts=[vertices],
                            faces=[faces])

            cam = corrected_cameras_from_opencv_projection(camera_matrix=intr, R=R[i:i + 1].cuda(),
                                                           tvec=t[i:i + 1].squeeze(2).cuda(),
                                                           image_size=sizes[i:i + 1, [1, 0]].cuda())
            cam = cam.cuda()
            raster_settings = rasterizer.RasterizationSettings(image_size=tuple(sizes[i, [1, 0]].long().tolist()),
                                                               faces_per_pixel=1)
            meshRasterizer = rasterizer.MeshRasterizer(cam, raster_settings)

            with torch.no_grad():
                pix_to_face, zbuf, bar, pixd = meshRasterizer(meshes)

            visible_faces = pix_to_face.view(-1).unique()
            count[visible_faces[visible_faces > -1]] += 1


    return (count >= nb_visible).cpu()

# correction from pytorch3d (v0.5.0)
def corrected_cameras_from_opencv_projection( R, tvec, camera_matrix, image_size):
    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Get the PyTorch3D focal length and principal point.
    s = (image_size_wh).min(dim=1).values

    focal_pytorch3d = focal_length / (0.5 * s)
    p0_pytorch3d = -(principal_point - image_size_wh / 2) * 2 / s

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
    )


def visual_hull_mask(full_points, cams, masks, nb_visible=1):
    # K, R: N * 3 * 3 // t: N * 3 * 1; sizes N*2; points n*3; masks: N * h * w
    K, R, t, sizes = cams[:4]
    res = list()

    with torch.no_grad():
        for points in torch.split(full_points, 100000):
            n = points.shape[0]

            proj = (K @ (R @ points.view(n, 1, 3, 1) + t)).squeeze(-1) # n * N * 3
            in_cam_mask = proj[..., 2] > 1e-18
            proj = proj[..., :2] / torch.clamp(proj[..., 2:], 1e-18)

            proj = normalize(proj, sizes[0:1, 1], sizes[0:1, 0])
            grid = torch.clamp(proj.transpose(0, 1).unsqueeze(1), -10, 10)
            warped_masks = F.grid_sample(masks.unsqueeze(1).float(), grid, align_corners=True).squeeze().t() > 0

            # in cam_mask: n * N
            in_cam_mask = in_cam_mask & (proj <= 1).all(dim=-1) & (proj >= -1).all(dim=-1)
            is_not_obj_mask = in_cam_mask & ~warped_masks

            res.append((in_cam_mask.sum(dim=1) >= nb_visible) & ~(is_not_obj_mask.any(dim=1)))

    return torch.cat(res)


def mesh_filter(mesh, masks, cams):
    nb_visible = 2
    filter_visible_triangles = False

    vert_hull_mask = visual_hull_mask(torch.from_numpy(mesh.vertices).float().cuda(), cams, masks,
                                      nb_visible)
    hull_mask = vert_hull_mask[mesh.faces].all(dim=-1).cpu().numpy()

    if filter_visible_triangles:
        pred_visible_mask = visible_mask(mesh, cams, nb_visible).cpu().numpy()
        mesh.update_faces(hull_mask & pred_visible_mask)

    else:
        mesh.update_faces(hull_mask)

    return mesh
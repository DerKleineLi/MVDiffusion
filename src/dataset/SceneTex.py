import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRendererWithFragments,
    RasterizationSettings,
    TexturesUV,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.shader import ShaderBase

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()


def init_camera_R_T(R, T, image_size, device, fov=60):
    """init camera using R and T matrics

    Args:
        R (torch.FloatTensor): Rotation matrix, (N, 3, 3)
        T (torch.FloatTensor): Translation matrix, (N, 3)
        image_size (int): rendering size
        device (torch.device): CPU or GPU

    Returns:
        camera: PyTorch3D camera instance
    """
    image_size = torch.tensor([image_size, image_size]).unsqueeze(0)
    # cameras = PerspectiveCameras(R=R, T=T, device=device, image_size=image_size)
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device, fov=fov)

    return cameras


def init_trajectory(dist_list, elev_list, azim_list, at):
    Rs, Ts = [], []
    for dist, elev, azim in zip(dist_list, elev_list, azim_list):
        R, T = look_at_view_transform(dist, elev, azim, at=at)

        Rs.append(R)  # 1, 3, 3
        Ts.append(T)  # 1, 3

    return Rs, Ts


def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(
        image_size=image_size, faces_per_pixel=faces_per_pixel
    )
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=shader,
    )

    return renderer


class FlatTexelShader(ShaderBase):
    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__(device, cameras, lights, materials, blend_params)

    # override to enable half precision
    def _sample_textures(self, texture_maps, fragments, faces_verts_uvs):
        """
        Interpolate a 2D texture map using uv vertex texture coordinates for each
        face in the mesh. First interpolate the vertex uvs using barycentric coordinates
        for each pixel in the rasterized output. Then interpolate the texture map
        using the uv coordinate for each pixel.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordinates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """

        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape
        N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

        # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

        # textures.map:
        #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
        #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
        texture_maps = (
            texture_maps.permute(0, 3, 1, 2)[None, ...]
            .expand(K, -1, -1, -1, -1)
            .transpose(0, 1)
            .reshape(N * K, C, H_in, W_in)
        )

        # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
        # Now need to format the pixel uvs and the texture map correctly!
        # From pytorch docs, grid_sample takes `grid` and `input`:
        #   grid specifies the sampling pixel locations normalized by
        #   the input spatial dimensions It should have most
        #   values in the range of [-1, 1]. Values x = -1, y = -1
        #   is the left-top pixel of input, and values x = 1, y = 1 is the
        #   right-bottom pixel of input.

        pixel_uvs = pixel_uvs * 2.0 - 1.0

        texture_maps = torch.flip(texture_maps, [2])  # flip y axis of the texture map
        if texture_maps.device != pixel_uvs.device:
            texture_maps = texture_maps.to(pixel_uvs.device)

        pixel_uvs = pixel_uvs.to(texture_maps.dtype)

        texels = F.grid_sample(texture_maps, pixel_uvs)
        # texels now has shape (NK, C, H_out, W_out)
        texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)

        return texels

    def forward(self, fragments, meshes, **_kwargs):
        texels = meshes.sample_textures(fragments)
        texels[(fragments.pix_to_face == -1), :] = 0
        return texels.squeeze(-2)


def init_flat_texel_shader(camera, device):
    shader = FlatTexelShader(cameras=camera, device=device)

    return shader


def init_mesh(mesh_file, device=DEVICE):
    verts, faces, aux = load_obj(mesh_file, device=device)
    mesh = load_objs_as_meshes([mesh_file], device=device)

    # texture = Image.open(TEXTURE)
    # texture = (
    #     torchvision.transforms.ToTensor()(texture)
    #     .permute(1, 2, 0)
    #     .unsqueeze(0)
    #     .to(device)
    # )
    texture = torch.ones(1, 512, 512, 3, dtype=torch.float32, device=device) * 0.8275
    mesh.textures = TexturesUV(
        maps=texture,  # B, H, W, C
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=aux.verts_uvs[None, ...],
        sampling_mode="bilinear",
    )

    return mesh


def init_poses(dist, elev, azim, fov, center):
    dist_linspace = np.linspace(dist[0], dist[1], 1 if dist[0] == dist[1] else dist[2])
    elev_linspace = np.array(elev)
    azim_linspace = np.linspace(azim[0], azim[1], 1 if azim[0] == azim[1] else azim[2])
    fov_linspace = np.linspace(fov[0], fov[1], 1 if fov[0] == fov[1] else fov[2])
    at = np.array([center])

    combinations = np.array(
        np.meshgrid(dist_linspace, elev_linspace, azim_linspace, fov_linspace)
    ).T.reshape(-1, 4)
    dist_list = combinations[:, 0].tolist()
    elev_list = combinations[:, 1].tolist()
    azim_list = combinations[:, 2].tolist()
    fov_list = combinations[:, 3].tolist()

    Rs, Ts = init_trajectory(dist_list, elev_list, azim_list, at)

    return Rs, Ts, at, dist_list, elev_list, azim_list, fov_list


def get_uv_coordinates(mesh, fragments):
    xyzs = mesh.verts_padded()  # (N, V, 3)
    faces = mesh.faces_padded()  # (N, F, 3)

    faces_uvs = mesh.textures.faces_uvs_padded()
    verts_uvs = mesh.textures.verts_uvs_padded()

    # NOTE Meshes are replicated in batch. Taking the first one is enough.
    batch_size, _, _ = xyzs.shape
    xyzs, faces, faces_uvs, verts_uvs = xyzs[0], faces[0], faces_uvs[0], verts_uvs[0]
    faces_coords = verts_uvs[faces_uvs]  # (F, 3, 2)

    # replicate the coordinates as batch
    faces_coords = faces_coords.repeat(batch_size, 1, 1)

    invalid_mask = fragments.pix_to_face == -1
    target_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_coords
    )  # (N, H, W, 1, 3)
    _, H, W, K, _ = target_coords.shape
    target_coords[invalid_mask] = 0
    assert K == 1  # pixel_per_faces should be 1
    target_coords = target_coords.squeeze(3)  # (N, H, W, 2)

    return target_coords


def init_renderer_(camera, image_size, device=DEVICE):
    return init_renderer(
        camera,
        shader=init_flat_texel_shader(camera=camera, device=device),
        image_size=image_size,
        faces_per_pixel=1,
    )


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def intr(image_size, fov):
    fl = fov2focal(fov, image_size)
    intr = np.array(
        [
            [fl, 0, image_size / 2, 0],
            [0, fl, image_size / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return intr


class SceneTexDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="test"):
        super().__init__()
        self.dist = [config["dist"], config["dist"], 1]  # [2.5, 2.5, 1]
        self.center = config["center"]  # [x,y,z]
        self.elev = [30, 45, -15]  # list
        self.azim = [-180, 135, 8]  # linspace
        self.fov = [60, 60, 1]  # linspace
        self.prompt = config["prompt"]  # "a classic style living room"

        self.mesh_file = config["mesh_file"]  # scene.obj
        self.image_size = config["image_size"]  # 512

        mesh_dir = os.path.dirname(self.mesh_file)
        fake_image = np.zeros((self.image_size, self.image_size, 3))
        self.fake_image_path = os.path.join(mesh_dir, "fake.jpg")
        cv2.imwrite(self.fake_image_path, fake_image)

        self.mesh = init_mesh(self.mesh_file)
        (
            self.Rs,
            self.Ts,
            self.at,
            self.dist_list,
            self.elev_list,
            self.azim_list,
            self.fov_list,
        ) = init_poses(self.dist, self.elev, self.azim, self.fov, self.center)

    def __len__(self):
        return 1

    def _get_consecutive_kfs_inp(self):
        img_paths = [self.fake_image_path] * len(self.Rs)
        mask = np.zeros(len(img_paths)).astype(np.bool)
        mask[0] = True
        mask[-1] = True
        return img_paths, mask

    def load_seq(self, image_paths):
        images_ori = []
        poses = []
        prompts = []
        depths = []
        uvs = []
        for view_id, (R, T, dist, elev, azim, fov) in enumerate(
            zip(
                self.Rs,
                self.Ts,
                self.dist_list,
                self.elev_list,
                self.azim_list,
                self.fov_list,
            )
        ):
            camera = init_camera_R_T(R, T, self.image_size, DEVICE, fov)
            renderer = init_renderer_(camera, self.image_size, DEVICE)

            rendering, fragments = renderer(self.mesh)
            depth = fragments.zbuf
            uv_coords = get_uv_coordinates(self.mesh, fragments)
            w2c = camera.get_world_to_view_transform().get_matrix().squeeze().T  # 4x4
            w2c[[0, 1]] *= -1
            c2w = torch.inverse(w2c)

            images_ori.append(rendering.squeeze().detach().cpu().numpy())
            poses.append(c2w.detach().cpu().numpy())
            prompts.append(self.prompt)
            depths.append(depth.squeeze().detach().cpu().numpy())
            uvs.append(uv_coords.squeeze().detach().cpu().numpy())

        poses = np.stack(poses, axis=0)  # [num_views, 4, 4]
        k = intr(self.image_size, fov / 180 * np.pi)  # [4, 4]
        images = np.stack(images_ori) * 2.0 - 1.0
        depths = np.stack(depths, axis=0)
        uvs = np.stack(uvs, axis=0)

        depth_valid_mask = depths > 0
        depth_inv = 1.0 / (depths + 1e-6)
        depth_max = [
            depth_inv[i][depth_valid_mask[i]].max() for i in range(depth_inv.shape[0])
        ]
        depth_min = [
            depth_inv[i][depth_valid_mask[i]].min() for i in range(depth_inv.shape[0])
        ]
        depth_max = np.stack(depth_max, axis=0)[:, None, None]
        depth_min = np.stack(depth_min, axis=0)[:, None, None]  # [num_views, 1, 1]
        depth_inv_norm_full = (depth_inv - depth_min) / (
            depth_max - depth_min + 1e-6
        ) * 2 - 1  # [-1, 1]
        depth_inv_norm_full[~depth_valid_mask] = -2
        depth_inv_norm_full = depth_inv_norm_full.astype(np.float32)
        return images, depths, depth_inv_norm_full, poses, k, prompts, uvs

    def __getitem__(self, idx):
        image_paths, mask = self._get_consecutive_kfs_inp()

        images, depths, depth_inv_norm, poses, K, prompts, uvs = self.load_seq(
            image_paths
        )

        depth_inv_norm_small = np.stack(
            [
                cv2.resize(
                    depth_inv_norm[i],
                    (self.image_size // 8, self.image_size // 8),
                    interpolation=cv2.INTER_NEAREST,
                )
                for i in range(depth_inv_norm.shape[0])
            ]
        )

        images = images.astype(np.float32)
        depths = depths.astype(np.float32)
        poses = poses.astype(np.float32)
        K = K.astype(np.float32)

        return {
            "image_paths": image_paths,
            "mask": mask,
            "images": images,
            "depths": depths,
            "poses": poses,
            "K": K,
            "prompt": prompts,
            "depth_inv_norm": depth_inv_norm,
            "depth_inv_norm_small": depth_inv_norm_small,
            "data_load_mode": "two_stage",
            "uvs": uvs,
        }

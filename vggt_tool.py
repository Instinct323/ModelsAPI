import json
from pathlib import Path

import cv2
import numpy as np
import torch
from vggt.models.vggt import VGGT as ModelType
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.visual_track import visualize_tracks_on_images

from utils.utils import huggingface_model_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class VGGT:

    def __init__(self,
                 repo_id: str = "facebook/VGGT-1B",
                 enable_camera: bool = True,
                 enable_depth: bool = True,
                 enable_point: bool = False):
        self.model = ModelType.from_pretrained(huggingface_model_path(repo_id)).to(DEVICE).eval()
        if not enable_camera: self.model.camera_head = None
        if not enable_depth: self.model.depth_head = None
        if not enable_point: self.model.point_head = None

    def __call__(self,
                 images: torch.Tensor,
                 query_points: torch.Tensor = None,
                 extract_extri_intri: bool = True,
                 unproject_depth_map: bool = True):
        """ Inference one sequence of images with query points."""
        output = self.model(images, query_points)
        if extract_extri_intri or unproject_depth_map:
            output["extrinsic"], output["intrinsic"] = pose_encoding_to_extri_intri(output["pose_enc"], output["images"].shape[-2:])
        # Construct 3D Points from Depth Maps and Cameras
        if unproject_depth_map:
            assert "world_points" not in output, "please disable point_head to avoid conflict"
            assert len(images.shape) == 4
            output["world_points"] = unproject_depth_map_to_point_map(
                output["depth"].squeeze(0), output["extrinsic"].squeeze(0), output["intrinsic"].squeeze(0))[None].astype(np.float32)
            output["world_points_conf"] = output["depth_conf"]
        # Correct the format
        output["images"] = output["images"].permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        output["depth"] = output["depth"].squeeze(-1)
        for k in output:
            output[k] = output[k].squeeze(0)
            if isinstance(output[k], torch.Tensor):
                output[k] = output[k].cpu().data.to(torch.float32).numpy()
        return output

    def export_sequence(self,
                        output: dict,
                        depth_conf_thresh: float,
                        depth_scale: float = 1000.0,
                        project_dir: str = "runs/vggt-seq"):
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        # color
        color_dir = project_dir / "color"
        color_dir.mkdir(parents=True, exist_ok=True)
        color_img = np.round(output["images"][..., ::-1] * 255).astype(np.uint8)
        # depth
        depth_dir = project_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_img = np.round(output["depth"] * depth_scale).astype(np.uint16)
        depth_img[output["depth_conf"] < depth_conf_thresh] = 0
        # save images
        for i, (color, depth) in enumerate(zip(color_img, depth_img)):
            fname = str(i).zfill(6) + ".png"
            cv2.imwrite(str(color_dir / fname), color)
            cv2.imwrite(str(depth_dir / fname), depth)
        # intrinsic
        intrinsic = output["intrinsic"].mean(axis=0).flatten().tolist()
        intrinsic = dict(fx=intrinsic[0], fy=intrinsic[4], cx=intrinsic[2], cy=intrinsic[5],
                         width=output["images"].shape[2], height=output["images"].shape[1],
                         depth_scale=depth_scale, depth_conf_thresh=depth_conf_thresh)
        with (project_dir / "intrinsic.json").open("w") as f:
            json.dump(intrinsic, f, indent=4)
        # extrinsic
        pose_fmt = ("%.8f " * 12 + "%d " * 4)[:-1] + "\n"
        w2c = np.tile(np.eye(4, dtype=np.float32), (output["extrinsic"].shape[0], 1, 1))
        w2c[:, :3] = output["extrinsic"]
        c2w = closed_form_inverse_se3(w2c)
        with (project_dir / f"T_world_cam.txt").open("w") as f:
            for pose in w2c: f.write(pose_fmt % tuple(pose.flatten()))
        with (project_dir / f"T_cam_world.txt").open("w") as f:
            for pose in c2w: f.write(pose_fmt % tuple(pose.flatten()))

    def visualize_tracks(self,
                         output: dict,
                         conf_thresh: float,
                         vis_thresh: float,
                         project_dir: str = "runs/vggt-track"):
        mask = (output["conf"] > conf_thresh) & (output["vis"] > vis_thresh)
        visualize_tracks_on_images(*map(torch.from_numpy, (output["images"].transpose(0, 3, 1, 2),
                                                           output["track"], mask)), out_dir=project_dir)


VGGT.__call__.__doc__ = ModelType.forward.__doc__

if __name__ == '__main__':
    import open3d as o3d
    from utils.zjcv import to_colorful_pcd


    def extract_pcd(output: dict,
                    depth_conf_thresh: float,
                    voxel_size: float):
        return to_colorful_pcd(
            output["world_points"], output["images"][..., ::-1] * 255, output["world_points_conf"] > depth_conf_thresh
        ).voxel_down_sample(voxel_size)


    depth_conf_thresh = 2.5
    vggt_ = VGGT()

    colors_name = list(Path("/media/tongzj/Data/Information/Data/dataset/concept-graph/scene01/rgb").iterdir())[:50]
    colors = load_and_preprocess_images(colors_name).to(DEVICE)
    query_points = torch.FloatTensor([[100.0, 200.0], [60.72, 259.94]]).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(DEVICE, dtype=DTYPE):
            output = vggt_(colors)
            for k, v in output.items(): print(f"{k}: {v.shape}")

            # vggt_.visualize_tracks(out, 0.2, 0.2)
            vggt_.export_sequence(output, depth_conf_thresh)

            pcd_o3d = extract_pcd(output, depth_conf_thresh, 0.002)
            o3d.visualization.draw_geometries([pcd_o3d])

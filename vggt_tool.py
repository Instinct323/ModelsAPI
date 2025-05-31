import torch
from vggt.models.vggt import VGGT as ModelType
from vggt.utils.geometry import unproject_depth_map_to_point_map
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
        output = self.model(images, query_points)
        if extract_extri_intri or unproject_depth_map:
            output["extrinsic"], output["intrinsic"] = pose_encoding_to_extri_intri(output["pose_enc"], output["images"].shape[-2:])
        # Construct 3D Points from Depth Maps and Cameras
        if unproject_depth_map:
            assert "world_points" not in output, "please disable point_head to avoid conflict"
            assert len(images.shape) == 4
            output["world_points"] = unproject_depth_map_to_point_map(
                output["depth"].squeeze(0), output["extrinsic"].squeeze(0), output["intrinsic"].squeeze(0))[None]
            output["world_points_conf"] = output["depth_conf"]
        return output

    def visualize_tracks(self,
                         output: dict,
                         conf_thresh: float,
                         vis_thresh: float,
                         out_dir: str = "runs/vggt-track"):
        mask = None if conf_thresh <= 0 and vis_thresh <= 0 else \
            (output["conf"] > conf_thresh) & (output["vis"] > vis_thresh)
        visualize_tracks_on_images(output["images"], output["track"], mask, out_dir=out_dir)


VGGT.__call__.__doc__ = ModelType.forward.__doc__

if __name__ == '__main__':
    from pathlib import Path

    vggt_ = VGGT()

    image_names = list(Path("assets/kitchen").iterdir())
    images = load_and_preprocess_images(image_names).to(DEVICE)

    query_points = torch.FloatTensor([[100.0, 200.0],
                                      [60.72, 259.94]]).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(DEVICE, dtype=DTYPE):

            out = vggt_(images, query_points)
            for k, v in out.items(): print(f"{k}: {v.shape}")

            vggt_.visualize_tracks(out, 0.2, 0.2)

import cv2
import numpy as np
import pyrealsense2 as rs

DEVICES = rs.context().query_devices()


def frame2numpy(frame: rs.frame) -> np.ndarray:
    return np.asanyarray(frame.get_data())


def get_intrinsics(frame: rs.frame):
    return frame.profile.as_video_stream_profile().intrinsics


def rgbd_flow(w, h, fps=30, device_id=0):
    cfg = rs.config()
    cfg.enable_device(DEVICES[device_id].get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

    pipe = rs.pipeline()
    pipe.start(cfg)
    align = rs.align(rs.stream.color)

    while True:
        frames = align.process(pipe.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame: continue
        color_image = cv2.cvtColor(frame2numpy(color_frame), cv2.COLOR_RGB2BGR)
        depth_image = frame2numpy(depth_frame)

        yield color_image, depth_image


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    root = "../runs/"

    for color, depth in rgbd_flow(640, 480, device_id=0):
        to_show = np.vstack((color, cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET)))
        cv2.imwrite(root + "c.png", color)
        cv2.imwrite(root + "d.png", depth)
        plt.imshow(to_show[..., ::-1])
        plt.pause(1e-3)


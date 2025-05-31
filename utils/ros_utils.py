from pathlib import Path

import cv2
import rosbag
from cv_bridge import CvBridge


def rosbag_to_images(bag_file: Path,
                     topic_name: str,
                     output_dir: Path,
                     stride: int,
                     encoding: str):
    """ Convert images from a ROS bag file to a directory of images. """
    bridge = CvBridge()
    cnt = 0

    output_dir.mkdir(exist_ok=True, parents=True)
    with rosbag.Bag(bag_file, 'r') as bag:
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[topic_name])):

            if i % stride == 0:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
                cv2.imwrite(output_dir / f"{str(cnt).zfill(6)}.png", cv_image)
                cnt += 1


if __name__ == '__main__':
    ROOT = Path("/media/tongzj/Data/Information/Data/dataset/scene-test")

    rosbag_to_images(ROOT / "scene-test.bag",
                     "/camera/color/image_raw",
                     ROOT / "rgb", 20, "rgb8")
    rosbag_to_images(ROOT / "scene-test.bag",
                     "/camera/aligned_depth_to_color/image_raw",
                     ROOT / "depth", 20, "16UC1")

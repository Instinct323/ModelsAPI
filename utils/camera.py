import cv2
import numpy as np
import yaml


class Camera:

    def __init__(self,
                 img_size: tuple[int, int],
                 intrinsics: list[float],
                 dist_coeffs: list[float] = None):
        self.img_size = img_size
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        assert len(intrinsics) == 4
        self._pixels = np.stack(np.meshgrid(*map(np.arange, self.img_size)), axis=-1)
        self._unproj: np.ndarray = None

    @classmethod
    def from_yaml(cls, cfg) -> 'Camera':
        with open(cfg, 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.Loader)
        return eval("_" + cfg.pop("type"))(**cfg)

    def camera_matrix(self):
        """Camera matrix"""
        fx, fy, cx, cy = self.intrinsics
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def project(self, pcd: np.ndarray):
        """ Project 3D points to 2D image plane """
        raise NotImplementedError

    def unproject(self, pixels: np.ndarray):
        """ Unproject 2D image plane to 3D points """
        pixels = pixels.astype(np.float32)
        return cv2.remap(self._unproj[..., :2], pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)

    def unproject_depth(self, depth: np.ndarray = None):
        """ Unproject 2D image plane to 3D points """
        return self._unproj.copy() if depth is None else self._unproj * depth[..., None]

    def undistort(self, img: np.ndarray):
        """ Undistort image """
        return img.copy()

    def normalized_plane(self, img: np.ndarray):
        w, h = map((-1).__add__, self.img_size)
        xyxy = self.unproject(np.array([[0, h / 2], [w / 2, 0], [w, h / 2], [w / 2, h]])
                              )[:, :2].flatten()[[0, 3, 4, 7]]
        pcd = self._pixels / self.img_size * (xyxy[2:] - xyxy[:2]) + xyxy[:2]
        pcd = np.concatenate([pcd, np.ones_like(pcd[..., :1])], axis=-1)
        pixels = self.project(pcd).astype(np.float32)
        return cv2.remap(img, pixels[..., 0], pixels[..., 1], interpolation=cv2.INTER_LINEAR)


class _Pinhole(Camera):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dist_coeffs:
            K = self.camera_matrix()
            self._undist_map = cv2.initUndistortRectifyMap(K, np.array(self.dist_coeffs), None, K, self.img_size, cv2.CV_32FC1)
        # cache for unprojection
        self._unproj = np.concatenate([(self._pixels - self.intrinsics[2:]) / self.intrinsics[:2],
                                       np.ones_like(self._pixels[..., :1])], axis=-1)

    def undistort(self, img: np.ndarray):
        if not self.dist_coeffs: return img
        return cv2.remap(img, *self._undist_map, interpolation=cv2.INTER_LINEAR)

    def project(self, pcd: np.ndarray):
        return pcd[..., :2] / pcd[..., 2:] * self.intrinsics[:2] + self.intrinsics[2:]


class _KannalaBrandt(Camera):
    MAX_FOV = 180

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert len(self.dist_coeffs) == 4
        # cache for unprojection
        coords = (self._pixels - self.intrinsics[2:]) / self.intrinsics[:2]
        coords /= self._solveWZ(coords)[..., None]
        self._unproj = np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1)

    def _computeR(self, theta: np.ndarray):
        theta2 = np.square(theta)
        return theta + theta2 * (
                self.dist_coeffs[0] + theta2 * (
                self.dist_coeffs[1] + theta2 * (
                self.dist_coeffs[2] + theta2 * self.dist_coeffs[3])))

    def _solveWZ(self,
                 coords: np.ndarray,
                 niter: int = 20):
        R = np.linalg.norm(coords, axis=-1)
        # wz = lim_{theta -> 0} R / tan(theta) = 1
        wz = np.ones_like(R)
        theta = np.deg2rad(self.MAX_FOV)
        is_valid = (R > 1e-6) & (R < self._computeR(theta))
        R = R[is_valid]
        theta = np.full_like(R, theta)
        for i in range(niter):
            # Minimize: (poly(theta) - R)^2
            thetai = [np.square(theta)]
            for j in range(3): thetai.append(thetai[-1] * thetai[0])
            thetai = np.stack(thetai, axis=-1)
            ki_thetai = thetai * self.dist_coeffs
            e = theta * (1 + ki_thetai.sum(axis=-1)) - R
            # grad = (poly(theta) - R) / poly'(theta)
            theta -= e / (1 + ki_thetai @ np.arange(3, 10, 2))
        wz[is_valid] = R / np.tan(theta)
        return wz

    def project(self, pcd: np.ndarray):
        R = self._computeR(np.arctan2(np.linalg.norm(pcd[..., :2], axis=-1, keepdims=True), pcd[..., 2:3]))
        psi = np.arctan2(pcd[..., 1], pcd[..., 0])
        polars = np.stack([np.cos(psi), np.sin(psi)], axis=-1)
        return polars * R * self.intrinsics[:2] + self.intrinsics[2:]


if __name__ == '__main__':
    cam = Camera.from_yaml("../cfg/RS-D405.yaml")
    print(cam)
    img = cv2.imread("../assets/color.png")
    normal_plane = cam.normalized_plane(img)

    # plt.imshow(normal_plane), plt.show()

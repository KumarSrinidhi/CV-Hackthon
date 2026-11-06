"""Official 2005/2006 third-size Middlebury disparity-SR adapter.
Not metric depth: stored disparity/3 = pixels at third-size resolution.
Zero in the original PNG is invalid, preserved before filling or degradation.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from data.dataset import DepthSRDataset
from data.preprocessing import to_float_rgb


class MiddleburyRawDataset(DepthSRDataset):
    name = 'middlebury2006_third_disparity'
    unit = 'disparity_px'
    data_range = 255.0 / 3.0
    split = 'test'

    def __init__(self, root='datasets/Middlebury_raw', scale=4, views=(1,), degradation=None):
        self.root = Path(root)
        self.pairs = []
        scenes = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not scenes:
            raise FileNotFoundError(f'No scene folders under {self.root}')
        for scene in scenes:
            for view in views:
                color = scene / f'view{view}.png'
                disparity = scene / f'disp{view}.png'
                if not color.is_file() or not disparity.is_file():
                    raise FileNotFoundError(f'Missing view{view}.png or disp{view}.png in {scene}')
                self.pairs.append((scene.name, view, color, disparity))
        super().__init__(scale, degradation)

    def _num_raw(self):
        return len(self.pairs)

    def _load_raw(self, k):
        scene, view, color_path, disparity_path = self.pairs[k]
        with Image.open(color_path) as im:
            rgb = to_float_rgb(np.asarray(im.convert('RGB')))
        with Image.open(disparity_path) as im:
            encoded = np.asarray(im)
        if encoded.ndim != 2 or encoded.dtype != np.uint8:
            raise ValueError('Expected original 8-bit third-size disparity PNG.')
        if rgb.shape[:2] != encoded.shape:
            raise ValueError(f'RGB/disparity geometry mismatch: {scene}, view {view}')
        valid = encoded > 0
        disparity = encoded.astype(np.float32) / 3.0
        return f'{scene}_view{view}', rgb, disparity, valid

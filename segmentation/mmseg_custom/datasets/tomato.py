# /root/workspace/InternImage/segmentation/mmseg_custom/datasets/tomato.py

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from pathlib import Path
import os
import os.path as osp
import mmcv


@DATASETS.register_module()
class TomatoDataset(CustomDataset):
    
    CLASSES = (
        "background",
        "Core",
        "Locule",
        "Navel",
        "Pericarp",
        "Placenta",
        "Septum",
        "Tomato"
    )

    PALETTE = [
        [0, 0, 0],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0]
    ]

    METAINFO = {
        "classes": (
            "background",
            "Core",
            "Locule",
            "Navel",
            "Pericarp",
            "Placenta",
            "Septum",
            "Tomato"
        ),
        "palette": [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0]
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                        split):
        """Load annotation from directory.
        
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images (ignored, we scan all extensions)
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str): Suffix of segmentation maps.
            split (str|None): Split txt file.

        Returns:
            list[dict]: All data info of dataset.
        """
        img_infos = []
        
        if split is not None:
            # Load from split file
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            # Scan directory for all image files
            valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
            
            for img_file in mmcv.scandir(img_dir, recursive=False):
                # Check if file has valid image extension
                if not any(img_file.endswith(ext) for ext in valid_extensions):
                    continue
                
                img_info = dict(filename=img_file)
                
                if ann_dir is not None:
                    # Get basename without extension
                    img_name = osp.splitext(img_file)[0]
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                
                img_infos.append(img_info)
        
        print(f'Loaded {len(img_infos)} images from {img_dir}')
        return img_infos



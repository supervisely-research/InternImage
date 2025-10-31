# /root/workspace/InternImage/segmentation/mmseg_custom/datasets/tomato.py

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


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
    
    def load_data_list(self):
        data_list = []
        for img_path in self.img_dir.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg"):
                seg_path = self.ann_dir / f"{img_path.stem}.png"
                if seg_path.exists():
                    data_list.append(dict(img_path=str(img_path), seg_map_path=str(seg_path)))
        return data_list

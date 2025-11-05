import os.path as osp
import mmcv
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class AdaptiveTomatoDataset(CustomDataset):
    """Adaptive Tomato Dataset with dynamic sample addition.
    
    This dataset starts with a small number of samples and can dynamically
    add more samples during training via the add_next_samples() method.
    
    Args:
        max_samples (int): Maximum number of samples to use. Default: 5.
        initial_samples (int): Number of samples to start with. Default: 3.
        samples_per_stage (int): Number of samples to add per stage. Default: 1.
        **kwargs: Additional arguments passed to CustomDataset.
    """
    
    CLASSES = (
        "background", "Core", "Locule", "Navel",
        "Pericarp", "Placenta", "Septum", "Tomato"
    )

    PALETTE = [
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 128, 0]
    ]

    METAINFO = {
        "classes": CLASSES,
        "palette": PALETTE
    }

    def __init__(self, 
                 max_samples=5,
                 initial_samples=3,
                 samples_per_stage=1,
                 **kwargs):
        self.max_samples = max_samples
        self.initial_samples = initial_samples
        self.samples_per_stage = samples_per_stage
        self.current_sample_count = initial_samples
        self.full_img_infos = None
        
        super().__init__(
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
        
        print(f"Initialized AdaptiveTomatoDataset:")
        print(f"  - Max samples: {self.max_samples}")
        print(f"  - Initial samples: {self.initial_samples}")
        print(f"  - Current samples: {self.current_sample_count}")
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotations and store full dataset info."""
        all_img_infos = []
        
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    all_img_infos.append(img_info)
        else:
            valid_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
            
            for img_file in mmcv.scandir(img_dir, recursive=False):
                if not any(img_file.endswith(ext) for ext in valid_extensions):
                    continue
                
                img_info = dict(filename=img_file)
                
                if ann_dir is not None:
                    img_name = osp.splitext(img_file)[0]
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                
                all_img_infos.append(img_info)
        
        self.full_img_infos = all_img_infos
        current_img_infos = all_img_infos[:self.current_sample_count]
        
        print(f'Loaded {len(current_img_infos)}/{len(all_img_infos)} images from {img_dir}')
        return current_img_infos
    
    def add_next_samples(self):
        """Add next batch of samples to the dataset.
        
        Returns:
            bool: True if samples were added, False otherwise.
        """
        if self.current_sample_count >= self.max_samples:
            return False
        
        if self.full_img_infos is None:
            print("Warning: full_img_infos not loaded yet")
            return False
        
        available_samples = len(self.full_img_infos)
        new_count = min(
            self.current_sample_count + self.samples_per_stage,
            self.max_samples,
            available_samples
        )
        
        if new_count == self.current_sample_count:
            return False
        
        old_count = self.current_sample_count
        self.current_sample_count = new_count
        self.img_infos = self.full_img_infos[:self.current_sample_count]
        
        print(f"Added samples: {old_count} -> {self.current_sample_count}/{self.max_samples}")
        return True
    
    def get_current_sample_count(self):
        """Get current number of samples in use.
        
        Returns:
            int: Current sample count.
        """
        return self.current_sample_count
    
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        if idx >= len(self.img_infos):
            idx = idx % len(self.img_infos)
        return super().__getitem__(idx)
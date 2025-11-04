import os
import os.path as osp
from pathlib import Path
import mmcv
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv.runner import HOOKS, Hook


@DATASETS.register_module()
class AdaptiveTomatoDataset(CustomDataset):
    """
    Adaptive Tomato Dataset that can dynamically add samples during training.
    
    Inherits from CustomDataset and adds functionality to incrementally
    add training samples based on adaptive learning strategy.
    """
    
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

    def __init__(self, 
                 max_samples=5,
                 initial_samples=2,
                 samples_per_stage=1,
                 **kwargs):
        """
        Args:
            max_samples (int): Maximum number of samples to use from dataset
            initial_samples (int): Number of samples to start with
            samples_per_stage (int): Number of samples to add at each stage
            **kwargs: Arguments passed to parent CustomDataset
        """
        self.max_samples = max_samples
        self.initial_samples = initial_samples
        self.samples_per_stage = samples_per_stage
        self.current_sample_count = initial_samples
        
        # Store full dataset info
        self.full_img_infos = None
        
        # Initialize parent
        super().__init__(
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs
        )
        
        print(f"Initialized AdaptiveTomatoDataset:")
        print(f"  - Max samples: {self.max_samples}")
        print(f"  - Initial samples: {self.initial_samples}")
        print(f"  - Samples per stage: {self.samples_per_stage}")
        print(f"  - Current samples: {self.current_sample_count}")
    
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                        split):
        """Load annotation from directory and store full list.
        
        This method loads ALL available images first, then filters
        to current_sample_count for adaptive learning.
        """
        # Load all available images first
        all_img_infos = []
        
        if split is not None:
            # Load from split file
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    all_img_infos.append(img_info)
        else:
            # Scan directory for all image files
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
        
        # Store full list for later use
        self.full_img_infos = all_img_infos
        
        # Return only current_sample_count samples
        current_img_infos = all_img_infos[:self.current_sample_count]
        
        print(f'Loaded {len(current_img_infos)}/{len(all_img_infos)} images from {img_dir}')
        
        return current_img_infos
    
    def add_next_samples(self):
        """Add next batch of samples and reload dataset.
        
        Returns:
            bool: True if samples were added, False if already at max
        """
        if self.current_sample_count >= self.max_samples:
            # print(f"Already at maximum samples: {self.max_samples}")
            return False
        
        if self.full_img_infos is None:
            print("Warning: full_img_infos not loaded yet")
            return False
        
        # Calculate new count (don't exceed max_samples or available samples)
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
        
        # Update img_infos with new sample count
        self.img_infos = self.full_img_infos[:self.current_sample_count]
        
        print(f"Added samples: {old_count} -> {self.current_sample_count}/{self.max_samples}")
        
        return True
    
    def get_current_sample_count(self):
        """Return current number of samples in dataset."""
        return self.current_sample_count
    
    def __len__(self):
        """Return current dataset length.
        
        This is important for DataLoader to know the correct size
        after adding new samples.
        """
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        """Get item with safety check for adaptive learning.
        
        During adaptive learning, DataLoader might request indices
        that are out of bounds if iterator wasn't reset properly.
        This provides a safety net.
        """
        # Safety check: if index out of bounds, wrap it
        if idx >= len(self.img_infos):
            idx = idx % len(self.img_infos)
        
        return super().__getitem__(idx)


@HOOKS.register_module()
class AdaptiveLearningHook(Hook):
    """
    Hook to manage adaptive learning process with iteration-based schedule.
    
    Adds new samples every N iterations, saves checkpoints, and logs metrics.
    Works with IterBasedRunner (not EpochBasedRunner).
    """
    
    def __init__(self, 
                 iters_per_stage=400,
                 save_checkpoint=True):
        """
        Args:
            iters_per_stage (int): Number of iterations between adding samples
            save_checkpoint (bool): Whether to save checkpoint when adding samples
        """
        self.iters_per_stage = iters_per_stage
        self.save_checkpoint = save_checkpoint
        self.stage_count = 0
        self.last_add_iter = 0
    
    def before_train_iter(self, runner):
        """Check if we need to add new samples at the start of iteration.
        
        This is called before each training iteration.
        """
        current_iter = runner.iter
        
        # Check if it's time to add new samples
        # (every iters_per_stage iterations, but not at iter 0)
        if current_iter > 0 and (current_iter - self.last_add_iter) >= self.iters_per_stage:
            # Get dataset from IterLoader (MMSeg uses wrapped dataloader)
            if hasattr(runner.data_loader, '_dataloader'):
                dataset = runner.data_loader._dataloader.dataset
            else:
                dataset = runner.data_loader.dataset
            
            if hasattr(dataset, 'add_next_samples'):
                old_count = dataset.get_current_sample_count()
                success = dataset.add_next_samples()
                
                if success:
                    self.stage_count += 1
                    self.last_add_iter = current_iter
                    new_count = dataset.get_current_sample_count()
                    
                    # Reset iterator to force DataLoader to recreate it with new dataset size
                    if hasattr(runner.data_loader, '_iterator'):
                        runner.data_loader._iterator = None
                        runner.logger.info("[Adaptive Learning] Reset data_loader iterator")
                    
                    # Log stage change
                    runner.logger.info(
                        f"[Adaptive Learning] Stage {self.stage_count} at iter {current_iter}: "
                        f"samples {old_count} -> {new_count}"
                    )
                    
                    # Log to tensorboard if available
                    if hasattr(runner, 'log_buffer'):
                        runner.log_buffer.output['adaptive/sample_count'] = new_count
                        runner.log_buffer.output['adaptive/stage'] = self.stage_count
                    
                    # Save checkpoint with meaningful name
                    if self.save_checkpoint:
                        self._save_adaptive_checkpoint(runner, new_count)
    
    def _save_adaptive_checkpoint(self, runner, sample_count):
        """Save checkpoint with adaptive learning info in filename."""
        checkpoint_name = f"iter_{runner.iter}_samples_{sample_count}.pth"
        checkpoint_path = osp.join(runner.work_dir, checkpoint_name)
        
        runner.save_checkpoint(
            runner.work_dir,
            filename_tmpl=checkpoint_name,
            save_optimizer=True,
            meta=dict(
                iter=runner.iter,
                sample_count=sample_count,
                stage=self.stage_count
            )
        )
        
        runner.logger.info(f"[Adaptive Learning] Saved checkpoint: {checkpoint_name}")
    
    def after_train_iter(self, runner):
        """Log additional metrics after each iteration."""
        # Log current sample count periodically (every 50 iters)
        if runner.iter % 50 == 0:
            # Get dataset from IterLoader (MMSeg uses wrapped dataloader)
            if hasattr(runner.data_loader, '_dataloader'):
                dataset = runner.data_loader._dataloader.dataset
            else:
                dataset = runner.data_loader.dataset
                
            if hasattr(dataset, 'get_current_sample_count'):
                current_samples = dataset.get_current_sample_count()
                
                if hasattr(runner, 'log_buffer'):
                    runner.log_buffer.output['adaptive/current_samples'] = current_samples
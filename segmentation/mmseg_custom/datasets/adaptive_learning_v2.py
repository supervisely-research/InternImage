import os
import os.path as osp
from pathlib import Path
import numpy as np
import mmcv
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv.runner import HOOKS, Hook


@DATASETS.register_module()
class AdaptiveTomatoDataset(CustomDataset):
    """
    Adaptive Tomato Dataset with dynamic sample addition.
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
        return self.current_sample_count
    
    def __len__(self):
        return len(self.img_infos)
    
    def __getitem__(self, idx):
        if idx >= len(self.img_infos):
            idx = idx % len(self.img_infos)
        return super().__getitem__(idx)


@HOOKS.register_module()
class AdaptiveLearningWithLossDecayHook(Hook):
    """
    Adaptive learning hook with train loss decay monitoring.
    
    Dynamically adjusts early stopping parameters based on sample count:
    - Few samples (2-3) -> strict monitoring (quick reaction)
    - Medium samples (4) -> moderate tolerance
    - Many samples (5+) -> high tolerance
    
    Args:
        iters_per_stage (int): Regular interval for adding samples. Default: 150
        save_checkpoint (bool): Whether to save checkpoint when adding samples. Default: True
        window_size (int): Number of iterations for decay calculation. Default: 40
        decay_threshold (float): Minimum decay rate to continue. Default: 0.001
        patience (int): How many slow decay checks before action. Default: 10
        lr_warmup_after_add (bool): Increase LR after adding samples. Default: True
        lr_warmup_factor (float): LR multiplier for warmup. Default: 2.0
        lr_warmup_iters (int): Warmup duration. Default: 50
    """
    
    def __init__(self, 
                 iters_per_stage=150,
                 save_checkpoint=True,
                 window_size=40,
                 decay_threshold=0.001,
                 patience=10,
                 lr_warmup_after_add=True,
                 lr_warmup_factor=2.0,
                 lr_warmup_iters=50):
        
        self.iters_per_stage = iters_per_stage
        self.save_checkpoint = save_checkpoint
        self.lr_warmup_after_add = lr_warmup_after_add
        self.lr_warmup_factor = lr_warmup_factor
        self.lr_warmup_iters = lr_warmup_iters
        
        # Store base parameters for adaptive scaling
        self.base_window_size = window_size
        self.base_decay_threshold = decay_threshold
        self.base_patience = patience
        
        # Current adaptive parameters (will be set on first iter)
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.patience = patience
        
        # Training state
        self.stage_count = 0
        self.last_add_iter = 0
        self.loss_history = []
        self.patience_counter = 0
        self.original_lr = None
        self.warmup_start_iter = None

    def _get_dataset(self, runner):
        """Helper to get dataset from runner."""
        if hasattr(runner.data_loader, '_dataloader'):
            return runner.data_loader._dataloader.dataset
        return runner.data_loader.dataset
    
    def _get_adaptive_parameters(self, n_samples):
        """
        Calculate adaptive parameters based on sample count.
        
        Returns:
            dict: {'patience': int, 'decay_threshold': float, 'window_size': int}
        """
        if n_samples <= 2:
            # Very strict - quick reaction to overfitting
            return {
                'patience': self.base_patience // 3,
                'decay_threshold': self.base_decay_threshold * 10.0,
                'window_size': self.base_window_size
            }
        elif n_samples <= 3:
            # Moderate
            return {
                'patience': self.base_patience,
                'decay_threshold': self.base_decay_threshold,
                'window_size': self.base_window_size
            }
        elif n_samples <= 4:
            # Tolerant
            return {
                'patience': int(self.base_patience * 1.5),
                'decay_threshold': self.base_decay_threshold * 0.5,
                'window_size': self.base_window_size
            }
        else:
            # Very tolerant
            return {
                'patience': max(15, self.base_patience * 2),
                'decay_threshold': self.base_decay_threshold * 0.01,
                'window_size': self.base_window_size * 2
            }
    
    def _apply_adaptive_parameters(self, runner, n_samples, reason):
        """Apply adaptive parameters and log changes."""
        old_params = {
            'patience': self.patience,
            'decay_threshold': self.decay_threshold,
            'window_size': self.window_size
        }
        
        new_params = self._get_adaptive_parameters(n_samples)
        
        self.patience = new_params['patience']
        self.decay_threshold = new_params['decay_threshold']
        self.window_size = new_params['window_size']
        self.patience_counter = 0
        
        # Log changes
        runner.logger.info(f"[Adaptive Params] {reason} with {n_samples} samples:")
        runner.logger.info(f"  - patience: {old_params['patience']} -> {self.patience}")
        runner.logger.info(f"  - decay_threshold: {old_params['decay_threshold']:.6f} -> {self.decay_threshold:.6f}")
        runner.logger.info(f"  - window_size: {old_params['window_size']} -> {self.window_size}")
    
    def before_train_iter(self, runner):
        """Initialize adaptive params on first iter and check schedule."""
        current_iter = runner.iter
        
        # Apply adaptive parameters on first iteration
        if current_iter == 0:
            dataset = self._get_dataset(runner)
            if hasattr(dataset, 'get_current_sample_count'):
                initial_count = dataset.get_current_sample_count()
                self._apply_adaptive_parameters(runner, initial_count, "Initialized")
        
        # Regular schedule check
        if current_iter > 0 and (current_iter - self.last_add_iter) >= self.iters_per_stage:
            self._try_add_samples(runner, current_iter, "regular schedule")
    
    def after_train_iter(self, runner):
        """Monitor loss decay and manage LR warmup."""
        current_iter = runner.iter
        
        # Track loss
        loss_value = runner.outputs['loss'].item()
        self.loss_history.append(loss_value)
        
        # Keep history bounded
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
        
        # Check loss decay if we have enough data
        if len(self.loss_history) >= self.window_size:
            self._check_loss_decay(runner, current_iter)
        
        # Restore LR after warmup
        if self.warmup_start_iter is not None:
            if (current_iter - self.warmup_start_iter) >= self.lr_warmup_iters:
                self._restore_lr(runner)
        
        # Log metrics periodically
        if current_iter % 50 == 0:
            self._log_metrics(runner)
    
    def _check_loss_decay(self, runner, current_iter):
        """Check if loss decay is too slow."""
        recent_losses = self.loss_history[-self.window_size:]
        decay_rate = self._calculate_decay_rate(recent_losses)

        if decay_rate < self.decay_threshold:
            self.patience_counter += 1
            runner.logger.info(
                f'[Loss Decay] Slow decay: {decay_rate:.6f}, '
                f'patience: {self.patience_counter}/{self.patience}'
            )
            
            if self.patience_counter >= self.patience:
                # Try to add samples instead of stopping
                if self._try_add_samples(runner, current_iter, f"slow loss decay (rate={decay_rate:.6f})"):
                    self.patience_counter = 0
                else:
                    self._early_stop_training(runner, decay_rate)
        else:
            self.patience_counter = 0
    
    def _try_add_samples(self, runner, current_iter, reason):
        """Try to add samples to dataset."""
        dataset = self._get_dataset(runner)
        
        if not hasattr(dataset, 'add_next_samples'):
            return False
        
        old_count = dataset.get_current_sample_count()
        success = dataset.add_next_samples()
        
        if success:
            self.stage_count += 1
            self.last_add_iter = current_iter
            new_count = dataset.get_current_sample_count()
            
            # Apply new adaptive parameters
            self._apply_adaptive_parameters(runner, new_count, f"Stage {self.stage_count}")
            
            # Reset iterator and loss history
            if hasattr(runner.data_loader, '_iterator'):
                runner.data_loader._iterator = None
            self.loss_history = []
            
            # Apply LR warmup if enabled
            if self.lr_warmup_after_add and hasattr(runner, 'optimizer'):
                self._apply_lr_warmup(runner, current_iter)
            
            # Log sample addition
            runner.logger.info(
                f"[Adaptive Learning] Stage {self.stage_count} at iter {current_iter}: "
                f"samples {old_count} -> {new_count} (reason: {reason})"
            )
            
            # Check if reached max
            if new_count >= dataset.max_samples:
                runner.logger.info(
                    f"[Adaptive Learning] Reached maximum samples: {new_count}/{dataset.max_samples}"
                )
            
            # Save checkpoint
            if self.save_checkpoint:
                self._save_checkpoint(runner, new_count)
            
            return True
        
        return False
    
    def _early_stop_training(self, runner, decay_rate):
        """Stop training when loss decay is too slow and no more samples."""
        checkpoint_name = f'early_stop_iter_{runner.iter}.pth'
        
        runner.save_checkpoint(
            runner.work_dir,
            filename_tmpl=checkpoint_name,
            save_optimizer=True,
            meta=dict(
                iter=runner.iter,
                early_stop=True,
                final_loss=self.loss_history[-1],
                decay_rate=decay_rate
            )
        )
        
        runner.logger.info(f'[Early Stop] Saved checkpoint: {checkpoint_name}')
        runner.logger.info('[Early Stop] Loss decay too slow and no more samples!')
        raise RuntimeError('Early stopping: train loss decay rate below threshold')
    
    def _calculate_decay_rate(self, losses):
        """Calculate decay rate using linear regression."""
        x = np.arange(len(losses))
        slope = np.polyfit(x, losses, 1)[0]
        return -slope  # Negative slope = positive decay
    
    def _apply_lr_warmup(self, runner, current_iter):
        """Temporarily increase learning rate after adding samples."""
        if self.original_lr is None:
            self.original_lr = runner.optimizer.param_groups[0]['lr']
        
        new_lr = self.original_lr * self.lr_warmup_factor
        for param_group in runner.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.warmup_start_iter = current_iter
        
        runner.logger.info(
            f"[LR Warmup] Increased: {self.original_lr:.2e} -> {new_lr:.2e} "
            f"for {self.lr_warmup_iters} iters"
        )
    
    def _restore_lr(self, runner):
        """Restore original learning rate after warmup."""
        if self.original_lr is not None:
            for param_group in runner.optimizer.param_groups:
                param_group['lr'] = self.original_lr
            
            runner.logger.info(f"[LR Warmup] Restored to {self.original_lr:.2e}")
            self.warmup_start_iter = None
    
    def _log_metrics(self, runner):
        """Log adaptive learning metrics."""
        dataset = self._get_dataset(runner)
        
        if hasattr(dataset, 'get_current_sample_count') and hasattr(runner, 'log_buffer'):
            current_samples = dataset.get_current_sample_count()
            
            runner.log_buffer.output['adaptive/current_samples'] = current_samples
            runner.log_buffer.output['adaptive/stage'] = self.stage_count
            
            if len(self.loss_history) >= 2:
                recent_losses = self.loss_history[-min(self.window_size, len(self.loss_history)):]
                decay_rate = self._calculate_decay_rate(recent_losses)
                runner.log_buffer.output['adaptive/loss_decay_rate'] = decay_rate
    
    def _save_checkpoint(self, runner, sample_count):
        """Save checkpoint with adaptive learning info."""
        checkpoint_name = f"iter_{runner.iter}_samples_{sample_count}.pth"
        
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
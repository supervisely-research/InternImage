from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class AdaptiveLearningHook(Hook):
    """Adaptive learning hook for adding samples during training.
    
    This hook works with AdaptiveTomatoDataset to dynamically add samples
    at regular intervals during training.
    
    Args:
        iters_per_stage (int): Interval for adding samples (in iterations).
            Default: 150.
        save_checkpoint (bool): Whether to save checkpoint when adding samples.
            Default: True.
    """
    
    def __init__(
        self,
        iters_per_stage: int = 150,
        save_checkpoint: bool = True,
    ):
        self.iters_per_stage = iters_per_stage
        self.save_checkpoint = save_checkpoint
        
        # State
        self.stage_count = 0
        self.last_add_iter = 0
    
    def _get_dataset(self, runner):
        """Helper to get dataset from runner."""
        if hasattr(runner.data_loader, '_dataloader'):
            return runner.data_loader._dataloader.dataset
        return runner.data_loader.dataset
    
    def before_train_iter(self, runner):
        """Check if it's time to add samples."""
        current_iter = runner.iter
        
        # Regular schedule check
        if current_iter > 0 and (current_iter - self.last_add_iter) >= self.iters_per_stage:
            self._try_add_samples(runner, current_iter, "regular schedule")
    
    def _try_add_samples(self, runner, current_iter, reason):
        """Try to add samples to dataset."""
        dataset = self._get_dataset(runner)
        
        if not hasattr(dataset, 'add_next_samples'):
            runner.logger.warning(
                '[Adaptive Learning] Dataset does not support add_next_samples()'
            )
            return False
        
        old_count = dataset.get_current_sample_count()
        success = dataset.add_next_samples()
        
        if success:
            self.stage_count += 1
            self.last_add_iter = current_iter
            new_count = dataset.get_current_sample_count()
            
            # Reset iterator
            if hasattr(runner.data_loader, '_iterator'):
                runner.data_loader._iterator = None
            
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
import numpy as np
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class TrainLossDecayStoppingHook(Hook):
    """Early stop when train loss decay rate slows down.
    
    This hook monitors the training loss and stops training when the decay
    rate becomes too slow, indicating potential overfitting or convergence.
    
    Args:
        window_size (int): Number of iterations to consider for decay calculation.
            Default: 20
        decay_threshold (float): Minimum decay rate to continue training.
            If decay rate falls below this threshold, training will stop.
            Default: 0.001
        patience (int): Number of consecutive slow decay checks before stopping.
            Default: 5
    """
    
    def __init__(self, 
                 window_size=20,
                 decay_threshold=0.001,
                 patience=5):
        self.window_size = window_size
        self.decay_threshold = decay_threshold  
        self.patience = patience
        self.loss_history = []
        self.patience_counter = 0
        
    def after_train_iter(self, runner):
        """Check loss decay after each training iteration."""
        # Get loss from log buffer
        if not hasattr(runner, 'log_buffer') or 'loss' not in runner.log_buffer.output:
            return
            
        loss_value = runner.log_buffer.output['loss']
        self.loss_history.append(loss_value)
        
        # Wait until we have enough data
        if len(self.loss_history) < self.window_size:
            return
            
        # Keep only recent history (2x window for stability)
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
        
        # Calculate decay rate over window
        recent_losses = self.loss_history[-self.window_size:]
        if len(recent_losses) >= self.window_size:
            decay_rate = self._calculate_decay_rate(recent_losses)
            print(f'decay rate: {decay_rate}, threshold: {self.decay_threshold}')
            
            if decay_rate < self.decay_threshold:
                self.patience_counter += 1
                runner.logger.info(
                    f'[Early Stop] Slow decay detected: {decay_rate:.6f}, '
                    f'patience: {self.patience_counter}/{self.patience}'
                )
                
                if self.patience_counter >= self.patience:
                    # Save checkpoint before stopping
                    checkpoint_name = f'early_stop_iter_{runner.iter}.pth'
                    runner.logger.info(f'[Early Stop] Saving checkpoint: {checkpoint_name}')
                    
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
                    
                    runner.logger.info(
                        '[Early Stop] Train loss decay rate too slow, stopping training!'
                    )
                    raise RuntimeError('Early stopping: train loss decay rate below threshold')
            else:
                # Reset counter if decay is good
                self.patience_counter = 0
                
    def _calculate_decay_rate(self, losses):
        """Calculate decay rate using linear regression.
        
        Fits a linear trend to losses to detect decay rate.
        Negative slope indicates loss is decreasing (positive decay rate).
        
        Args:
            losses (list): List of recent loss values
            
        Returns:
            float: Decay rate (positive = decreasing loss)
        """
        x = np.arange(len(losses))
        slope = np.polyfit(x, losses, 1)[0]
        return -slope  # Negative slope = positive decay rate
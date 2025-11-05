import numpy as np
from mmcv.runner import HOOKS, Hook
from torch.utils.tensorboard import SummaryWriter


class LossPlateauDetector:
    """Detect plateau in training loss using moving average comparison.
    
    This detector monitors the training loss and detects when it has plateaued
    by comparing the moving average of recent losses with the previous window.
    
    Args:
        window_size (int): Number of iterations to use for computing the 
            moving average. Default: 50.
        threshold (float): Relative change threshold below which plateau is 
            detected (e.g., 0.005 means 0.5% change). Default: 0.005.
        patience (int): Number of consecutive plateau detections required before
            taking action. This prevents premature stopping due to temporary
            fluctuations. Default: 1.
        check_interval (int): How often to check for plateau (every N iterations).
            If None, defaults to window_size.
            Default: None.
        log_tensorboard (bool): Whether to log metrics to TensorBoard.
            Default: False.
    """

    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 0.005,
        patience: int = 1,
        check_interval: int = 1,
        log_tensorboard: bool = False,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.check_interval = check_interval or window_size
        self.patience = patience
        self._min_iterations = 2 * window_size
        
        if log_tensorboard:
            self._tb_writer = SummaryWriter(log_dir='loss_plateau_logs')
        else:
            self._tb_writer = None

        # State variables
        self.loss_history = []
        self.consecutive_plateau_count = 0
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.loss_history = []
        self.consecutive_plateau_count = 0
        self._min_iterations = 2 * self.window_size

    def step(
        self,
        loss: float,
        current_iter: int,
    ) -> bool:
        """Process one training iteration.
        
        Args:
            loss (float): Current training loss value.
            current_iter (int): Current iteration number.
            
        Returns:
            bool: True if plateau is detected (patience threshold reached),
                  False otherwise.
        """
        self.loss_history.append(loss)
        
        if self._tb_writer:
            self._tb_writer.add_scalar('loss_plateau_detector/loss', loss, current_iter)

        # Only check at specified intervals
        if (current_iter + 1) % self.check_interval != 0:
            return False
        
        # Check if we have enough data
        if len(self.loss_history) < self._min_iterations:
            return False
        
        # Perform plateau detection
        is_plateau, info = self._check_plateau(current_iter)
        
        if self._tb_writer:
            self._tb_writer.add_scalar('loss_plateau_detector/metric', info['metric'], current_iter)
        
        # Handle plateau detection
        if is_plateau:
            self.consecutive_plateau_count += 1
            print(
                f'[Plateau Detection] Iteration {current_iter}: '
                f'Plateau signal {self.consecutive_plateau_count}/{self.patience}\n'
                f'    Change: {info["metric"]:.6f} '
                f'(Previous Avg: {info["previous_avg"]:.6f}, Current Avg: {info["current_avg"]:.6f}) '
                f'(threshold: {self.threshold})'
            )
            
            # Check if we've reached patience threshold
            if self.consecutive_plateau_count >= self.patience:
                return True  # Indicate plateau detected
        else:
            # Reset counter if no plateau
            self.consecutive_plateau_count = 0
        
        return False
    
    def _check_plateau(self, current_iter: int) -> tuple:
        """Check if loss has plateaued by comparing window averages.
        
        Returns:
            tuple: (is_plateau, info_dict)
        """
        # Calculate current window average (last N steps)
        current_window = self.loss_history[-self.window_size:]
        current_avg = np.mean(current_window)
        
        # Calculate previous window average (N steps before current window)
        previous_window = self.loss_history[-2*self.window_size:-self.window_size]
        previous_avg = np.mean(previous_window)
        
        change = previous_avg - current_avg
        is_plateau = change < self.threshold
        
        info = {
            'iter': current_iter,
            'metric': change,
            'threshold': self.threshold,
            'previous_avg': previous_avg,
            'current_avg': current_avg,
        }
        
        return is_plateau, info


@HOOKS.register_module()
class LossPlateauEarlyStopHook(Hook):
    """Early stopping hook based on loss plateau detection.
    
    This hook uses LossPlateauDetector to monitor training loss and stops
    training when the loss plateaus (stops improving significantly).
    
    Args:
        window_size (int): Number of iterations for moving average window.
            Default: 20.
        threshold (float): Minimum change between windows to avoid plateau.
            Default: 0.005.
        patience (int): Number of consecutive plateau signals before stopping.
            Default: 1.
        check_interval (int): Check frequency in iterations. Default: 1.
        log_tensorboard (bool): Enable TensorBoard logging. Default: False.
        save_checkpoint_on_stop (bool): Save checkpoint when early stopping.
            Default: True.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        threshold: float = 0.005,
        patience: int = 1,
        check_interval: int = 1,
        log_tensorboard: bool = False,
        save_checkpoint_on_stop: bool = True,
    ):
        self.detector = LossPlateauDetector(
            window_size=window_size,
            threshold=threshold,
            patience=patience,
            check_interval=check_interval,
            log_tensorboard=log_tensorboard,
        )
        self.save_checkpoint_on_stop = save_checkpoint_on_stop
    
    def after_train_iter(self, runner):
        """Called after each training iteration.
        
        Args:
            runner: MMSegmentation runner instance.
        """
        current_iter = runner.iter
        loss_value = runner.outputs['loss'].item()
        
        # Check for plateau
        plateau_detected = self.detector.step(loss_value, current_iter)
        
        if plateau_detected:
            self._handle_early_stop(runner)
    
    def _handle_early_stop(self, runner):
        """Handle early stopping procedure.
        
        Args:
            runner: MMSegmentation runner instance.
        """
        runner.logger.info('[Early Stop] Loss plateau detected!')
        runner.logger.info(
            f'[Early Stop] Training stopped at iteration {runner.iter}'
        )
        
        # Save checkpoint if enabled
        if self.save_checkpoint_on_stop:
            checkpoint_name = f'early_stop_iter_{runner.iter}.pth'
            
            runner.save_checkpoint(
                runner.work_dir,
                filename_tmpl=checkpoint_name,
                save_optimizer=True,
                meta=dict(
                    iter=runner.iter,
                    early_stop=True,
                    final_loss=self.detector.loss_history[-1] if self.detector.loss_history else None,
                    consecutive_plateau_count=self.detector.consecutive_plateau_count,
                )
            )
            
            runner.logger.info(f'[Early Stop] Saved checkpoint: {checkpoint_name}')
        
        # Stop training
        raise RuntimeError('Early stopping: loss plateau detected')
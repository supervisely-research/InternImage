# --------------------------------------------------------
# Adaptive Learning + Loss Plateau Early Stop Config
# Combines sample addition during training with early stopping
# --------------------------------------------------------

_base_ = [
    '../_base_/models/upernet_r50.py', 
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# Import custom modules
import sys
sys.path.append('/root/workspace/InternImage/segmentation/mmseg_custom')
from datasets import AdaptiveTomatoDataset
from hooks import AdaptiveLearningHook, LossPlateauEarlyStopHook

num_classes = 8
pretrained = '/root/workspace/InternImage/upernet_internimage_l_640_160k_ade20k.pth'

# Model configuration
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=num_classes, in_channels=[160, 320, 640, 1280]),
    auxiliary_head=dict(num_classes=num_classes, in_channels=640),
    test_cfg=dict(mode='whole'))

# Image normalization
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2560, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Optimizer
optimizer = dict(
    _delete_=True, 
    type='AdamW', 
    lr=0.00002,
    betas=(0.9, 0.999), 
    weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=37, 
        layer_decay_rate=0.94,
        depths=[5, 5, 22, 5], 
        offset_lr_scale=1.0))

# Learning rate schedule - FIXED LR with warmup only
lr_config = dict(
    policy='fixed',          # fixed LR after warmup
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1e-3,
    by_epoch=False
)

# Adaptive dataset configuration
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        _delete_=True,
        type='AdaptiveTomatoDataset',
        data_root='/root/workspace/InternImage/data/train_5shot_seed0/',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline,
        max_samples=5,           # maximum samples to use
        initial_samples=2,       # start with 2 samples
        samples_per_stage=3      # add 1 sample at a time
    ),
    val=dict(
        type='TomatoDataset',
        data_root='/root/workspace/InternImage/data/val/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline),
    test=dict(
        type='TomatoDataset',
        data_root='/root/workspace/InternImage/data/val/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline))

# Runner
runner = dict(type='IterBasedRunner', max_iters=2000)

# Optimizer config
optimizer_config = dict(
    _delete_=True, 
    grad_clip=dict(max_norm=0.1, norm_type=2))

# Checkpoint config
checkpoint_config = dict(
    by_epoch=False, 
    interval=500, 
    max_keep_ckpts=3)

# Evaluation
evaluation = dict(
    interval=10, 
    metric='mIoU', 
    save_best='mIoU')

# Custom hooks - TWO SEPARATE HOOKS
custom_hooks = [
    # Hook 1: Adaptive Learning (adds samples on schedule)
    dict(
        type='AdaptiveLearningHook',
        iters_per_stage=500,         # add samples every 400 iterations
        save_checkpoint=True,
        lr_warmup_after_add=False,   # optional LR warmup after adding samples
        lr_warmup_factor=2.0,
        lr_warmup_iters=50
    ),
    # Hook 2: Early Stop (stops training on plateau)
    dict(
        type='LossPlateauEarlyStopHook',
        window_size=40,              # window for moving average
        threshold=0.001,             # minimum change to avoid plateau
        patience=10,                 # consecutive plateau signals before stop
        check_interval=1,            # check every iteration
        log_tensorboard=False,
        save_checkpoint_on_stop=True
    )
]

# Logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
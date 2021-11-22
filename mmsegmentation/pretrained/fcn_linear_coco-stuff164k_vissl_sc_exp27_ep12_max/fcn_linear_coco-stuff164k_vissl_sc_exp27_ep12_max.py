norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderVISSLFCN',
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=256,
        num_convs=1,
        kernel_size=1,
        concat_input=False,
        dropout_ratio=0.0,
        num_classes=27,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        act_cfg=None,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    vissl_params=dict(
        vissl_dir='/home/r_karlsson/workspace6/vissl',
        config_path='sc_exp27/dense_swav_8node_resnet_coco_exp27.yaml',
        checkpoint_path='sc_exp27/model_final_checkpoint_phase11.torch',
        output_type='trunk',
        default_config_path='vissl/config/defaults.yaml'))
dataset_type = 'COCOStuffCoarseDataset'
data_root = 'data/coco_stuff164k_coarse'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='COCOStuffCoarseDataset',
        data_root='data/coco_stuff164k_coarse',
        img_dir='images_coarse/train2017',
        ann_dir='annotations_coarse/train2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='COCOStuffCoarseDataset',
        data_root='data/coco_stuff164k_coarse',
        img_dir='images_coarse/val2017',
        ann_dir='annotations_coarse/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='COCOStuffCoarseDataset',
        data_root='data/coco_stuff164k_coarse',
        img_dir='images_coarse/val2017',
        ann_dir='annotations_coarse/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=12000)
checkpoint_config = dict(by_epoch=False, interval=12000)
evaluation = dict(interval=3000, metric='mIoU', pre_eval=True)
work_dir = 'fcn_linear_coco-stuff164k_vissl_sc_exp27_ep12_max'
gpu_ids = range(0, 8)

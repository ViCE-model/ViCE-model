_base_ = [
    '_base_/models/fcn_vissl.py', '_base_/datasets/coco-stuff164k_coarse.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_12k.py'
]
model = dict(
    decode_head=dict(
        in_channels=128,
        channels=256,
        num_convs=1,
        kernel_size=1,
        num_classes=27,
        act_cfg=None),
    vissl_params=dict(
        config_path='pretrained/exp27_coco/vice_8node_resnet_coco_exp27.yaml',
        checkpoint_path='pretrained/exp27_coco/model_final_checkpoint_phase11.torch'
    ))

optimizer = dict(lr=0.01)  # default0.01

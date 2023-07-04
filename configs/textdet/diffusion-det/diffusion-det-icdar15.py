file_client_args = dict(backend='disk')

custom_imports = dict(
    imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)

model = dict(
    type='MMDetWrapper',
    text_repr_type='poly',
    cfg=dict(
        type='DiffusionDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
            _scope_='mmocr',
            type='CLIPResNet',
            init_cfg=dict(
                type='Pretrained',
                checkpoint=
                'https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth'
            )),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    bbox_head=dict(
        type='DynamicDiffusionDetHead',
        num_classes=1,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        single_head=dict(
            type='SingleDiffusionDetHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type='DiffusionDetCriterion',
            num_classes=1,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum',
                           loss_weight=2.0))),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    )))
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=0.12549019607843137,
        saturation=0.5,
        contrast=0.5),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(1.0, 4.125),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='TextDetRandomCrop', target_size=(640, 640)),
    dict(type='MMOCR2MMDet', poly2mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'flip',
                   'scale_factor', 'flip_direction'))
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
icdar2015_textdet_data_root = 'data/icdar2015'
icdar2015_textdet_train = dict(
    type='OCRDataset',
    data_root='data/icdar2015',
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            color_type='color_ignore_orientation'),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True),
        dict(
            type='TorchVisionWrapper',
            op='ColorJitter',
            brightness=0.12549019607843137,
            saturation=0.5,
            contrast=0.5),
        dict(
            type='RandomResize',
            scale=(640, 640),
            ratio_range=(1.0, 4.125),
            keep_ratio=True),
        dict(type='RandomFlip', prob=0.5),
        dict(type='TextDetRandomCrop', target_size=(640, 640)),
        dict(type='MMOCR2MMDet', poly2mask=True),
        dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'flip',
                       'scale_factor', 'flip_direction'))
    ])
icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root='data/icdar2015',
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            color_type='color_ignore_orientation'),
        dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
        dict(
            type='LoadOCRAnnotations',
            with_polygon=True,
            with_bbox=True,
            with_label=True),
        dict(
            type='PackTextDetInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ])
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=None)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',by_epoch=False, interval=75000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))
log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=False)
load_from = None
resume = False
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2))
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=450000,
    val_interval=75000)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=450000,
        by_epoch=False,
        milestones=[350000, 420000],
        gamma=0.1)
]
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

backend = 'pillow'

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textdet_train.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.12549019607843137,
                saturation=0.5,
                contrast=0.5),
            dict(
                type='RandomResize',
                scale=(640, 640),
                ratio_range=(1.0, 4.125),
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='TextDetRandomCrop', target_size=(640, 640)),
            dict(type='MMOCR2MMDet', poly2mask=True),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape', 'flip',
                           'scale_factor', 'flip_direction'))
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textdet_test.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='data/icdar2015',
        ann_file='textdet_test.json',
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type='color_ignore_orientation'),
            dict(type='Resize', scale=(1920, 1920), keep_ratio=True),
            dict(
                type='LoadOCRAnnotations',
                with_polygon=True,
                with_bbox=True,
                with_label=True),
            dict(
                type='PackTextDetInputs',
                meta_keys=('img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
auto_scale_lr = dict(base_batch_size=4)
launcher = 'none'
work_dir = './work-dirs/difdet-oclip'

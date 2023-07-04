file_client_args = dict(backend='disk')

model = dict(
    type='MMDetWrapper',
    text_repr_type='poly',
    cfg=dict(
        type='CascadeRCNN',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=False,
            pad_size_divisor=32),
        backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)),
        _scope_='mmdet'))
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
ctw_det_data_root = '/content/mmocr/ctw1500'
ctw_det_train = dict(
    type='OCRDataset',
    data_root='/content/mmocr/ctw1500',
    ann_file='instances_training.json',
    data_prefix=dict(img_path='imgs/'),
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
ctw_det_test = dict(
    type='OCRDataset',
    data_root='/content/mmocr/ctw1500',
    ann_file='instances_test.json',
    data_prefix=dict(img_path='imgs/'),
    test_mode=True,
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            color_type='color_ignore_orientation'),
        dict(type='Resize', scale=(1600, 1600), keep_ratio=True),
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
    checkpoint=dict(type='CheckpointHook', interval=20),
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
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
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
    optimizer=dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=160, val_interval=20)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[80, 128], end=160)
]
ctw_test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1600, 1600), keep_ratio=True),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRDataset',
        data_root='/content/mmocr/ctw1500',
        ann_file='instances_training.json',
        data_prefix=dict(img_path='imgs/'),
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
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='OCRDataset',
        data_root='/content/mmocr/ctw1500',
        ann_file='instances_test.json',
        data_prefix=dict(img_path='imgs/'),

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
        data_root='/content/mmocr/ctw1500',
        ann_file='instances_test.json',
        data_prefix=dict(img_path='imgs/'),
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
auto_scale_lr = dict(base_batch_size=8)
launcher = 'none'
work_dir = './work_dirs/mask-rcnn_resnet50_fpn_160e_ctw'

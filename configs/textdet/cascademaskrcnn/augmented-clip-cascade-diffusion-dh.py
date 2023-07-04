file_client_args = dict(backend='disk')
custom_imports = dict(
    imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)
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
        rpn_head=dict(
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
                num_reg_convs=1,
                dim_feedforward=2048,
                num_heads=8,
                dropout=0.0,
                act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv=dict(dynamic_dim=64, dynamic_num=2)),
            roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
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
                            eps=1e-08),
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
                loss_bbox=dict(
                    type='L1Loss', reduction='sum', loss_weight=5.0),
                loss_giou=dict(
                    type='GIoULoss', reduction='sum', loss_weight=2.0))),
        roi_head=dict(
            type='CascadeRoIHead',
            reg_roi_scale_factor=1.3,
            num_stages=3,
            stage_loss_weights=[1, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=[
                dict(
                    type='DoubleConvFCBBoxHead',
                    num_convs=4,
                    num_fcs=2,
                    in_channels=256,
                    conv_out_channels=1024,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                dict(
                    type='DoubleConvFCBBoxHead',
                    num_convs=4,
                    num_fcs=2,
                    in_channels=256,
                    conv_out_channels=1024,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
                dict(
                    type='DoubleConvFCBBoxHead',
                    num_convs=4,
                    num_fcs=2,
                    in_channels=256,
                    conv_out_channels=1024,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=1,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            ],
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
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
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
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
resume = True
val_evaluator = dict(type='HmeanIOUMetric')
test_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TextDetLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=160, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[80, 128], end=160)
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
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
        ]))
val_dataloader = dict(
    batch_size=10,
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
auto_scale_lr = dict(base_batch_size=2)
launcher = 'none'
work_dir = '/content/drive/MyDrive/OCR/ckpt/augmented-clip-cascade-diffusion-dh'

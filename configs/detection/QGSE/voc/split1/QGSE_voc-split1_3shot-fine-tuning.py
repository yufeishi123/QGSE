_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../meta-rcnn_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT1_3SHOT')],
            num_novel_shots=3,
            num_base_shots=3,
            classes='ALL_CLASSES_SPLIT1',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'),
    model_init=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=1000, class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=1000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=4000)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/meta-rcnn_r101_c4_8xb4_voc-split1_base-training/latest.pth'
# model settings
model = dict(
    information_BGSM=dict(
        type='BGSMModule'
    ),
    information_QGM=dict(
        type='QGMModule',
        in_channels=1024,
        inter_channels=1024,
        dimension=2,
        sub_sample=False,
        bn_layer=True
    ),
    roi_head=dict(
        shared_head=dict(
            type='MetaRCNNResLayer',
            depth=101,),
        bbox_head=dict(
            num_classes=20,
            num_meta_classes=20
        ),
        information_VSEM=dict(
            type='VSEMModule'
        )
    ),
    frozen_parameters=[
    'backbone', 'shared_head', 'rpn_head', 'aggregation_layer','roi_head.VSEM.clip'
])
model = dict(
    name='Detector',
)

#backbone = dict(
#    name='ResNet',
#    output_stride=32,
#    multi_grid=[1, 1, 1],
#    return_idx=[0,1,2,3],
#    pretrained='https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
#)

backbone = dict(
    name='ResNetWrapper',
    resnet='resnet18',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

featuremap_out_channel = 512

griding_num = 100
num_classes = 6
heads = dict(name='LaneCls',
        dim = (griding_num + 1, 56, num_classes))

epochs = 150
batch_size = 30
total_iter = (3616 // batch_size + 1) * epochs 

lr_scheduler = dict(
    name = 'PolynomialLR',
    decay_batch  = total_iter,
    power=0.9
)

optimizer = dict(
  name = 'SGD',
  lr=0.025,
  weight_decay = 1e-4,
  momentum = 0.9
)

ori_img_h = 720
ori_img_w = 1280
img_h = 288
img_w = 800
cut_height=0
sample_y = range(710, 150, -10)

img_norm = dict(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

row_anchor = 'tusimple_row_anchor'

train_transform = [
    dict(name='RandomRotation', degree=(-6, 6)),
    dict(name='RandomUDoffsetLABEL', max_offset=100),
    dict(name='RandomLROffsetLABEL', max_offset=200),
    dict(name='GenerateLaneCls', row_anchor=row_anchor,
        num_cols=griding_num, num_classes=num_classes),
    dict(name='Resize', size=(img_w, img_h)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img', 'cls_label']),
]

val_transform = [
    dict(name='Resize', size=(img_w, img_h)),
    dict(name='Normalize', img_norm=img_norm),
    dict(name='ToTensor', keys=['img']),
]

dataset_path = '/home/fyj/zky/tusimple'

dataset = dict(
    train=dict(
        name='CULane',
        data_root=dataset_path,
        split='train',
        processes=train_transform,
    ),
    val=dict(
        name='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_transform,
    ),
    test=dict(
        name='CULane',
        data_root=dataset_path,
        split='test',
        processes=val_transform,
    )
)

log_config = dict(
    name = 'LogHook',
    interval = 50
    )

custom_config = [dict(
    name = 'EvaluateHook'
    )]

device = 'gpu'
seed =  0
save_inference_dir = './inference'
output_dir = './output_dir'
best_dir = './output_dir/best_dir'
pred_save_dir = './pred_save'
num_workers = 4
num_classes = 6 + 1
view = False
ignore_label = 255
test_json_file='/home/fyj/zky/tusimple/test_label.json'
seg=False
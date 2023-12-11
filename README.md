
<font size=4> 简体中文
## 🚀FlowLane: 基于Oneflow的车道线检测的工具包

<font size=3> 在这个项目中，我们开发了FlowLane用于车道线检测。我们致力于为Oneflow社区开发一款实用的车道线检测工具包，欢迎加入我们来完善FlowLane。如果您觉得FlowLane不错，可以给我们项目一个star。


## 🆕新闻
在这个部分中，我们展示FlowLane中最新的改进。
<ul class="nobull">
  <li>[2023-12-11] :fire: 发布FlowLanev1,并提供UFLD模型的复现和预训练权重，以及CULane数据集的训练。

</ul>

## 👀介绍
FlowLane是一个基于Oneflow的车道线检测工具包。Oneflow是一种高性能的深度学习框架。FlowLane开发的初衷是希望科研人员或者工程师能够通过一个框架方便地开发各类车道线检测算法。如果您对FlowLane有任何疑问或者建议，欢迎和我联系。

## 🌟框架总览

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>模型</b>
      </td>
      <td colspan="2">
        <b>框架组件</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Segmentation based</b></summary>
          <ul>
          </ul>
        </details>
        <details><summary><b>Keypoint(anchor) based</b></summary>
          <ul>
            <li><a href="configs\ufld">UFLD</a></li>
          </ul>
        </details>
        <details><summary><b>GAN based</b></summary>
          <ul>
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Backbones</b></summary>
          <ul>
            <li><a href="flowlane\model\backbones\resnet.py">ResNet</a></li>
          </ul>
        </details>
        <details><summary><b>Necks</b></summary>
          <ul>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="flowlane\model\losses\focal_loss.py">Focal Loss</a></li>
          </ul>
        </details>
        <details><summary><b>Metrics</b></summary>
          <ul>
            <li>Accuracy</li>
            <li>FP</li>
            <li>FN</li>
          </ul>  
        </details>
      </td>
      <td>
        <details><summary><b>Datasets</b></summary>
          <ul>
            <li><a href="flowlane\datasets\culane.py">CULane</a></li>
          </ul>
        </details>
        <details><summary><b>Data Augmentation</b></summary>
          <ul>
            <li>RandomLROffsetLABEL</li>  
            <li>Resize</li>  
            <li>RandomUDoffsetLABEL</li>
            <li>RandomCrop</li>
            <li>CenterCrop</li>  
            <li>RandomRotation</li>  
            <li>RandomBlur</li>
            <li>Normalize</li>
            <li>RandomHorizontalFlip</li>
            <li>Colorjitters</li>
            <li>RandomErasings</li>
            <li>GaussianBlur</li>
            <li>RandomGrayScale</li>
            <li>Alaug</li> 
          </ul>
        </details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>


## 🛠️安装
<details>
<summary>具体步骤</summary>

 步骤1 安装 oneflow
```Shell
conda create -n flowlane python=3.8 -y
conda activate flowlane
python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu117 #具体的cuda和oneflow版本可以参照官网
```

 步骤2 Git clone FlowLane
```Shell
git clone https://github.com/zkyseu/FlowLane
```

 步骤3 安装必要的依赖库
```Shell
cd FlowLane
pip install -r requirements.txt
```
</details>

## 📘数据集准备(CULane和Tusimple为例)
### CULane
<details>
<summary>CULane数据集准备步骤</summary>

下载 [CULane](https://xingangpan.github.io/projects/CULane.html). 接着解压到 `$CULANEROOT`. 创建 `data` 目录.

```Shell
cd $LANEDET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

对于CULane数据集, 完成以上步骤你应该有下列数据集结构:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```
</details>

### 自制数据集
我们将会在后续版本发布自制数据集的教程。

## 💎开始快乐炼丹
### 1、训练的命令
<details>
<summary>开启训练</summary>

对于训练, 运行以下命令(shell脚本在script文件夹下)。其中config_path表示config的路径
```Shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
bash scripts/train.sh config_path
```
</details>

### 2、测试
<details>
<summary>开启测试</summary>

运行以下命令开启模型的测试，model_path表示模型权重。
```Shell
bash scripts/eval.sh config_path model_path
```
</details>


### 3、模型导出
<details>
<summary>开启模型导出</summary>

如果你想将模型导出为预训练的格式(只保留模型权重去除优化器以及学习率的权重)，可以使用以下命令，其中 model.pth为需要被导出的模型权重
```
python tools/train.py -c configs/ufld/resnet50_culane.py --export output_dir/model.pth
```
</details>

## License
PPLanedet使用[MIT license](LICENSE)。但是我们仅允许您将FlowLane用于学术用途。

## 致谢
* 非常感谢[PPLanedet](https://github.com/zkyseu/PPlanedet)提供base代码

## 引用
如果您认为我们的项目对您的研究有用，请引用我们的项目

```latex
@misc{FlowLane,
    title={FlowLane, A Toolkit for lane detection based on Oneflow},
    author={Kunyang Zhou},
    howpublished = {\url{https://github.com/zkyseu/FlowLane}},
    year={2023}
}
```



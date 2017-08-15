Rocket Launching
==============

PyTorch code for "Rocket Launching: A universal and efficient framework for training
well-performing light net" <https://arxiv.org/abs/1708.04106><br>

<img src=./img/rocket_online.png width=25%><img src=./img/vis_rocket.jpg width=75% height="60%">

## About this code
This code is based on the [attention-transfer code](https://github.com/szagoruyko/attention-transfer), the code uses [PyTorch](https://pytorch.org).

What's in this repo so far:
 * Rocket-Interval code for CIFAR-10,CIFAR-100 experiments
 * Code for Rocket-Bottom (ResNet-16-ResNet-40)
 * gradient block
 * parameter sharing

bibtex:
```
@article{zhou2017Rocket,
  title={Rocket Launching: A unified and effecient framework for training well-behaved light net},
    author={Zhou, Guorui and Fan, Ying and Cui, Runpeng and Bian, Weijie and Zhu, Xiaoqiang and Kun, Gai},
      journal={arXiv preprint arXiv:1708.04106},
        year={2017}
}
```

## Requirements

First install [PyTorch](https://pytorch.org), then install [torchnet](https://github.com/pytorch/tnt):

```
pip install git+https://github.com/pytorch/tnt.git@master
```

Then install [OpenCV](https://opencv.org) with Python bindings (e.g. `conda install -c menpo opencv3`), and other Python packages:

```
pip install -r requirements.txt
```

## Experiments

### Table 1 

This section describes how to get the results in the table 1 of the paper.

Third column, train light basic:

```
python rocket_interval.py --save logs/resnet_16_1_basic --depth 16 --width 1
python rocket_interval.py --save logs/resnet_16_2_basic --depth 16 --width 2
python rocket_bottom.py --save logs/resnet_bottom_16_1_basic --depth 16 --width 1
```

Ninth column, train booster only:

```
python rocket_interval.py --save logs/resnet_40_1_booster --depth 40 --width 1
python rocket_interval.py --save logs/resnet_40_2_booster --depth 40 --width 2
```

Fouth column, train attention transfer:
```
python rocket_interval.py --save logs/at_16_1_40_1 --width 1 --teacher_id resnet_40_1_booster --beta 1e+3
python rocket_interval.py --save logs/at_16_2_40_2 --width 2 --teacher_id resnet_40_2_booster --beta 1e+3
```

Fifth column, train with KD:
```
python rocket_interval.py --save logs/kd_16_1_16_2 --width 1 --teacher_id resnet_40_1_booster --alpha 0.9
python rocket_interval.py --save logs/kd_16_2_16_2 --width 2 --teacher_id resnet_40_2_booster --alpha 0.9
python rocket_bottom.py --save logs/kd_bottom_16_1_40_1 --teacher_id resnet_40_1_booster --alpha 0.9
```

Sixth column and eighth column, train rocket launching:
```
python rocket_interval.py --save logs/rocket_interval_16_1_40_1 --width 1 --student_depth 16  --depth 40 --gamma 0.03 
python rocket_interval.py --save logs/rocket_interval_16_2_40_2 --width 2 --student_depth 16  --depth 40 --gamma 0.03 
python rocket_bottom.py --save logs/rocket_bottom_16_1_40_1 --width 1 --student_depth 16  --depth 40 --gamma 0.03 
```

Seventh column, train rocket launching with KD:
```
python rocket_interval.py --save logs/rocket_interval_16_1_40_1_R_KD --width 1 --student_depth 16  --depth 40 --teacher_id resnet_40_1_booster --gamma 0.03 --alpha 0.9
python rocket_interval.py --save logs/rocket_interval_16_2_40_2_R_KD --width 2 --student_depth 16  --depth 40 --teacher_id resnet_40_2_booster --gamma 0.03 --alpha 0.9
python rocket_bottom.py --save logs/rocket_bottom_16_1_40_1_R_KD --width 1 --student_depth 16  --depth 40 --teacher_id resnet_40_1_booster --gamma 0.03 --alpha 0.9
```

### Table 2
rocket launching without gradient block
```
python rocket_interval.py --save logs/rocket_interval_16_1_40_1_no_gb --width 1 --student_depth 16  --depth 40 --gamma 0.03 --grad_block False
```
rocket launching without parameter sharing
```
python rocket_interval.py --save logs/rocket_interval_16_1_40_1_no_ps --width 1 --student_depth 16  --depth 40 --gamma 0.03 --param_share False
```

### Table 4 (only Cifar 100)
The same running command with Table 1
just add parameter --dataset CIFAR100

For example: rocket-launching
```
python rocket_interval.py --save logs/rocket_interval_16_1_40_1_cifar100 --width 1 --student_depth 16  --depth 40 --gamma 0.03 --dataset CIFAR100
```




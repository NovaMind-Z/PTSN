
# PTSN: Progressive Tree-Structured Prototype Network


This repository contains the reference code for the <b>ACM MM2022</b> paper "Progressive Tree-Structured Prototype Network for End-to-End Image Captioning"
<p align="center">
  <img src="images/framework.png" alt="Progressive Tree-Structured Prototype Network" width="850"/>
</p>

Note: PTSN also achieves *No.2* on the COCO image captioning online [ leaderboard ](https://competitions.codalab.org/competitions/3221#results) (Team name is CMG). More information will be released soon.

<p align="center">$\color{blue}{[Paper Access](https://dl.acm.org/doi/abs/10.1145/3503161.3548024)}$ |</p>


## Environment setup
## Data preparation
To run this code, pre-trained vision backbones, MSCOCO raw pictures and annotations should be downloaded. 
```bash
mkdir $DataPath/coco_caption/
```
1. pre-trained vision backbones
   There are several kinds of vision backbones we can use in our model. Take swintransformer as an example. Please download [SwinT-B/16 22k 224x224](https://pan.baidu.com/s/1y1Ec3UlrKSI8IMtEs-oBXA) (password:swin). As for the other backbones(e.g. SwinT-L 384x384), you can download them at their [offical link](https://github.com/microsoft/Swin-Transformer).

2. Raw data:
   please download [train2014.zip](http://images.cocodataset.org/zips/train2014.zip), [val2014.zip](http://images.cocodataset.org/zips/val2014.zip) and [test2014.zip](http://images.cocodataset.org/zips/test2014.zip). Then unzip and put these files in $DataPath/coco_caption/annotations/IMAGE_COCO/
   
3. Annotations:
   please download [annotations](https://drive.google.com/drive/folders/1tJnetunBkQ4Y5A3pq2P53yeJuGa4lX9e) and put it in $DataPath/coco_caption/annotations/

## Training procedure
To train a Swin-B version of our PTSN model, run the code below:
```bash
cd ./PTSN
python train_ptsn.py --IMG_SIZE 224 --img_root_path '~/data/dataset/coco_caption/IMAGE_COCO' --backbone_resume_path '~/data/swin_base_patch4_window7_224_22k.pth' --num_gpus 4 
```
## Evaluation

To evaluate our PTSN based on the trained checkpoints, please run the code below:
```bash
cd ./PTSN
python test_ptsn.py
```

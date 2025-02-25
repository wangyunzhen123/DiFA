## Create Environment
```
conda env create -n DiFA python=3.10
conda activate DiFA
pip install -r requirements.txt
```

## Prepare Dataset
Download NTIRE ([Baidu Disk](https://pan.baidu.com/s/1tjM5PKznKaNkwbbfekneYw?pwd=ntir), code: ntire), ICVL ([Baidu Disk](https://pan.baidu.com/s/12Tangm7beo_to8OcQKtbvg?pwd=icvl), code: icvl), Harvard([Baidu Disk](https://pan.baidu.com/s/1ui2SsR3EFMVTFBUrDvD3Zg?pwd=hard), code: hard), and then put them into the corresponding folders of data/ and recollect them as the following form:
```
|--DiFASCI
    :
    |--data
        |--NTIRE
          |--ntire_test
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene10.mat
          |--ntire_train
            |--scene1.mat
            |--scene2.mat
            :
            |--scene900.mat
          |--ntire_train.list
          |--ntire_valid.list
        |--ICVL
          |--icvl_test 
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene10.mat
          |--icvl_train 
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene100.mat
          |--icvl_train.list
          |--icvl_test.list
        |--Harvard
          |--harvard_test 
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene10.mat
          |--harvard_train 
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene200.mat
          |--harvard_train.list
          |--harvard_test.list
```

## Test:
Download the pre-trained model zoo from ([Baidu Disk](https://pan.baidu.com/s/1jS_e8gYutfJ_dMjIhmh1lQ?pwd=mzoo), code:mzoo) and place them to \model_zoo\DiFA and \model_zoo\Init_predictor. And recollect them as the following form:
```
|--DiFASCI
    :
    |--model_zoo
     |--DiFA
      |--DAUHST-DiFA
       |--dauhst_ntire_difa.pth
       |--dauhst_icvl_difa.pth
       |--dauhst_harvard_difa.pth
      |--DPU-DiFA
       :
      |--SSR-DiFA
     |--Initial_predictor
      |--hdnet.pth
      |--mst_l.pth
      |--ssr_l.pkl
      |--padut_3stg.pth
      |--dauhst_9stg.pth
      |--dpu_9stg.pkl
```

```sh
# Original model (w/o DiFA)
python inference_initail -i [image folder/image path] --ckpt [model folder/model path] --pretrained_model [initial predictor] --dataset [dataset] --gpu [gpu_id]
# initial predictor = [hdnet, mst, ssr, dauhst, padut, dpu], dataset = [ntire, icvl，harvard]. If we want to get results of DAUHST on NTIRE dataset, we can run below command
python inference_initail -i data/NTIRE/ntire_test --ckpt model_zoo/DiFA/DAUSHT-DiFA/dauhst_ntire_difa.pth --pretrained_model dauhst --dataset ntire --gpu cuda:0

# DifASCI (with DiFA)
python inference_difa -i [image folder/image path] --ckpt [model folder/model path] --pretraine_model [initial predictor] --dataset [dataset] --gpu [gpu_id]
# initial predictor = [hdnet, mst, ssr, dauhst, padut, dpu], dataset = [ntire, icvl，harvard]. If we want to get results of DAUHST-DiFA on NTIRE dataset, we can run below command
python inference_difa -i data/NTIRE/ntire_test --ckpt model_zoo/DiFA/DAUSHT-DiFA/dauhst_ntire_difa.pth --pretrained_model dauhst --dataset ntire --gpu cuda:0
```

## Train
Download the necessary pre-trained model, i.e., pretrained Teacher model Resshift, Autoencoder and MSItoRGBnetwork ([Baidu Disk](https://pan.baidu.com/s/1biDFqlwSqOhj9S7yZ12_eA?pwd=weig), code:weig), and place them to \model_zoo\weights. pretrained initial predictor ([Baidu Disk](https://pan.baidu.com/s/1jS_e8gYutfJ_dMjIhmh1lQ?pwd=mzoo), code:mzoo), place them to \model_zoo\weights. 

1. Adjust the data path in the config file. Specifically, correct and complete paths in files of [traindata](./traindata/)
2. Adjust batchsize according your GPUS.

```sh
python main.py --cfg_path configs/DiFA.yaml --save_dir logs/SinSR
```

## Acknowledgement

This project is based on [ResShift](https://github.com/zsyOAOA/ResShift), [SinSR](https://github.com/wyf0912/SinSR) and [A Toolbox for Spectral Compressive Imaging](https://github.com/caiyuanhao1998/MST). Thanks for the help from the author.

## Citation
Please cite our paper if you find our work useful. Thanks! 

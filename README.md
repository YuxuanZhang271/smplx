# Align MANO 2 SMPLX


## Table of Content
- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Run Demo Code](#run-demo-code)
- [Outputs](#outputs)
- [License](#license)


## Data Preparation

**Recording** <br>
We use ORBBEC Femto Bolt RGBD Camera to record RGB and depth data, SDK [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk.git). The data specifications are `1920x1080, 30fps`. <br>
For better recording with `multi devices` and to record data `in pieces`, you can also refer to my own [pyorbbecsdk](https://github.com/YuxuanZhang271/pyorbbecsdk.git) repo. 

**Tracking** <br>
Body Tracking: [4D-Humans](https://github.com/shubham-goel/4D-Humans.git) <br>
Hand Tracking: [HAMER](https://github.com/geopavlakos/hamer.git) <br>
To get all the parameters you need, you need <br>
`pred_smpl_params (global_orient, body_pose, betas), ` <br> 
`scaled_focal_length, ` <br> 
`pred_cam_t_full` from 4D-Humans, <br>
and <br>
`pred_mano_params (global_orient, hand_pose, betas), ` <br> 
`is_right, ` <br> 
`scaled_focal_length, ` <br> 
`pred_cam_t_full` from HAMER. <br>
You can also refer to the code i provided in demo folder. 

Put all the data into following folder: 
```bash
root_folder
|
|--data_folder
|  |
|  |--body
|  |  |
|  |  |--*.json
|  |  |
|  |  |--···
|  |
|  |--hand
|  |  |
|  |  |--*.json
|  |  |
|  |  |--···
```


## Environment
```bash
conda env create -f environment.yml
```

Or you can refer to `requirements.txt`. Generally you may need following packages: `numpy, torch, open3d, smplx, scipy, chumpy`, and I recommend the python version should be `3.10`.

Besides, you need to download [SMPLX](https://smpl-x.is.tue.mpg.de) and [MANO](https://mano.is.tue.mpg.de) model from official websites. Put them into following folders: 
```bash
root_folder
|
|--models
|  |
|  |--mano
|  |  |
|  |  |--MANO_LEFT.pkl
|  |  |
|  |  |--MANO_RIGHT.pkl
|  |
|  |--smplx
|  |  |
|  |  |--SMPLX_FEMALE.npz
|  |  |
|  |  |--SMPLX_FEMALE.pkl
|  |  |
|  |  |--SMPLX_MALE.npz
|  |  |
|  |  |--···
```


## Run Demo Code
```bash
conda activate smplx
python align_hand2body.py
```

## Outputs


## License
MIT License © 2025 Yuxuan Zhang

# Align MANO 2 SMPLX


## Table of Content
- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Run Demo Code](#run-demo-code)
- [Outputs](#outputs)
- [License](#license)


## Data Preparation

**Recording** <br>
We use `ORBBEC Femto Bolt` RGBD Camera to record RGB and depth data, using the python version SDK [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk.git). The data specifications are `1920x1080, 30fps`. <br>
For the basic recording with multi devices and to get RGB and depth images, you can refer to the code i provide at `demo/pyorbbecsdk_demo/`. For more recording or visualization codes, you can refer to my own [pyorbbecsdk](https://github.com/YuxuanZhang271/pyorbbecsdk.git) repo. 

**Tracking** <br>
Body Tracking: [4D-Humans](https://github.com/shubham-goel/4D-Humans.git) <br>
Hand Tracking: [HAMER](https://github.com/geopavlakos/hamer.git) <br>
You can also refer to the code i provided at `demo/4dHuman_demo.py` and `demo/hamer_demo.py`. 

To get all the parameters you need, you need <br>
`pred_smpl_params (global_orient, body_pose, betas), ` <br> 
`scaled_focal_length, ` <br> 
`pred_cam_t_full` from 4D-Humans, <br>

and <br>
`pred_mano_params (global_orient, hand_pose, betas), ` <br> 
`is_right, ` <br> 
`scaled_focal_length, ` <br> 
`pred_cam_t_full` from HAMER. <br>

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

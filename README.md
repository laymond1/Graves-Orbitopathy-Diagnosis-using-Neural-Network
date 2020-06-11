# Graves-Orbitopathy-Diagnosis-using-Neural-Network
This repository shows how to preprocss data and train/test model from CT data using Tensorflow and Opencv.

## [Requirements]
* ### tensorflow-gpu
* ### keras 
* ### opencv-python
* ### scikit-learn

## [Data Preprocessing]
We cut unnecessary part of the image, extract the muscles and fat part using HU-Filtering and Normalize them.

``` bash
  zoom_resize.py --data_path '../data/' --size 128 --out_path '../output/'
```

## [Training & Testing]
1. ### Training
    - #### Train BinaryClass Model
    ``` bash
    python train_binaryclass.py --data_path '../data/S_C/' \
    --dataset 'S_C' \
    --model 'axcosa' \
    --iteration 1 \
    --result_path '../results/' \
    --check_path '../saved_model/'
    ```
    
    - #### Train MultiClass Model
    ``` bash
    python train_multiclass.py --data_path '../data/S_M_C/' \
    --dataset 'S_M_C' \
    --model 'axcosa' \
    --iteration 1 \
    --result_path '../results/' \
    --check_path '../saved_model/'
    ```

2. ### Testing
    ### Pre-trained Models
    | Dataset | Model |
    |---|:---:|
    | S_C |  axcosa |
    | S_M |  axsa |
    | M_C |  axcosa |
    | S_M_C | axcosa |

    You can load the pre-trained models according to your load_path.
    
    - #### Test BinaryClass Model
    ``` bash
    python test_binaryclass.py --data_path '../data/S_C/' \
    --model 'axcosa' \
    --load_path '../saved_model/S_C/axcosa/1/'
    ```

    - #### Test MultiClass Model
    ``` bash
    python test_multiclass.py --data_path '../data/S_M_C/' \
    --model 'axcosa' \
    --load_path '../saved_model/S_M_C/axcosa/1/'
    ```

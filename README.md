## **ICCPS 25 paper: Uncertainty Quantification for Physics-Informed Traffic Graph Networks**
Tianshu Bao, Xiaoou Liu, Meiyi Ma, Taylor T. Johnson, Hua Wei


### Requirements:
torch \
scipy>=0.19.0 \
numpy>=1.12.1 \
pandas>=0.19.2 \
pyyaml==5.3.0 \
statsmodels \
tensorflow>=1.3.0 \
tables \
future 


### Ways to reproduce the results: 
- install the package locally
- docker
  
Instructions for docker user:
- Activate Docker on your local machine. 
- Run the command: ``` docker image pull tianshubao/uqtraffic:iccps ``` 
- Run the command: ``` docker run -it tianshubao/uqtraffic:iccps ``` to start the container.  

The docker image contains 4 models with datasets. We didn't provide all the models due to space limits. Each model must be assigned the whole dataset to make them runnable. You can manually put the dataset into each model and test them. The docker image does not contain GPU infrastructure. To run them on GPU, you will need to install nvidia-docker. 

### System Requirements: 
The host platform we used to prepare the docker image is a Dell Precision 5680 with Windows 11 Pro 64-bit OS, i7-13800H CPU, and NVIDIA RTX 3500 Ada GPU.

### Command: 
```cd ANY_UQ_MODEL```            # you can replace ANY_UQ_MODEL by any folder in the repo,  \
```python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml```    (Table 1, 2)

### Dataset: 
https://drive.google.com/drive/folders/1s1NaJ2DNgQWQr-p7i0586fjJsGidrvEU?usp=drive_link

### Zenodo: 
https://zenodo.org/records/14783003 \
DOI: 10.5281/zenodo.14783003

### Figshare datasets:
https://figshare.com/articles/dataset/METR-LA/28347830 \
https://figshare.com/articles/dataset/PEMSD8/28347662

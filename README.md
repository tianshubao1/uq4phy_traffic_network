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
future \
jupyter \
matplotlib


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
```python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml```    #by running this command every time, you will generate each row of Table 1, 2

### Results for plots: ###
The instructions for generating the figures are listed below: \
```cd plot``` \
We use jupyter notebook which will open a HTML page, please use Chrome or Web Explorer to open it.
| Artifact | Result Location | Description|
| -------- | -------- | -------- |
| **Figure 3:**    | `jupyter notebook maemis_model_seperate.ipynb` | Ablation study for MAEMIS-based methods in PEMSD8 datasets.|
| **Figure 4:**    | `jupyter notebook quantile_model_seperate_PEMSD8.ipynb` | Ablation study for Quantile methods in PEMSD8 datasets.|
| **Figure 5, 6:**   | `jupyter notebook plot_PEMSD8_regularization.ipynb` | GPDE+Quantile+Phy, GPDE+MAEMIS+Phy model error distribution with various 𝜆 values.|
| **Figure 7:**    | `results/figure_7.txt` | The test loss for model convergence of quantile-based method.|
| **Figure 8:**    | `results/figure_8.txt` | The predictions of three Quantile-based models.|
| **Figure 9:**    | `results/figure_9.txt` | The traffic speed predictions of three MAEMIS-based model.|
| **Figure 10:**    | `results/figure_10.txt` | The predictions of three MAEMIS-based models.|

### Dataset: 
https://drive.google.com/drive/folders/1s1NaJ2DNgQWQr-p7i0586fjJsGidrvEU?usp=drive_link

### Zenodo: 
https://zenodo.org/records/14783003 \
DOI: 10.5281/zenodo.14783003

### Figshare datasets:
https://figshare.com/articles/dataset/METR-LA/28347830 \
https://figshare.com/articles/dataset/PEMSD8/28347662

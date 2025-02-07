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

### Generating Plots: ###
The instructions for generating the figures are listed below: \
```cd plot``` \
We use Jupyter to convert the .ipynb file to .py file and then we run this python file to generate the figures.
|  | Command | Description|
| -------- | -------- | -------- |
| **Figure 3:**    | `jupyter nbconvert --to script maemis_model_seperate.ipynb` `python maemis_model_seperate.py` | Ablation study for MAEMIS-based methods in PEMSD8 datasets.|
| **Figure 4:**    | `jupyter nbconvert --to script quantile_model_seperate_PEMSD8.ipynb` `python quantile_model_seperate_PEMSD8.py` | Ablation study for Quantile methods in PEMSD8 datasets.|
| **Figure 5, 6:**   | `jupyter nbconvert --to script plot_PEMSD8_regularization.ipynb` `python plot_PEMSD8_regularization.py`| GPDE+Quantile+Phy, GPDE+MAEMIS+Phy model error distribution with various 𝜆 values.|
| **Figure 7:**    | `jupyter nbconvert --to script quantile_model_seperate.ipynb` `python quantile_model_seperate.py`| The test loss for model convergence of quantile-based method.|
| **Figure 8:**    | `jupyter nbconvert --to script plot_PEMSD8.ipynb` `python plot_PEMSD8.py`| The predictions of three Quantile-based models.|
| **Figure 9:**    | `jupyter nbconvert --to script plot_PEMSD8_maemis.ipynb` `python plot_PEMSD8_maemis.py`| The traffic speed predictions of three MAEMIS-based model.|
| **Figure 10:**   | `jupyter nbconvert --to script plot_METR-LA_maemis.ipynb` `python plot_METR-LA_maemis.py`| The predictions of three MAEMIS-based models.| \


The above cmd would generate the figures in the folder `/ICCPS25_repo/plot/outputs`. The next step is to copy these figures out. \
```docker cp big-container:/ICCPS25_repo/plot/outputs ./results``` 
This cmd copies the folder `/ICCPS25_repo/plot/output` to your local directory. Please ensure that the container is on while you copy the files.

### Dataset: 
https://drive.google.com/drive/folders/1s1NaJ2DNgQWQr-p7i0586fjJsGidrvEU?usp=drive_link

### Zenodo: 
https://zenodo.org/records/14783003 \
DOI: 10.5281/zenodo.14783003

### Figshare datasets:
https://figshare.com/articles/dataset/METR-LA/28347830 \
https://figshare.com/articles/dataset/PEMSD8/28347662

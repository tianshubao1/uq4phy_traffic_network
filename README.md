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
- Run the command: ``` docker image pull tianshubao/uqtraffic:iccpsv1 ``` 
- Run the command: ``` docker run --name big-container -it tianshubao/uqtraffic:iccpsv1 ``` to start the container.  The container's name is `big-container`.

The docker image contains 24 models and codes for plots with all METR-LA and PEMSD8 datasets. The docker image does not include GPU infrastructure. To run them on GPU, you will need to install nvidia-docker which may not be easy. So in the docker, reproducing the training steps of the 50 epochs takes a few hours for each model. The codes for plots can finish quickly since they use existing results.

### System Requirements: 
The host platform is a Dell Precision 5680 with Windows 11 Pro 64-bit OS, i7-13800H CPU, and NVIDIA RTX 3500 Ada GPU.

### Command for Training: 
```cd ANY_UQ_MODEL```            # you can replace ANY_UQ_MODEL by any folder in the repo,  \
```python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml```    
By running this command every time, you will generate each row of Table 1, 2. The cmd will print the MAE, MIS, MSE, RMSE for validation datasets and test datasets for each epoch. The output contains 12-step future predictions, and each step represents a 5-minute stride. So, it outputs the predictions for the next 5 min - 60 min and the metrics measure the average of them. When we calculate the metrics for 15 min, 30 min, and 60 min, we extract the corresponding values from the output and calculate these metrics separately.

For example, you can do the following for `quantile` model using METR-LA datasets.

|          | Command | Description|
| -------- | -------- | -------- |
| Quantile    | `cd quantile_model` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml` | Start training quantile_model using METR-LA datasets|
| GPDE+Quantile   | `cd gpde_quantile_model` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml`| Start training gpde_quantile_model using METR-LA datasets|
| GPDE+Quantile+Phy    | `cd gpde_quantile_phy_model` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml` | Start training gpde_quantile_phy_model using METR-LA datasets|

These cmds may take a few hours for each one to finish.

You can do the following for `maemis` model using METR-LA datasets.

|          | Command | Description|
| -------- | -------- | -------- |
| MAEMIS    | `cd maemis_model` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml` | Start training maemis_model using METR-LA datasets|
| GPDE+MAEMIS   | `cd gpde_maemis_model` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml`| Start training gpde_maemis_model using METR-LA datasets|
| GPDE+MAEMIS+Phy    | `cd gpde_maemis_phy_model` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml` | Start training gpde_maemis_phy_model using METR-LA datasets|



For `gpde_quantile_phy_model` using PEMSD8 dataset in Table 2, you can follow the steps below:
|          | Command | Description|
| -------- | -------- | -------- |
| 𝜆=0.01:    | `cd gpde_quantile_phy_model_PEMSD8_0.01` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml` | Start training model GPDE+Quantile+Phy with 𝜆=0.01|
| 𝜆=0.005    | `cd gpde_quantile_phy_model_PEMSD8_0.005` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml`| Start training model GPDE+Quantile+Phy with 𝜆=0.005|
| 𝜆=0.002    | `cd gpde_quantile_phy_model_PEMSD8` `python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml` | Start training model GPDE+Quantile+Phy with 𝜆=0.002|




### Plots: ###
The instructions for generating the figures are listed below: \
```cd plot``` \
We use Jupyter to convert the .ipynb file to .py file and then we run this python file to generate the figures.
|          | Command | Description|
| -------- | -------- | -------- |
| **Figure 3:**    | ```jupyter nbconvert --to script maemis_model_seperate.ipynb``` ```python maemis_model_seperate.py``` | Ablation study for MAEMIS-based methods in PEMSD8 datasets.|
| **Figure 4:**    | ```jupyter nbconvert --to script quantile_model_seperate_PEMSD8.ipynb``` ```python quantile_model_seperate_PEMSD8.py``` | Ablation study for Quantile methods in PEMSD8 datasets.|
| **Figure 5, 6:**   | `jupyter nbconvert --to script plot_PEMSD8_regularization.ipynb` `python plot_PEMSD8_regularization.py`| GPDE+Quantile+Phy, GPDE+MAEMIS+Phy model error distribution with various 𝜆 values.|
| **Figure 7:**    | `jupyter nbconvert --to script quantile_model_seperate.ipynb` `python quantile_model_seperate.py`| The test loss for model convergence of quantile-based method.|
| **Figure 8:**    | ```jupyter nbconvert --to script plot_PEMSD8.ipynb``` ```python plot_PEMSD8.py```| The predictions of three Quantile-based models.|
| **Figure 9:**    | `jupyter nbconvert --to script plot_PEMSD8_maemis.ipynb` `python plot_PEMSD8_maemis.py`| The traffic speed predictions of three MAEMIS-based model.|
| **Figure 10:**   | `jupyter nbconvert --to script plot_METR-LA_maemis.ipynb` `python plot_METR-LA_maemis.py`| The predictions of three MAEMIS-based models.| \

The above cmds run fast and generate the figures in the folder `/ICCPS25_repo/plot/outputs`. The next step is to copy these figures out by using: \
```docker cp big-container:/ICCPS25_repo/plot/outputs ./``` \
This cmd copies the folder `/ICCPS25_repo/plot/output` to your local directory. Please ensure that the container `big-container` is on while you copy the files.

### Zenodo: 
https://zenodo.org/records/14783003 \
DOI: 10.5281/zenodo.14783003

### Figshare datasets:
https://figshare.com/articles/dataset/METR-LA/28347830 \
https://figshare.com/articles/dataset/PEMSD8/28347662

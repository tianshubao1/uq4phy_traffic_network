# **ICCPS 25 paper: Uncertainty Quantification for Physics-Informed Traffic Graph Networks**

requirements:\
torch \
scipy>=0.19.0 \
numpy>=1.12.1 \
pandas>=0.19.2 \
pyyaml==5.3.0 \
statsmodels \
tensorflow>=1.3.0 \
tables \
future 

Ways to reproduce the results in the paper: \

Activate Docker on your local machine. \
Run the command: \textbf{docker image pull tianshubao/uqtraffic:iccps} \
Run the command: \textbf{docker run -it tianshubao/uqtraffic:iccps} to start the container. \ 
The docker image contains 4 models with datasets. We didn't provide all the models due to space limits. Each model must be assigned the whole dataset to make them runnable. You can manually put the dataset into each model and test them. The docker image does not contain GPU infrastructure. To run them on GPU, you will need to install nvidia-docker. 


command: \
cd ANY_UQ_MODEL            # you can replace ANY_UQ_MODEL by any folder in the repo,  \
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml  #Table 1

dataset: \
https://drive.google.com/drive/folders/1s1NaJ2DNgQWQr-p7i0586fjJsGidrvEU?usp=drive_link

Zenodo: \
https://zenodo.org/records/14783003 \
DOI: 10.5281/zenodo.14783003



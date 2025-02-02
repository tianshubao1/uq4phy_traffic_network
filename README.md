requirements:

torch \
scipy>=0.19.0 \
numpy>=1.12.1 \
pandas>=0.19.2 \
pyyaml==5.3.0 \
statsmodels \
tensorflow>=1.3.0 \
tables \
future 

cd ANY_UQ_MODEL  #you can replace ANY_UQ_MODEL by yourself \
python dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml


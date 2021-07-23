# Embedding Normalization

* The pytorch implementation for "Embedding Normalization: Significance Preserving Feature Normalization for Click-Through Rate Prediction" submitted to CIKM'21.
* Codes are forked from [pytorch-fm](https://rixwew.github.io/pytorch-fm/) and slightly modified for our purpose.

## Environments

* Install
```
conda create -n <env> python=3.7
conda activate <env>
conda install -c pytorch torchvision cudatoolkit=10.0 pytorch
pip install -r requirements.txt
```

# How to run

```
python -m fm.trainer
python -m fm.trainer --model-name=fm-en
python -m fm.trainer --model-name=ffm-ln
python -m fm.trainer --model-name=afi-bn
```

## Licence

MIT

# Multi-Granularity for Temporal Knowledge Graph Reasoning


### Environment variables & dependencies
```
conda create -n mghgn python=3.8

conda activate mghgn

pip install -r requirement.txt
```

### Download and Process data
The dataset files can be found in the project of baseline[(RE-GCN)](https://github.com/Lee-zix/RE-GCN/blob/master/data-release.tar.gz).

First, unzip and unpack the data files,
```
tar -zxvf data-release.tar.gz
```

#### Process history data

For all the datasets, the following command can be used to get the history of their entities.
```
cd src
python get_history.py --dataset ICEWS14s
```

### Training
Then the following commands can be used to train the offline models.

Train models with the maximum length k.
```
cd src
python main.py --dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction --gpu 1 -d ICEWS14s --train-history-len k --test -1  --ft_lr=0.001 --norm_weight 1 --alpha 0.5 --beta 0.5
```


### Evaluate the offline models
To generate the evaluation results of a offline model, set the `--test` to 0 and the `--train-history-len` to k (k is the maximum length of history) in the commands above. 

For example
```
python main.py --dilate-len 1 --n-epochs 30 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm  --entity-prediction --gpu 1 -d ICEWS14s --train-history-len 10 --test 0  --ft_lr=0.001 --norm_weight 1 --alpha 0.5 --beta 0.5
```

### Change the hyperparameters
To get the optimal result reported in the paper, change the hyperparameters and other experiment set up according to Section 4 in the paper. 

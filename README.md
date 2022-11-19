# seq-to-seq-auto-phoneme-recognition
For the project: Sequence-to-Sequence Frame-Level Phoneme Recognition, as part of homework 3 under the course 11-785: Intro to Deep Learning @ CMU 2022 Fall.


**Author**: Siyu Chen (schen4)

**Wandb Project**: [Click Here](https://wandb.ai/astromsoc/785-hw2/overview)

**Github Repo**: [Click Here](https://github.com/Astromsoc/seq-to-seq-auto-phoneme-recognition) (will be made public after the end of this semester, as is required) 



### Repository Structure
```
.
├── LICENSE
├── README.md
├── clear.sh
├── configs
│   └── sample_config.yml
├── data
│   ├── __init__.py
│   └── extract_mini.py
├── imgs
│   ├── sample_cosine_linearwarmup.png
│   └── sample_cosine_multistage.png
├── run.sh
├── setup.sh
└── src
    ├── constants.py
    ├── infer.py
    ├── models.py
    ├── train.py
    └── utils.py

5 directories, 16 files
```

**Note**: Remember to install the `ctcdecode` package from the appointed github repo as is specified in `setup.sh`. It's necessary to preinstall relevant c/c++ libraries, so make sure if they exist in the base environment, or else the installation will easily fail.


### Best Model Architecture
The best result I've obtained on the Kaggle leaderboard is a Levenshtein distance of `4.09064` (on the public sector). It is obtained by training the following network for 40 epochs (20 in exp-15, and another finetuning 20, resumed after checkpoint, in exp-17).

```
                           Totals 
Total params           22.322217M 
Trainable params       22.322217M 
Non-trainable params          0.0 
Mult-Adds             2.16736768G 
======================================================================================================= 
Model Summary: 
                                                 Kernel Shape     Output Shape     Params     Mult-Adds 
Layer 
0_feat_ext.shrinks.0.Conv1d_0                   [15, 1024, 7]  [64, 1024, 788]   108544.0  8.472576e+07 
1_feat_ext.shrinks.0.BatchNorm1d_1                     [1024]  [64, 1024, 788]     2048.0  1.024000e+03 
2_feat_ext.shrinks.0.GELU_2                                 -  [64, 1024, 788]        NaN           NaN 
3_feat_ext.shrinks.1.Conv1d_0                 [1024, 1024, 5]  [64, 1024, 394]  5243904.0  2.065695e+09 
4_feat_ext.shrinks.1.BatchNorm1d_1                     [1024]  [64, 1024, 394]     2048.0  1.024000e+03 
5_feat_ext.shrinks.1.GELU_2                                 -  [64, 1024, 394]        NaN           NaN 
6_lstm_stack.0.lstms.0.LSTM_vanilla                         -    [19712, 1024]  6299648.0  6.291456e+06 
7_lstm_stack.0.lstms.0.LockedDropout_dropout                -  [64, 394, 1024]        NaN           NaN 
8_lstm_stack.0.lstms.1.LSTM_vanilla                         -    [19712, 1024]  6299648.0  6.291456e+06 
9_lstm_stack.0.lstms.1.LockedDropout_dropout                -  [64, 394, 1024]        NaN           NaN 
10_cls.linears.Linear_0                          [1024, 4096]  [64, 394, 4096]  4198400.0  4.194304e+06 
11_cls.linears.Dropout_1                                    -  [64, 394, 4096]        NaN           NaN 
12_cls.linears.GELU_2                                       -  [64, 394, 4096]        NaN           NaN 
13_cls.linears.Linear_3                            [4096, 41]    [64, 394, 41]   167977.0  1.679360e+05 
14_cls.LogSoftmax_logsoftmax                                -    [64, 394, 41]        NaN           NaN 
```

Corresponding hyperparameters include:
```
Optimizer:
    AdamW(lr=0.002)
Scheduler:
    [initial 20 epochs]
        CosineAnnealingWithLinearWarmUp(warmup_epochs=0.2, stages=3)
    [another 20 epochs for finetuning]
        ReduceLROnPlateau(patience=3, mode='min', factor=0.5)
```
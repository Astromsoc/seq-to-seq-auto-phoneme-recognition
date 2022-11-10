"""
    utility classes & functions 
"""
import os
import math
import numpy as np
from tqdm import tqdm
from Levenshtein import distance

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader



class cfgClass(object):
    """
        Convert configuration dictionary into a simple class instance.
    """
    def __init__(self, cfg_dict: dict):
        # initial buildup
        self.__dict__.update(cfg_dict)
        for k, v in self.__dict__.items():
            if not k.endswith('configs') and isinstance(v, dict):
                self.__dict__.update({k: cfgClass(v)})



class datasetTrainDev(Dataset):
    """
        Dataloader for training and dev sets: both features & labels
    """
    def __init__(
        self, 
        mfccDir: str=None, transDir: str=None, stdDir: str=None, 
        labelToIdx: dict=None, keepTags: bool=False
    ):
        # using default structure (./mfcc + ./transcript/raw)
        if stdDir:
            mfccDir = f"{stdDir}/mfcc"
            transDir = f"{stdDir}/transcript/raw"
        # bookkeeping
        self.mfccDir = mfccDir
        self.transDir = transDir
        self.labelToIdx = labelToIdx
        self.keepTags = keepTags
        
        # load all filenames
        mfccFNs = sorted([
            f"{mfccDir}/{f}" for f in os.listdir(mfccDir) if f.endswith('.npy')
        ])
        transFNs = sorted([
            f"{transDir}/{f}" for f in os.listdir(transDir) if f.endswith('.npy')
        ])

        # whether to keep <sos> / <eos>
        if self.keepTags: 
            l, r = 1, -1

        # load files
        self.mfccs = [
            torch.from_numpy(np.load(f)) 
            for f in tqdm(mfccFNs, leave=False, desc='loading mfccs...') if f.endswith('.npy')
        ]
        self.transcripts = [
            torch.tensor([self.labelToIdx[p] for p in (np.load(f) if self.keepTags else np.load(f)[l: r])]) 
            for f in tqdm(transFNs, leave=False, desc='loading transcripts...') if f.endswith('.npy')
        ]

        # dataset size
        self.size = len(self.mfccs)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, index):
        return self.mfccs[index], self.transcripts[index]



class datasetTest(Dataset):
    """
        Dataset for test set: only features
    """
    def __init__(
        self, 
        mfccDir: str=None
    ):
        # bookkeeping
        self.mfccDir = mfccDir

        # load all filenames
        mfccFNs = sorted([
            f"{mfccDir}/{f}" for f in os.listdir(mfccDir) if f.endswith('.npy')
        ])
        # load files
        self.mfccs = [
            torch.from_numpy(np.load(f)) 
            for f in tqdm(mfccFNs, leave=False, desc='loading mfccs...') 
            if f.endswith('.npy')
        ]
        # dataset size
        self.size = len(self.mfccs)
    

    def __len__(self):
        return self.size
    

    def __getitem__(self, index):
        return self.mfccs[index]



def collateTrainDev(
        batch, mfccPadding=0, transPadding=0
    ):
    """
        Collate function for training and dev sets, 4 returns
    """
    mfccs = [u[0] for u in batch]
    transcripts = [u[1] for u in batch]

    # obtain original lengths for both mfccs & transcripts
    mfccLens = torch.tensor([len(m) for m in mfccs])
    transcriptLens = torch.tensor([len(t) for t in transcripts])

    # pad both mfccs & transcripts
    mfccs = pad_sequence(
        mfccs, batch_first=True, padding_value=mfccPadding
    )
    transcripts = pad_sequence(
        transcripts, batch_first=True, padding_value=transPadding
    )
    return mfccs, transcripts, mfccLens, transcriptLens



def collateTest(
        batch, mfccPadding=0
    ):
    """
        Collate function for test set: 2 returns
    """
    mfccs = batch
    # obtain original lengths
    mfccLens = torch.tensor([len(m) for m in mfccs])
    # pad 
    mfccs = pad_sequence(
        mfccs, batch_first=True, padding_value=mfccPadding
    )
    return mfccs, mfccLens
    


def compute_levenshtein(h, y, lh, ly, decoder, LABELS):
    # decode the output (taking the best output from beam search)
    # h <- (batch, seq_len, n_labels)
    beamResults, _, _, outLens = decoder.decode(h, lh)
    totalDist = 0
    batchSize = len(beamResults)
    for b in range(batchSize):
        predStr = ''.join(LABELS[l] for l in beamResults[b, 0, :outLens[b, 0]])
        trueStr = ''.join(LABELS[l] for l in y[b, :ly[b]])
        totalDist += distance(predStr, trueStr)
    return totalDist



def cosine_linearwarmup_scheduler(
        totalEpochs: int=50, batchesPerEpoch: int=200, lr_decay: float=0.9999,
        init_lr: float=1e-3, min_lr: float=1e-5, warmupEpochs: int=5
    ):
    # total batches
    totalBatches = totalEpochs * batchesPerEpoch
    lr_list = np.zeros((totalBatches, ))

    # linear warmup
    warmupBatches = batchesPerEpoch * warmupEpochs
    lr_list[:warmupBatches] = np.linspace(0, init_lr, warmupBatches)

    # cosine w/ shrinking amplitudes
    cosineBatches = totalBatches - warmupBatches
    lr_list[warmupBatches:] = np.array([
        min_lr + 0.5 * (init_lr - min_lr) * (lr_decay ** b) * (
            1 + math.cos(2 * math.pi * b / cosineBatches)
        ) for b in range(cosineBatches)
    ])
    return lr_list
            


"""
    The main function below is for local test only
"""
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    lr_list = cosine_linearwarmup_scheduler()
    plt.title("Sample Graph for Learning Rate Schedule")
    plt.plot(lr_list)
    plt.xlabel('Batches')
    plt.ylabel('learning rate')
    plt.savefig('./imgs/sample_cosine_linearwarmup.png', dpi=128)
    plt.show()
    # you can use the plot to explore what decay rate to set for lr

    from data.phonetics import *
    p2i = {p: i for i, p in enumerate(CMUdict)}

    trnDir = '../autodl-tmp/mini-train'
    trnDataset = datasetTrainDev(
        stdDir=trnDir,
        labelToIdx=p2i,
        keepTags=False
    )
    trnLoader = DataLoader(
        trnDataset,
        shuffle=True,
        num_workers=4,
        batch_size=10,
        collate_fn=collateTrainDev,
        pin_memory=True
    )

    devDir = '../autodl-tmp/mini-dev'
    devDataset = datasetTrainDev(
        stdDir=devDir,
        labelToIdx=p2i,
        keepTags=False
    )
    devLoader = DataLoader(
        devDataset,
        shuffle=False,
        num_workers=4,
        batch_size=10,
        collate_fn=collateTrainDev
    )

    tstDir = '../autodl-tmp/mini-test/mfcc'
    tstDataset = datasetTest(
        mfccDir=tstDir
    )
    tstLoader = DataLoader(
        tstDataset,
        shuffle=False,
        num_workers=4,
        batch_size=10,
        collate_fn=collateTest
    )

    for batch in trnLoader:
        x, y, lx, ly = batch
        print("Training dataset and loader: ", x.shape, y.shape, lx.shape, ly.shape)
        break
    for batch in devLoader:
        x, y, lx, ly = batch
        print("Dev dataset and loader: ", x.shape, y.shape, lx.shape, ly.shape)
        break
    for batch in tstLoader:
        x, lx = batch
        print("Test dataset and loader: ", x.shape, lx.shape)
        break
    

    
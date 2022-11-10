"""
    Driver script to train the customized model(s).
"""
import time
import yaml
import wandb
import shutil
import argparse


from src.utils import *
from src.models import *
from data.phonetics import *
from ctcdecode import CTCBeamDecoder



def unit_train(
        model, trn_loader, criterion, optimizer, scaler=None, lr_list: list=None,
        transforms=None, use_wandb: bool=False, device: str='cuda'
    ):

    model.train()
    batch_bar = tqdm(
        total=len(trn_loader), dynamic_ncols=True, leave=False, desc='training...'
    ) 
    # true loss computation
    trn_loss, trn_cnt = 0, 0 
    
    for batch in trn_loader:

        optimizer.zero_grad()
        # set the lr rate manually if provided the array
        if lr_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[batch]

        x, y, lx, ly = batch
        # augmentations
        if transforms:
            x, lx = transforms(x, lx)

        x = x.to(device)
        y = y.to(device)
        cnt = len(x)

        if scaler and 'cuda' in device:
            with torch.cuda.amp.autocast():
                h, lh = model(x, lx)
                h = h.permute((1, 0, 2))
                loss = criterion(h, y, lh, ly)
        
        dev_loss += loss.item() * cnt
        dev_cnt += cnt

        batch_bar.set_postfix(
            trn_loss=f"{trn_loss / trn_cnt:.6f}", lr=optimizer.param_groups[0]['lr']
        )
        batch_bar.update()

        # add to wandb log
        if use_wandb:
            wandb.log({'microavg-trn-loss': trn_loss / trn_cnt})
    
    batch_bar.close()
    if use_wandb:
        wandb.log({
            'epoch-trn-loss-running-avg': trn_loss / trn_cnt,
            'learning-rate': optimizer.param_groups[0]['lr']
        })
    return trn_loss / trn_cnt



def unit_eval(
        model, dev_loader, criterion, decoder, LABELS, 
        scaler=None, use_wandb: bool=False, device: str='cuda', comp_dist: bool=True
    ):
    """
        Evaluation function per epoch.
            Compute dev loss (must) and/or dev levenshtein distance(optional)
    """
    model.eval()
    batch_bar = tqdm(
        total=len(dev_loader), dynamic_ncols=True, leave=False, desc='training...'
    ) 
    # true loss & dist computation
    dev_loss, dev_dist, dev_cnt = 0, 0, 0

    with torch.inference_mode():
        for batch in dev_loader:
            x, y, lx, ly = batch
            x = x.to(device)
            y = y.to(device)
            cnt = len(x)

            if scaler and 'cuda' in device:
                with torch.cuda.amp.autocast():
                    h, lh = model(x, lx)
                    h = h.permute((1, 0, 2))
                    loss = criterion(h, y, lh, ly)
            if comp_dist:
                dist = compute_levenshtein(h, y, lh, ly, decoder, LABELS)
            
            dev_dist += dist
            dev_loss += loss.item() * cnt
            dev_cnt += cnt

            dev_loss_show = dev_loss / dev_cnt
            dev_dist_show = dev_dist / dev_cnt if comp_dist else -1
            batch_bar.set_postfix(
                dev_loss=f"{dev_loss_show:.6f}", dev_dist=f"{dev_dist_show:.6f}"
            )
            batch_bar.update()

    batch_bar.close()
    # add to wandb log
    if use_wandb:
        wandb.log({'epoch-dev-loss': dev_loss_show})
        if comp_dist:
            wandb.log({'epoch-dev-dist': dev_dist_show})
    return dev_loss_show, dev_dist_show




def main(args):
    # load configurations
    allcfgs = cfgClass(yaml.safe_load(open(args.config_file, 'r')))

    tgt_folder = time.strftime("%Y%m%d-%H%M%S")[2:]
    # init wandb proj if set to
    if allcfgs.wandb.use:
        wandb.init(**allcfgs.wandb.configs, config=allcfgs)
        tgt_folder = wandb.run.name
    # setup output filepath
    tgt_folder = f"{allcfgs.OUTPUT_DIR}/{tgt_folder}"
    # copy the configuration file there
    shutil.copy(args.config_file, f"{tgt_folder}/config.yml")


    # find the device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\n\nRunning on [{device}]...\n")

    # phonemes, labels & mapping
    PHONEMES = CMUdict
    # whether to skip sequence tags in labels & phonemes
    NUM_LABELS = len(PHONEMES) if allcfgs.keep_seq_tags else len(PHONEMES) - 2
    # add to the configuration dict
    allcfgs.model_configs.cls['num_labels'] = NUM_LABELS

    # truncate
    PHONEMES = PHONEMES[: NUM_LABELS]
    PNM2IDX = {p: i for i, p in enumerate(PHONEMES)}
    LABELS = ARPAbet[: NUM_LABELS]

    """
        load data & build data loaders
    """
    trnDataset = datasetTrainDev(
        stdDir=allcfgs.TRAIN_DATA_DIR,
        labelToIdx=PNM2IDX,
        keepTags=allcfgs.keepSeqTags
    )
    devDataset = datasetTrainDev(
        stdDir=allcfgs.DEV_DATA_DIR,
        labelToIdx=PNM2IDX,
        keepTags=allcfgs.keepSeqTags
    )
    trnLoader = DataLoader(
        trnDataset,
        batch_size=allcfgs.batchSize,
        num_workers=allcfgs.num_workers,
        collate_fn=collateTrainDev,
        shuffle=True,
        pin_memory=True
    )
    devLoader = DataLoader(
        devDataset,
        batch_size=allcfgs.batchSize,
        num_workers=allcfgs.num_workers,
        collate_fn=collateTrainDev
    )

    # model buildup
    model = OneForAll(**allcfgs.model_configs)

    # criterion, optimizer and scheduler
    criterion = nn.CTCLoss()
    optimizer = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }[allcfgs.optimizer.name](model.parameters(), allcfgs.optimizer.cfgs)
    # TODO: add official optimizers as well

    batches_per_epoch = len(trnLoader)
    lr_list = cosine_linearwarmup_scheduler(
        totalEpochs=allcfgs.epochs, batchesPerEpoch=batches_per_epoch,
        init_lr=allcfgs.optimizer.cfgs.lr, **allcfgs.scheduler_manual
    )
    scaler = torch.cuda.amp.GradScaler()

    # decoder for beam search
    decoder = CTCBeamDecoder(
        LABELS, log_probs_input=True,
        **allcfgs.decoder_configs
    )

    # keep track of best models
    min_dev_loss = float('inf')
    min_dev_dist = float('inf')
    # filepaths for best models
    min_loss_ckpt_fp = f"{tgt_folder}/min_loss.pt"
    min_dist_ckpt_fp = f"{tgt_folder}/min_dist.pt"

    # metrics tracking
    trn_losses = list()
    dev_losses = list()
    dev_dists = list()

    # start training
    for epoch in range(allcfgs.epochs):

        lr_slice = lr_list[epoch * batches_per_epoch: (epoch + 1) * batches_per_epoch]

        # train the model
        trn_loss = unit_train(
            model=model, trn_loader=trnLoader, criterion=criterion, 
            optimizer=optimizer, lr_list=lr_slice,
            scaler=scaler, use_wandb=allcfgs.wandb.use, device=device
        )
        # evaluate the model
        comp_dist = (epoch % allcfgs.comp_dist_int == 0) or (epoch == allcfgs.epochs - 1)
        dev_loss, dev_dist = unit_eval(
            model=model, dev_loader=devLoader, criterion=criterion,
            decoder=decoder, LABELS=LABELS, comp_dist=comp_dist,
            scaler=scaler, use_wandb=allcfgs.wandb.use, device=device
        )

        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            # saving checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': lr_list
            }, min_loss_ckpt_fp)
            wandb.log({'min_loss_new_saving': epoch})

        # local loss tracking
        trn_losses.append(trn_loss)
        dev_losses.append(dev_loss)

        if dev_dist != -1:
            dev_dists.append(dev_dist)
            if dev_dist < min_dev_dist:
                min_dev_dist = dev_dist
                # saving checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': lr_list
                }, min_dist_ckpt_fp)
                wandb.log({'min_dist_new_saving': epoch})

    # save the local tracking result
    np.save(
        f"{tgt_folder}/log.npy", {
            'epoch_train_losses': trn_losses,
            'epoch_dev_losses': dev_losses,
            'epoch_dev_dists': dev_dists,
            'lr_list': lr_list,
            'min_dev_loss': min_dev_loss,
            'min_dev_dist': min_dev_dist
        }
    )




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Driver script for model training.')
    parser.add_argument(
        '--config-file',
        '-c',
        default='./configs/sample_config.yml',
        type=str,
        help='Filepath of configuration yaml file to be read.'
    )

    args = parser.parse_args()
    main(args)
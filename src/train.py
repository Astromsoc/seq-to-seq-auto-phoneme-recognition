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
from src.constants import *
from ctcdecode import CTCBeamDecoder



def unit_train(
        model, trn_loader, criterion, optimizer,  epoch: int,
        scaler=None, lr_list: list=None, noise_level:float=0, 
        use_wandb: bool=False, device: str='cuda'
    ):

    batch_bar = tqdm(
        total=len(trn_loader), dynamic_ncols=True, leave=False, desc='training...'
    ) 
    # true loss computation
    trn_loss = 0 
    model.train()

    for b, batch in enumerate(trn_loader):

        optimizer.zero_grad()
        # set the lr rate manually if provided the array
        if lr_list is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[b]
        if use_wandb:
            wandb.log({'learning-rate': optimizer.param_groups[0]['lr']})

        x, y, lx, ly = batch

        x, y = x.to(device), y.to(device)
        # add random noise to input
        if noise_level != 0:
            x += torch.randn(x.size(), device=x.device) * noise_level

        if scaler and device.startswith('cuda'):
            with torch.cuda.amp.autocast():
                h, lh = model(x, lx)
                loss = criterion(h.permute((1, 0, 2)), y, lh, ly)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            h, lh = model(x, lx)
            loss = criterion(h.permute((1, 0, 2)), y, lh, ly)
            # backprop
            loss.backward()
            optimizer.step()
        
        trn_loss += loss.item() 
        trn_loss_show = trn_loss / (b + 1)
        batch_bar.set_postfix(
            trn_loss=f"{trn_loss_show:.6f}", lr=optimizer.param_groups[0]['lr']
        )
        batch_bar.update()
        # add to wandb log
        if use_wandb:
            wandb.log({'microavg-trn-loss': trn_loss_show})
    
    batch_bar.close()
    final_trn_loss = trn_loss / len(trn_loader)
    if use_wandb:
        wandb.log({'epoch-trn-loss-running-avg': final_trn_loss, 'epoch': epoch + 1})
    return final_trn_loss



def unit_eval(
        model, dev_loader, criterion, decoder, LABELS, epoch: int,
        scaler=None, use_wandb: bool=False, device: str='cuda', comp_dist: bool=True
    ):
    """
        Evaluation function per epoch.
            Compute dev loss (must) and/or dev levenshtein distance(optional)
    """
    model.eval()
    batch_bar = tqdm(
        total=len(dev_loader), dynamic_ncols=True, leave=False, desc='evaluating...'
    ) 
    # true loss & dist computation
    dev_loss, dev_dist = 0, 0

    for b, batch in enumerate(dev_loader):
        x, y, lx, ly = batch
        x, y = x.to(device), y.to(device)

        with torch.inference_mode():
            if scaler and device.startswith('cuda'):
                with torch.cuda.amp.autocast():
                    h, lh = model(x, lx)
                    loss = criterion(h.permute((1, 0, 2)), y, lh, ly)
            else:
                h, lh = model(x, lx)
                loss = criterion(h.permute((1, 0, 2)), y, lh, ly)

        if comp_dist:
            dist = compute_levenshtein(h, y, lh, ly, decoder, LABELS)  
            dev_dist += dist

        dev_loss += loss.item()
        dev_loss_show = dev_loss / (b + 1)
        dev_dist_show = dev_dist / (b + 1) if comp_dist else -1
        batch_bar.set_postfix(
            dev_loss=f"{dev_loss_show:.6f}", dev_dist=f"{dev_dist_show:.6f}"
        )
        batch_bar.update()
        # delete variables to free some space
        del x, y, h

    batch_bar.close()
    final_dev_loss = dev_loss / len(dev_loader)
    final_dev_dist = dev_dist / len(dev_loader) if comp_dist else -1
    # add to wandb log
    if use_wandb:
        wandb.log({'epoch-dev-loss': final_dev_loss, 'epoch': epoch + 1})
        if comp_dist:
            wandb.log({'epoch-dev-dist': final_dev_dist, 'epoch': epoch + 1})
    return final_dev_loss, final_dev_dist



def main(args):
    # load configurations
    trncfgs = cfgClass(yaml.safe_load(open(args.config_file, 'r')))

    # use previous configurations if funetuning
    if trncfgs.finetune.use:
        ckpt = torch.load(trncfgs.finetune.checkpoint)
        trncfgs.model = ckpt['model_configs']

    # fix random seeds
    torch.manual_seed(trncfgs.SEED)
    np.random.seed(trncfgs.SEED)

    # output experiment folder
    tgt_folder = time.strftime("%Y%m%d-%H%M%S")[2:]
    # init wandb proj if set to
    if trncfgs.wandb.use:
        wandb.init(**trncfgs.wandb.configs, config=trncfgs)
        tgt_folder = wandb.run.name
    # setup output filepath
    tgt_folder = f"{trncfgs.OUTPUT_DIR}/{tgt_folder}"
    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)
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
    if not trncfgs.finetune.use:
        # whether to skip sequence tags in labels & phonemes
        NUM_LABELS = len(PHONEMES) if trncfgs.keep_seq_tags else len(PHONEMES) - 2
        trncfgs.model.configs['cls_cfgs']['num_labels'] = NUM_LABELS
    else:
        # load previous settings by inferring from old configs
        NUM_LABELS = trncfgs.model.configs['cls_cfgs']['num_labels']

    # truncate
    PHONEMES = PHONEMES[: NUM_LABELS]
    PNM2IDX = {p: i for i, p in enumerate(PHONEMES)}
    LABELS = ARPAbet[: NUM_LABELS]

    """
        load data & build data loaders
    """
    trnDataset = datasetTrainDev(
        stdDir=trncfgs.TRAIN_DATA_DIR,
        labelToIdx=PNM2IDX,
        keepTags=trncfgs.keep_seq_tags
    )
    devDataset = datasetTrainDev(
        stdDir=trncfgs.DEV_DATA_DIR,
        labelToIdx=PNM2IDX,
        keepTags=trncfgs.keep_seq_tags
    )
    trnLoader = DataLoader(
        trnDataset,
        batch_size=trncfgs.batch_size,
        num_workers=trncfgs.num_workers,
        collate_fn=collateTrainDev,
        shuffle=True,
        pin_memory=True
    )
    devLoader = DataLoader(
        devDataset,
        batch_size=trncfgs.batch_size,
        num_workers=trncfgs.num_workers,
        collate_fn=collateTrainDev
    )
    print(f"\nA total of [{len(trnLoader)}] batches in training set, and [{len(devLoader)}] in dev set.\n")

    # model buildup
    model = {
        'one-for-all': OneForAll,
        'one-for-all-unlocked': OneForAllUnlocked,
        'knees-and-toes': KneesAndToes
    }[trncfgs.model.choice](**trncfgs.model.configs)
    model.to(device)

    # randomly take a batch for model summary
    model.eval()
    x, _, lx, _ = next(iter(trnLoader))
    with torch.inference_mode():
        print(f"\n\nModel Summary: \n{summary(model, x.to(device), lx)}\n\n")

    # criterion, optimizer and scheduler
    criterion = nn.CTCLoss()

    optimizer = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }[trncfgs.optimizer.name](model.parameters(), **trncfgs.optimizer.configs)
    
    # manual cosine annealing w/ linear warmup
    batches_per_epoch = len(trnLoader)
    lr_list = cosine_linearwarmup_scheduler(
        total_epochs=trncfgs.epochs, batches_per_epoch=batches_per_epoch,
        init_lr=trncfgs.optimizer.configs['lr'], **trncfgs.scheduler_manual.configs
    ) if trncfgs.scheduler_manual.use else None
    if trncfgs.scheduler_manual.use:
        # save plot of lr scheduler in target folder
        plot_lr_schedule(lr_list, tgt_folder)
    
    # other pytorch schedulers
    scheduler = {
        'reduce-on-plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'multistep': torch.optim.lr_scheduler.MultiStepLR
    }[trncfgs.scheduler.choice](
        optimizer, **trncfgs.scheduler.configs
    ) if trncfgs.scheduler.use else None

    # mixed precision training
    scaler = torch.cuda.amp.GradScaler() if trncfgs.use_mixed_precision else None

    # decoder for beam search
    decoder = CTCBeamDecoder(
        LABELS, log_probs_input=True,
        num_processes=trncfgs.num_workers,
        **trncfgs.decoder_configs
    )

    # load checkpoints if finetuning
    if trncfgs.finetune.use:
        # for model
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        # for optimizer
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # keep track of best models
    min_dev_loss = float('inf')
    min_dev_dist = float('inf')
    # filepaths for best models
    min_loss_ckpt_fp = f"{tgt_folder}/min_loss.pt"
    min_dist_ckpt_fp = f"{tgt_folder}/min_dist.pt"
    last_ckpt_fp = f"{tgt_folder}/last.pt"

    # metrics tracking
    trn_losses = list()
    dev_losses = list()
    dev_dists = list()

    # start training
    for epoch in range(trncfgs.epochs):
        print(f"\n\nRunning on Epoch [#{epoch + 1}/{trncfgs.epochs}] now...\n")
        
        # manual schedule update
        lr_slice = (
            lr_list[epoch * batches_per_epoch: (epoch + 1) * batches_per_epoch] 
            if lr_list is not None else None
        )

        # train the model
        trn_loss = unit_train(
            model=model, trn_loader=trnLoader, criterion=criterion, epoch=epoch,
            optimizer=optimizer, lr_list=lr_slice, noise_level=trncfgs.noise_level,
            scaler=scaler, use_wandb=trncfgs.wandb.use, device=device
        )
        # evaluate the model
        comp_dist = (epoch % trncfgs.comp_dist_int == 0) or (epoch == trncfgs.epochs - 1)
        dev_loss, dev_dist = unit_eval(
            model=model, dev_loader=devLoader, criterion=criterion,
            decoder=decoder, LABELS=LABELS, comp_dist=comp_dist, epoch=epoch,
            scaler=scaler, use_wandb=trncfgs.wandb.use, device=device
        )

        if dev_loss < min_dev_loss:
            min_dev_loss = dev_loss
            # saving checkpoint
            torch.save({
                'epoch': epoch,
                'model_configs': trncfgs.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr': lr_list
            }, min_loss_ckpt_fp)

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
                    'model_configs': trncfgs.model,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr': lr_list
                }, min_dist_ckpt_fp)
                if trncfgs.wandb.use:
                    wandb.log({'min_dist_new_saving': epoch})

        # step for pytorch scheduler
        if scheduler:
            if trncfgs.scheduler.choice == 'reduce-on-plateau':
                if trncfgs.scheduler.use_dist_plateau:
                    scheduler.step(dev_dist)
                else:
                    scheduler.step(dev_loss)
            else:
                scheduler.step()


    # saving the last checkpoint
    torch.save({
        'epoch': epoch,
        'model_configs': trncfgs.model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': lr_list
    }, last_ckpt_fp)

    # save the local tracking result
    np.save(
        f"{tgt_folder}/log.npy", {
            'epoch_train_losses': trn_losses,
            'epoch_dev_losses': dev_losses,
            'epoch_dev_dists': dev_dists,
            'lr_list': lr_list,
            'model_configs': trncfgs.model,
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
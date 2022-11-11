"""
    Driver script to infer the transcriptions for test files.
"""
import yaml
import argparse
import pandas as pd
from tqdm import tqdm

from src.utils import *
from src.models import *
from src.constants import *
from ctcdecode import CTCBeamDecoder


def infer_one_checkpoint(
        trncfgs, checkpoint_filepath, tst_loader, template_filepath,
        decoder, scaler, LABELS, device
    ):

    print(f"\n\nRunning inference on checkpoint [{checkpoint_filepath}]...\n")

    # reconstruct the model
    model = {
        'one-for-all': OneForAll,
        'knees-and-toes': KneesAndToes,
        'shoulder-knees-and-toes': ShoulderKneesAndToes
    }[trncfgs.model.choice](**trncfgs.model.configs)

    # load from checkpoint
    ckpt = torch.load(checkpoint_filepath)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    all_preds = list()

    # iterate over batches
    for b, batch in tqdm(enumerate(tst_loader), total=len(tst_loader)):
        x, lx = batch
        x = x.to(device)

        if device.startswith("cuda") and scaler is not None:
            with torch.cuda.amp.autocast():
                h, lh = model(x, lx)
        else:
            h, lh = model(x, lx)
        
        # obtain batch predictions
        batch_preds = generate_batch_predictions(h, lh, decoder, LABELS)
        all_preds.extend(batch_preds)

    # output csv filename: adapted from checkpoint name
    out_filepath = checkpoint_filepath.replace('.pt', '_pred.csv')
    # generate csv file
    raw_df = pd.read_csv(template_filepath)
    raw_df.label = all_preds
    raw_df.to_csv(out_filepath, index=False)
    
    return all_preds




def main(args):
    tstcfgs = cfgClass(yaml.safe_load(open(args.config_file, 'r')))
    exp_folder = tstcfgs.exp_folder

    # find the device
    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"\n\nRunning on [{device}]...\n")

    # load configs & model checkpoints from given experiment folder
    trncfgs = cfgClass(yaml.safe_load(open(f"{exp_folder}/config.yml", 'r')))

    # reconstruct label sets using training configs
    # phonemes, labels & mapping
    PHONEMES = CMUdict
    # whether to skip sequence tags in labels & phonemes
    NUM_LABELS = len(PHONEMES) if trncfgs.keep_seq_tags else len(PHONEMES) - 2
    # add to the configuration dict
    trncfgs.model.configs['cls_cfgs']['num_labels'] = NUM_LABELS

    # truncate
    PHONEMES = PHONEMES[: NUM_LABELS]
    PNM2IDX = {p: i for i, p in enumerate(PHONEMES)}
    LABELS = ARPAbet[: NUM_LABELS]

    """
        load data & build data loaders
    """
    tstDataset = datasetTest(
        mfccDir=tstcfgs.TEST_DATA_DIR
    )
    tstLoader = DataLoader(
        tstDataset,
        batch_size=tstcfgs.batch_size,
        num_workers=tstcfgs.num_workers,
        collate_fn=collateTest
    )
    print(f"\nA total of [{len(tstLoader)}] batches in test set.\n")

    # load the template for test answer generation
    test_template_filepath = trncfgs.TEST_DATA_DIR.replace('mfcc', 'transcript/random_submission.csv')

    # build scaler
    if device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()

    # build a decoder class for common use
    tst_decoder = CTCBeamDecoder(
        LABELS, log_probs_input=True,
        num_processes=tstcfgs.num_workers,
        **tstcfgs.test_decoder_configs
    )

    if tstcfgs.use_min_loss:
        _ = infer_one_checkpoint(
            trncfgs=trncfgs, checkpoint_filepath=f"{exp_folder}/min_loss.pt", 
            tst_loader=tstLoader, template_filepath=test_template_filepath,
            decoder=tst_decoder, scaler=scaler, LABELS=LABELS, device=device
        )

    if tstcfgs.use_min_dist:
        _ = infer_one_checkpoint(
            trncfgs=trncfgs, checkpoint_filepath=f"{exp_folder}/min_dist.pt", 
            tst_loader=tstLoader, template_filepath=test_template_filepath,
            decoder=tst_decoder, scaler=scaler, LABELS=LABELS, device=device
        )
    
    if tstcfgs.use_last:
        _ = infer_one_checkpoint(
            trncfgs=trncfgs, checkpoint_filepath=f"{exp_folder}/last.pt", 
            tst_loader=tstLoader, template_filepath=test_template_filepath,
            decoder=tst_decoder, scaler=scaler, LABELS=LABELS, device=device
        )
        





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Driver script for model inference.')

    parser.add_argument(
        '--config-file',
        '-c',
        default='./configs/sample_config.yml',
        type=str,
        help='Filepath of configuration yaml file to be read.'
    )
    args = parser.parse_args()

    main(args)
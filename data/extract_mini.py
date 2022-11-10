import os
import shutil
import argparse
import numpy as np


def main(args):
    """
        using the following folder structure by default
            - (src_folder)
                - mfcc
                    ...
                - transcript
                    - raw
                        ...
    """
    # source folder
    srcRoot = args.src_folder
    srcMFCCs = sorted([
        f'mfcc/{f}' for f in os.listdir(f'{srcRoot}/mfcc')
    ])
    if not args.feat_only:
        srcTranscripts = sorted([
            f'transcript/raw/{f}' for f in os.listdir(f'{srcRoot}/transcript/raw')
        ])
        assert len(srcMFCCs) == len(srcTranscripts)

    # target folder
    tgtRoot = args.tgt_folder
    if not os.path.exists(tgtRoot):
        if not args.feat_only:
            os.makedirs(f'{tgtRoot}/transcript/raw')
        os.makedirs(f'{tgtRoot}/mfcc')
    
    # random filenames
    np.random.seed(args.seed)
    chosen = np.random.choice(len(srcMFCCs), args.kept_num)
    
    for i in chosen:
        m = srcMFCCs[i]
        if not args.feat_only:
            t = srcTranscripts[i]
            # basenames should be the same
            assert os.path.basename(m) == os.path.basename(t)
        # copy
        shutil.copyfile(f'{srcRoot}/{m}', f'{tgtRoot}/{m}')
        if not args.feat_only:
            shutil.copyfile(f'{srcRoot}/{t}', f'{tgtRoot}/{t}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract a subset of dataset for dev use.')

    parser.add_argument(
        '--src-folder',
        '-f',
        default='./data/train-clean-100',
        type=str,
        help='Original data folder.'
    )
    parser.add_argument(
        '--tgt-folder',
        '-t',
        default='./data/mini-train',
        type=str,
        help='Target output data folder.'
    )
    parser.add_argument(
        '--seed',
        '-s',
        default=11785,
        type=int,
        help='Seed for random splitting.'
    )
    parser.add_argument(
        '--kept-num',
        '-n',
        default=3000,
        type=int,
        help='Number of mfcc-transcript pairs to be kept.'
    )
    parser.add_argument(
        '--feat-only',
        action='store_true',
        default=False,
        help='Whether only the mfcc files are sampled and copied (test set).'
    )

    args = parser.parse_args()

    main(args)


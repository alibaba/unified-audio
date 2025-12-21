import argparse
import sys
import yaml
import pytorch_lightning as pl
from pathlib import Path

from model import Model
from dataloader import DataModule


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.save_enhanced is not None:
        config['save_enhanced'] = args.save_enhanced
        Path(args.save_enhanced).mkdir(parents=True, exist_ok=True)
    
    model = Model(config=config)

    # import ipdb; ipdb.set_trace()
    
    data_module = DataModule(**config['dataset_config'])
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        logger=False,
    )

    trainer.test(model, data_module, ckpt_path=config['ckpt_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser('test model')
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    parser.add_argument('--save_enhanced', type=str, default=None, help='The dir path to save enhanced wavs.')

    args = parser.parse_args()
    sys.exit(main(args))

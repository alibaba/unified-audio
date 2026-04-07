import argparse
import sys
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path

from model import Model_mono2stereo as Model
from dataloader import DataModule


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # override from command line
    if args.ckpt_path:
        config['ckpt_path'] = args.ckpt_path
    if args.input_dir:
        test_kwargs = {'speech_dir': args.input_dir, 'batch_size': 1, 'num_workers': 1, 'prefetch': 1}
    else:
        test_kwargs = {'speech_dir': './test_wavs', 'batch_size': 1, 'num_workers': 1, 'prefetch': 1}

    save_dir = args.output_dir or './output_mono2stereo'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    config['save_enhanced'] = save_dir

    trainer = pl.Trainer(
        accelerator=config.get('accelerator', 'gpu'),
        devices=config.get('devices', [0]),
        logger=False,
    )

    data_module = DataModule(test_kwargs=test_kwargs)
    model = Model(config=config)

    # Load checkpoint manually with strict=False to skip discriminator weights
    ckpt = torch.load(config['ckpt_path'], map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    trainer.test(model, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HCodec Mono2Stereo Inference')
    parser.add_argument('--config', type=str, default='./conf/config_base_mono2stereo.yaml')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory containing input wav files (mono)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output wav files')

    args = parser.parse_args()
    sys.exit(main(args))

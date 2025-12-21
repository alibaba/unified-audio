import pytorch_lightning as pl
import yaml
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Model
from dataloader import DataModule


def main(args):
    pl.seed_everything(3407)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger = TensorBoardLogger(save_dir=config['log_dir'], name='tensorboard')
    ckpt_dir = Path(config['log_dir']) / f'ckpts/version_{logger.version}' #change your folder, where to save files
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config['ckpt_dir'] = ckpt_dir
    model = Model(config=config)
    data_module = DataModule(**config['dataset_config'])
    checkpoint_callback_last = ModelCheckpoint(dirpath=ckpt_dir, save_on_train_epoch_end=True, filename='{epoch}-last')
    
    trainer = pl.Trainer(
        accelerator=config['accelerator'],
        devices=config['devices'],
        max_epochs=config['max_epochs'],
        val_check_interval=config['val_check_interval'],
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=[checkpoint_callback_last],
        logger=logger,
        strategy="auto" if len(config['devices']) == 1 else 'ddp_find_unused_parameters_true',
    )

    trainer.fit(model, data_module, ckpt_path=config['resume'])



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./conf/config.yaml')
    args = parser.parse_args()
    main(args)
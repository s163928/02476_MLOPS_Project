# -*- coding: utf-8 -*-
import click
import logging
import os
import argparse
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from timm.data import create_dataset


class DataLoader(object):
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description='DataLoader Arguments to create/process/load dataset')
        parser.add_argument('--input_filepath', 
            default= "./data/raw",required=False)
        parser.add_argument('--output_filepath', 
            default= "./data/processed",required=False)
        parser.add_argument('--framework', 
            default= "timm-pytorch",required=False)
        parser.add_argument('--dataset', 
            default= "cifar10",required=False)
        # add any additional argument that you want
        self.args = parser.parse_args(sys.argv[2:])
        print(self.args)


        self.logger = logging.getLogger(__name__)
        self.input_filepath = self.args.input_filepath
        self.output_filepath = self.args.output_filepath
        self.framework = self.args.framework
        self.dataset = self.args.dataset

        self.batch_size = 2

        self.load_data()

    def download_data(self):
        self.logger.info(f"Download Data - Framework: {self.framework}, Dataset: {self.dataset}")
        if self.framework == 'timm-pytorch':
            ds = create_dataset('/'.join(['torch',self.dataset]), 
                                os.path.join(self.args.input_filepath, self.dataset), 
                                download=True, 
                                split='train')
        if self.framework == 'timm-tfds':
            ds = create_dataset('/'.join(['tfds',self.dataset]),
                                # '/'.join(['tfds','beans']), 
                                os.path.join(self.args.input_filepath, self.dataset), 
                                download=True, 
                                split='train[:10%]',
                                batch_size=self.batch_size,
                                is_training=True)

    def process_data(self):
        if list(filter(lambda f: f.startswith(self.dataset), 
                            os.listdir(self.args.input_filepath))):
            self.logger.info('Loading Raw Data for processing...')
        else:
            self.logger.info('No Raw Data Found. Initiating Download...')
            self.download_data()

    def load_data(self):
        if list(filter(lambda f: not f.endswith('.gitkeep'), 
                            os.listdir(self.args.output_filepath))):
            self.logger.info('Loading Processed Data to Model...')
        else:
            self.logger.info('No Processed Data Found. Checking Raw Data to Process...')
            self.process_data()
 
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    DataLoader()
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

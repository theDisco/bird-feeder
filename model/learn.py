from fastai.vision.all import *
import pandas as pd
import argparse


class NabirdsLearner:
    MODE_TRAIN = "train"
    MODE_INFRE = "infer"

    def __init__(self, mode, file, infer="magpie.png", fine_tune=4, path="nabirds"):
        self.path = path
        self.file = file

        if mode == self.MODE_TRAIN:
            self.train(fine_tune=fine_tune)
        elif mode == self.MODE_INFRE:
            self.infer(infer=infer)

    def train(self, fine_tune):
        data = self.prepare_data()
        learner = self.create_learner(data)

        learner.fine_tune(fine_tune)
        print(learner.validate())
        learner.save(self.file)

    def infer(self, infer):
        data = self.prepare_data()
        learner = self.create_learner(data)
        learner.load(self.file)
        print(learner.predict(infer))

    def create_learner(self, data):
        item_tfms = [Resize(300)]
        # aug_transforms:
        # Used for GAN data augmentation. It is sometimes difficult and expensive
        # to annotate a large amount of training data. Therefore, proper data augmentation
        # is useful to increase the model performance.
        #
        # imagenet_stats:
        # Since we have a pretrained resnet backbone (trained on ImageNet),
        # we need to normalize our data based on what the model was trained on.
        # As a result we use the mean and std of the ImageNet dataset, or imagenet_stats.
        batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
        # TODO: Add valid_col= to denote train/test image.
        data = ImageDataLoaders.from_df(
            data, path=self.path+"/images", fn_col=1, item_tfms=item_tfms,
            batch_tfms=batch_tfms, label_col=3)

        return vision_learner(data, resnet50, metrics=accuracy)

    def prepare_data(self):
        images = pd.read_csv(self.path + '/images.txt', sep=" ",
                             header=None, names=['file', 'path'])
        images['cat_num'] = images['path'].str.split('/').str[0]

        classes = pd.read_table(self.path + '/classes.txt', delimiter=None)
        classes.columns = ['code']
        classes[['cat_num', 'common_name']] = classes['code'].str.split(
            " ", n=1, expand=True)
        classes = classes.drop(['code'], axis=1)
        # Fill leading zeros
        classes['cat_num'] = classes['cat_num'].str.zfill(4)
        images = pd.merge(images, classes,  how='left', on='cat_num')

        split = pd.read_csv(self.path + '/train_test_split.txt', sep=" ",
                            header=None, names=['file', 'train_test'])
        images = pd.merge(images, split,  how='left', on='file')

        bounding_box = pd.read_csv(self.path + '/bounding_boxes.txt',
                                   sep=" ", header=None, names=['file', 'p1', 'p2', 'p3', 'p4'])
        images = pd.merge(images, bounding_box,  how='left', on='file')


def mode_validator(string):
    modes = [NabirdsLearner.MODE_INFRE, NabirdsLearner.MODE_TRAIN]
    if string not in modes:
        raise argparse.ArgumentTypeError('train or infer')
    return string


def path_validator(string):
    if string is None:
        return 'nabirds/'
    return string


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode", help="Possible options: train, infer", type=mode_validator, required=True)
parser.add_argument(
    "-f", "--file", help="Name of the model file to save, no path, no extension", required=True)
parser.add_argument(
    "-p", "--path", help="Arbitrary path to NABirds model, default: nabirds")
parser.add_argument(
    "-i", "--infer", help="File to infer against the model")
args = parser.parse_args()

NabirdsLearner(mode=args.mode, path=args.path,
               file=args.file, infer=args.infer)

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import logging
import pprint

import numpy as np
import numpy
import torch
import torchvision.transforms as transforms
import cyanure as cyan
from openTSNE import affinity, initialization, TSNEEmbedding, TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib


import src.deit as deit
from src.data_manager import (
    init_train_test_dataloaders,
    ReturnIndexDataset,
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lambd', type=float,
    default=0.00025,
    help='regularization')
parser.add_argument(
    '--penalty', type=str,
    help='regularization for logistic classifier',
    default='l2',
    choices=[
        'l2',
        'elastic-net'
    ])
parser.add_argument(
    '--mask', type=float,
    default=0.0,
    help='regularization')
parser.add_argument(
    '--preload', action='store_true',
    help='whether to preload embs if possible')
parser.add_argument(
    '--fname', type=str,
    help='model architecture')
parser.add_argument(
    '--model-name', type=str,
    help='model architecture')
parser.add_argument(
    '--pretrained', type=str,
    help='path to pretrained model',
    default='')
parser.add_argument(
    '--device', type=str,
    default='cuda:0',
    help='device to run script on')
parser.add_argument(
    '--normalize', type=bool,
    default=True,
    help='whether to standardize images before feeding to nework')
parser.add_argument(
    '--root-path', type=str,
    default='/datasets/',
    help='root directory to data')
parser.add_argument(
    '--image-folder', type=str,
    default='imagenet_full_size/061417/',
    help='image directory inside root_path')
parser.add_argument(
    '--subset-path', type=str,
    default=None,
    help='name of dataset to evaluate on')

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(
    blocks,
    lambd,
    mask_frac,
    preload,
    pretrained,
    fname,
    subset_path,
    root_path,
    image_folder,
    penalty='l2',
    model_name=None,
    normalize=True,
    device_str='cuda:0'
):
    dataset = ReturnIndexDataset(os.path.join(root_path))
    device = torch.device(device_str)
    if 'cuda' in device_str:
        torch.cuda.set_device(device)

    # -- Define file names used to save computed embeddings (for efficient
    # -- reuse if running the script more than once)
    subset_tag = '-'.join(subset_path.split('/')).split('.txt')[0] if subset_path is not None else 'imagenet_subses1-100percent'
    train_embs_path = os.path.join(pretrained, f'train-features-{subset_tag}-{fname}')
    test_embs_path = os.path.join(pretrained, f'val-features-{fname}')
    logger.info(train_embs_path)
    logger.info(test_embs_path)

    pretrained = os.path.join(pretrained, fname)

    # -- Function to make train/test dataloader
    def init_pipe(training):
        # -- make data transforms
        # -- init data-loaders/samplers
        data_loader_train, data_loader_val, train_sampler, val_sampler = init_train_test_dataloaders(
            dataset,
            batch_size=16,
            num_workers=0,
            world_size=1,
            rank=0,
            root_path=root_path,
            drop_last=False)
        if training is True:
            return data_loader_train
        else:
            return data_loader_val

    # -- Initialize the model
    encoder = init_model(
        device=device,
        pretrained=pretrained,
        model_name=model_name)
    encoder.eval()

    # -- If train embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(train_embs_path):
        checkpoint = torch.load(train_embs_path, map_location='cpu')
        embs, labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded embs of shape {embs.shape}')
    else:
        data_loader = init_pipe(True)
        embs, labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=mask_frac,
            data_loader=data_loader,
            encoder=encoder)
        #save labels as class names
        classes = [s for s in dataset.classes]
        labs = [classes[dataset.targets[i]] for i in labs]
        torch.save({
            'embs': embs,
            'labs': labs
        }, train_embs_path)
        logger.info(f'saved train embs of shape {embs.shape}')

    #make plots
        #normalize tsne tensors
    def normalize_tensor_values_0_1(tensor):
        tensor_min, _ = torch.min(tensor, dim=0, keepdim=True)
        tensor_max, _ = torch.max(tensor, dim=0, keepdim=True)
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return tensor

    def make_plot(args, emb_tensor,val_labels_2, class_names, val_features,name="TSNE", description="T-SNE param.: Perplexity=40, Iterations=1500"):
        #create a list of colors for each label from the matplotlib color map
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]
        print(len(colors))
        plt.figure(figsize=(14, 10))
        plt.suptitle(name+" of MSN analyis: ", fontsize=13)
        plt.title("CLS Token embedding of "+str(len(val_labels_2))+" cells with a dimensionality of "+str(val_features.shape[1])+" \n"+description)
        scatter = plt.scatter(emb_tensor[:,0], emb_tensor[:,1],c=val_labels_2, alpha=1, cmap=matplotlib.colors.ListedColormap(colors), edgecolors='white', linewidths=0.1)
        handles, labels = scatter.legend_elements(num=len(class_names))
        plt.legend(handles,class_names,loc="upper right", title="Classes")
        plt.box(False)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("/nfs/nas22.ethz.ch/fs2202/biol_imsb_snijder_1/Data/Masters/Jacob/euler/msn/"+name+"num_cells_"+str(len(val_labels_2))+"_msn.png", dpi=300)

    def transform_val_labels_into_integers(val_labels):
        le = preprocessing.LabelEncoder()
        le.fit(val_labels)
        labels_numbers = le.transform(val_labels)
        class_names = tuple(le.classes_)
        return labels_numbers, class_names

    print("Running PCA on features...")
    pca = PCA(n_components=2)
    pca_emb= torch.from_numpy(pca.fit_transform(embs.cpu().numpy()))
    pca_emb = normalize_tensor_values_0_1(pca_emb)
    labels_numbers, class_names = transform_val_labels_into_integers(labs)
    make_plot(args, pca_emb, labels_numbers, class_names, embs, name="PCA", description="PCA param.: n_components=2")
   

    print("computing TSNE...")

    #Simple T-SNE
    tsne = TSNE(
    perplexity=40,
    metric="euclidean",
    n_jobs=8,
    random_state=0,
    verbose=False,
    n_iter=1500
    )

    tsne_emb = tsne.fit(numpy.array(embs))
    tsne_emb = torch.tensor(tsne_emb)
    tsne_emb = normalize_tensor_values_0_1(tsne_emb)
    labels_numbers, class_names = transform_val_labels_into_integers(labs)
    make_plot(args, tsne_emb, labels_numbers,class_names, embs, name="TSNE", description="T-SNE param.: Perplexity=40, Iterations=1500")
    print("finished TSNE")
    
    # -- Normalize embeddings
    cyan.preprocess(embs, normalize=normalize, columns=False, centering=True)

    # -- Fit Logistic Regression Classifier
    labs, _ = transform_val_labels_into_integers(labs)
    print(labs)
    classifier = cyan.MultiClassifier(loss='multiclass-logistic', penalty=penalty, fit_intercept=False)
    lambd /= len(embs)
    classifier.fit(
        embs.numpy(),
        labs,
        it0=10,
        lambd=lambd,
        lambd2=lambd,
        nthreads=-1,
        tol=1e-3,
        solver='auto',
        seed=0,
        max_epochs=300)

    # -- Evaluate and log
    train_score = classifier.score(embs.numpy(), labs)
    # -- (save train score)
    logger.info(f'train score: {train_score}')

    # -- If test embeddings already computed, load file, otherwise, compute
    # -- embeddings and save
    if preload and os.path.exists(test_embs_path):
        checkpoint = torch.load(test_embs_path, map_location='cpu')
        test_embs, test_labs = checkpoint['embs'], checkpoint['labs']
        logger.info(f'loaded test embs of shape {test_embs.shape}')
    else:
        data_loader = init_pipe(False)
        test_embs, test_labs = make_embeddings(
            blocks=blocks,
            device=device,
            mask_frac=0.0,
            data_loader=data_loader,
            encoder=encoder)
        #get class names of images
        classes = [s for s in dataset.classes]
        test_labs = [classes[dataset.targets[i]] for i in test_labs]
        torch.save({
            'embs': test_embs,
            'labs': test_labs
        }, test_embs_path)
        logger.info(f'saved test embs of shape {test_embs.shape}')


    # -- Normalize embeddings
    cyan.preprocess(test_embs, normalize=normalize, columns=False, centering=True)

    # -- Evaluate and log
    test_labs, _ = transform_val_labels_into_integers(test_labs)
    test_score = classifier.score(test_embs.numpy(), test_labs)
    # -- (save test score)
    logger.info(f'test score: {test_score}\n\n')

    return test_score


def make_embeddings(
    blocks,
    device,
    mask_frac,
    data_loader,
    encoder,
    epochs=1
):
    ipe = len(data_loader)

    z_mem, l_mem = [], []

    for _ in range(epochs):
        for itr, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            with torch.no_grad():
                z = encoder.forward_blocks(imgs, blocks, mask_frac).cpu()
            labels = labels.cpu()
            z_mem.append(z)
            l_mem.append(labels)
            if itr % 50 == 0:
                logger.info(f'[{itr}/{ipe}]')

    z_mem = torch.cat(z_mem, 0)
    l_mem = torch.cat(l_mem, 0)
    logger.info(z_mem.shape)
    logger.info(l_mem.shape)

    return z_mem, l_mem


def load_pretrained(
    encoder,
    pretrained
):
    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f'loaded pretrained model with msg: {msg}')
    try:
        logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]} '
                    f'path: {pretrained}')
    except Exception:
        pass
    del checkpoint
    return encoder


def init_model(
    device,
    pretrained,
    model_name,
):
    encoder = deit.__dict__[model_name]()
    encoder.fc = None
    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained)

    return encoder


if __name__ == '__main__':
    """'main' for launching script using params read from command line"""
    global args
    args = parser.parse_args()
    pp.pprint(args)
    main(
        blocks=1,
        lambd=args.lambd,
        penalty=args.penalty,
        mask_frac=args.mask,
        preload=args.preload,
        pretrained=args.pretrained,
        fname=args.fname,
        subset_path=args.subset_path,
        root_path=args.root_path,
        image_folder=args.image_folder,
        model_name=args.model_name,
        normalize=args.normalize,
        device_str=args.device
    )

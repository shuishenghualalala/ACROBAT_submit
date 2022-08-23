import os

import torch
import time
from tqdm import tqdm
from utils.basic_utils import FixedNum
from utils.criterion import curvature_regularization,Bend_Penalty

def train(args, model, loader, criterion, optimizer,scheduler):
    # Train
    model.train()
    train_losses = 0
    NCC_losses = 0
    curvature_losses = 0
    loc = args.device
    with tqdm(total=len(loader)) as pbar:
        for i, inputs in enumerate(loader):
            # print(model.alpha)

            sources = inputs['source'].to(loc).type(torch.cuda.FloatTensor)
            targets = inputs['target'].to(loc).type(torch.cuda.FloatTensor)

            transform_sources, fields = model(sources,targets)
            NCC_loss = torch.mean(criterion(transform_sources,targets))
            curvature_loss = torch.mean(curvature_regularization(fields,loc))
            total_loss = NCC_loss + args.alpha * curvature_loss
            train_losses += NCC_loss.item()+args.alpha*curvature_loss.item()
            NCC_losses += NCC_loss.item()
            curvature_losses += curvature_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.update(1)

    if not scheduler == None:
        scheduler.step()

    losses = [FixedNum(train_losses/len(loader)),FixedNum(NCC_losses/len(loader)),FixedNum(curvature_losses/len(loader))]

    return losses,optimizer

def DF_val(args, model, loader,criterion):
    # Train
    model.train()
    loc = args.device
    NCC_losses = 0
    curvature_losses = 0
    val_losses = 0

    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, inputs in enumerate(loader):
                # print(model.alpha)
                sources = inputs['source'].to(args.device).type(torch.cuda.FloatTensor)
                targets = inputs['target'].to(args.device).type(torch.cuda.FloatTensor)

                transform_sources, fields,_ = model(sources,targets)
                ncc_loss = torch.mean(criterion(transform_sources,targets)).item()
                curvature_loss = torch.mean(curvature_regularization(fields, loc)).item()

                NCC_losses += ncc_loss
                curvature_losses += curvature_loss
                val_losses += ncc_loss + args.alpha * curvature_loss
                pbar.update(1)
    losses = [FixedNum(val_losses/len(loader)),FixedNum(NCC_losses/len(loader)),FixedNum(curvature_losses/len(loader))]
    return losses

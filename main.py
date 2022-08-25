
import torch.distributed as dist
from tqdm import tqdm
import os
import torch
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import time

from utils.args import ACROBAT_args
from utils.train_val import train,val
from data.dataset import load_dataloader
from network.load_model import load_network


# parser
args = ACROBAT_args()
def main(args):
    print(args)
    print(torch.__version__)
    Flag = f'{args.exp}_{args.modelname}_{args.batch_size}_{args.lr}'
    log_dir = f'log_file/{Flag}/'
    if not os.path.exists(f'checkpoints/{Flag}'):
        os.mkdir(f'checkpoints/{Flag}')
    # 保存配置文件
    torch.save(args,f'checkpoints/{Flag}/args.dic')
    tensorboard_path = log_dir
    Writer = SummaryWriter(tensorboard_path)


    # model, optimizer and loss
    model = load_network(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    cudnn.benchmark = True

    Train_dataset, Train_dataloader = load_dataloader(args, 'train')
    Val_dataset, Val_dataloader = load_dataloader(args, 'validation')

    # loop
    best_cri = 100
    train_loss = 100
    val_loss = 100


    for epoch in range(args.purge_step, args.epoch):

        train_losses ,optimizer = train(args,model,Train_dataloader,optimizer,scheduler)
        val_losses = val(args,model,Val_dataloader)
        for param_group in optimizer.param_groups:
            lr_latest=param_group['lr']

        train_loss,train_ncc_loss,curvature_loss= train_losses
        val_loss, val_ncc_loss, val_curvature_loss = val_losses
        Writer.add_scalar('scalar/train_loss', train_loss, epoch)
        Writer.add_scalar('scalar/train_ncc_loss', train_ncc_loss, epoch)
        Writer.add_scalar('scalar/curvature_loss', curvature_loss, epoch)
        print("\r[Exp:{}][Train][Epoch {}/{}][lr{}][train_loss:{:.8f}][learning rate:{}]".format(args.exp,
                                                                                                 epoch + 1,
                                                                                                 args.epoch,
                                                                                                 lr_latest,
                                                                                                 train_loss,
                                                                                                 args.lr))




        '''save checkpoints'''
        cri = val_loss
        if epoch != args.epoch - 1 and cri <= best_cri:
            best_cri = cri
            path = f'checkpoints/{Flag}/{epoch}_{train_loss}_{val_loss}_{best_cri}_{lr_latest}.pth'
            torch.save(model.state_dict(), path)
        if epoch == args.epoch - 1:
            path = f'checkpoints/{Flag}/{epoch}_{train_loss}_{val_loss}_{cri}_{lr_latest}.pth'
            torch.save(model.state_dict(), path)







if __name__ == '__main__':


    args.Pair_path = '/data_sdd/lyh/ACROBAT/results/AF_validate'
    args.Pair_path_validate = '/data_sdd/lyh/ACROBAT/results/AF_validate'
    # args.modelname = 'Deformation'
    args.modelname = 'Affine'
    args.batch_size = 1
    args.patch_size = 512
    args.exp = 'exp010'
    args.device = 'cuda:1'
    # args.cpt = 'network/MaskFlownet-Pytorch/weights/5adNov03-0005_1000000.pth'

    main(args)

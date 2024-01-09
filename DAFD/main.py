import configargparse
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--text", type=str,default='')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_gamma', type=float, default=0)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)
    parser.add_argument('--transfer_loss_weight', type=float, default=0)
    parser.add_argument('--transfer_loss', type=str)
    parser.add_argument('--lmmd_lamb', type=float, default=0)
    parser.add_argument('--recon_lamb', type=float, default=0)
    parser.add_argument('--mi_lamb', type=float, default=0)
    parser.add_argument('--adv_lamb', type=float, default=0)
    parser.add_argument('--dynamic', type=str2bool, default=True)
    parser.add_argument('--pixel_adv_lamb', type=float, default=0)
    parser.add_argument('--d_lamb', type=float, default=0)
    parser.add_argument('--task',type=str)
    return parser

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone,\
              max_iter=args.max_iter, use_bottleneck=args.use_bottleneck,recon_lamb = args.recon_lamb,lmmd_lamb = args.lmmd_lamb,mi_lamb = args.mi_lamb,adv_lamb = args.adv_lamb,dynamic = args.dynamic,d_lamb = 1,pixel_adv_lamb=args.pixel_adv_lamb).to(args.device)
    state_dict = torch.load(f'./modelparameters/{args.task}/{args.src_domain}_{args.tgt_domain}.pth')
    model.load_state_dict(state_dict, strict=True)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def test_func(target_test_loader, model, args):
    log = []
    test_acc, test_loss = test(model, target_test_loader, args)
    info = ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
    np_log = np.array(log, dtype=float)
    np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
    print(info)
def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    test_func(target_test_loader, model,args)
if __name__ == "__main__":
    main()

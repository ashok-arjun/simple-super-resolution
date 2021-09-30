import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import get_most_recent_checkpoint,get_test_set,get_training_set, set_seed
from math import log10
from model.srcnn_upconv7 import Upconv
from model.rdn import RDN
import argparse
import os


from os.path import exists, join, basename
from os import makedirs, remove
import urllib
import tarfile

def download_bsd300(dest):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)
    else:
        print("BSDS300 dataset already exists")

    return output_image_dir


'''
Training Settings
'''
def str2bool(v):
  return str(v).lower() in ("y", "yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Pytorch Image/Video Super-Resolution')
parser.add_argument('--upscale_factor',type=int,required=True, help="Super-resolution upscale factor")
parser.add_argument('--datapath',type=str,required=True,help="Path to Original data")
parser.add_argument('--model',type=str,default="RDN",help="Choose which SR model to use")
parser.add_argument('--threads',type=int,default=4,help='Number of thread for DataLoader')

parser.add_argument('--lr',type=float,default=0.001,help='Learning rate')
parser.add_argument('--nEpochs',type=int,default=1000,help='Number of epochs')
parser.add_argument('--batchSize',type=int,default=8,help='Training batch size')
parser.add_argument('--testBatchSize',type=int,default=4,help='Test batch size')
parser.add_argument('--isCuda',type=str2bool,default=True,help='Cuda Usage')

opt = parser.parse_args()

print(opt)

lr = opt.lr
nEpochs = opt.nEpochs
batchSize = opt.batchSize
testBatchSize = opt.testBatchSize
isCuda = opt.isCuda

set_seed(0)

if isCuda and not torch.cuda.is_available():
    raise Exception("No GPU, please change isCuda False")

device = torch.device("cuda" if isCuda else "cpu")

print('===> Loading datasets')

dataset_path = download_bsd300(opt.datapath)
train_set = get_training_set(opt.upscale_factor,dataset_path)
test_set = get_test_set(opt.upscale_factor,dataset_path)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=testBatchSize, shuffle=False)

print('===> Datasets Loading Complete')


print('===> Model Initialize')


if opt.model == "Upconv":
    model = Upconv(upscale_factor=opt.upscale_factor).to(device)
    os.makedirs('ckpt/Upconv',exist_ok=True)
    criterion = model.criterion
    optimizer = model.optimizer
    #scheduler = model.scheduler
    if len(next(os.walk('ckpt/Upconv'))[2]) != 0:
        min_iter = 1
        last_ckpt, min_iter = get_most_recent_checkpoint('ckpt/Upconv')
        model = torch.load(last_ckpt)
    else :
        min_iter = 1

elif opt.model == "RDN":
    model = RDN(channel = 1,growth_rate = 64,rdb_number = 3,upscale_factor=opt.upscale_factor).to(device)
    os.makedirs('ckpt/RDN',exist_ok=True)
    criterion = model.criterion
    optimizer = model.optimizer
    scheduler = model.scheduler
    if len(next(os.walk('ckpt/RDN'))[2]) != 0:
        min_iter = 1
        last_ckpt, min_iter = get_most_recent_checkpoint('ckpt/')
        model = torch.load(last_ckpt)
    else :
        min_iter = 1

print('===> Model Initialize Complete')


'''

Model Implementation

elif opt.model == "Model_name":
    model = Model_name(upscale_factor=opt.upscale_factor).to(device)
    os.makedirs('ckpt/Model_name',exist_ok=True)
    criterion = model.criterion
    optimizer = model.optimizer
    scheduler = model.scheduler
    if len(next(os.walk('ckpt/Model_name'))[2]) != 0:
        min_iter = 1
        last_ckpt, min_iter = get_most_recent_checkpoint('ckpt/Model_name')
        model = torch.load(last_ckpt)
    else :
        min_iter = 1


'''

print('===> Training Initialize')



if torch.cuda.is_available():
    cudnn.benchmark = True
    criterion.cuda()

print('===> Training Initialize Complete')

def train(epoch):
    print('===> Training # %d epoch'%(epoch))

    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    print('===> Testing # %d epoch'%(epoch))
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.6f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):

    if opt.model == "Upconv":
        model_out_path = "ckpt/" + "Upconv" + "/model_epoch_{}.pth".format(epoch)

    elif opt.model == "RDN":
        model_out_path = "ckpt/" + "RDN" + "/model_epoch_{}.pth".format(epoch)

    '''
    Model Implementation
    elif opt.model == "Model_Name":
        model_out_path = "ckpt/" + "Model_Name" + "/model_epoch_{}.pth".format(epoch)
    '''

    print(model_out_path)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    for epoch in range(min_iter, nEpochs + 1):
        print("=====>  Training %d epochs"%(epoch))
        train(epoch)
        print("=====>  Training %d epochs completed"%(epoch))
        print("=====>  Testing %d epochs"%(epoch))
        test()
        print("=====>  Testing %d epochs completed"%(epoch))
        print("=====>  lr scheduler activated in %d epochs"%(epoch))
        scheduler.step(epoch)
        print("=====>  lr scheduler activated in %d epochs completed"%(epoch))
        print("=====>  Save checkpoint %d epochs"%(epoch))
        checkpoint(epoch)
        print("=====>  Save checkpoint %d epochs completed"%(epoch))


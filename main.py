# Code for "Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution"
# Author: Jin-Fan Hu

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DatasetFromHdf5_real, DatasetFromFolder, DatasetFromHdf5
# from model import PanNet,summaries
from model import *
import numpy as np
import scipy.io as sio
import shutil
from torch.utils.tensorboard import SummaryWriter
import time
from metrics import calc_psnr, calc_rmse, calc_ergas, calc_sam, ssim, scc
import os
from cal_ssim import SSIM, set_random_seed
from PIL import Image
import pytorch_msssim
# ================== Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True  ###????????
cudnn.deterministic = True
# cudnn.benchmark = False
# ============= HYPER PARAMS(Pre-Defined) ==========#
# lr = 0.001
# epochs = 510
# ckpt = 200
# batch_size = 32
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5)   # optimizer 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

lr = 1e-4
epochs = 200
ckpt_step = 5
batch_size = 64

model = ImageFusionNetwork().cuda()
# model = nn.DataParallel(model)
# model_path = "Weights/.pth"
# if os.path.isfile(model_path):
#     # Load the pretrained Encoder
#     model.load_state_dict(torch.load(model_path))
#     print('Network is Successfully Loaded from %s' % (model_path))
# from torchstat import stat
# stat(model, input_size=[(31, 16, 16), (3, 64, 64)])
# summaries(model, grad=True)
ssim_loss = pytorch_msssim.ms_ssim
PLoss = nn.L1Loss(size_average=True).cuda()
# Sparse_loss = SparseKLloss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)   # optimizer 1
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-7)  # optimizer 2
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                              gamma=0.1)  # lr = lr* 1/gamma for each step_size = 180

# if os.path.exists('train_logs'):  # for tensorboard: copy dir of train_logs
#     shutil.rmtree('train_logs')  # ---> console (see tensorboard): tensorboard --logdir = dir of train_logs  --host=127.0.0.1
# writer = SummaryWriter('./train_logs(model-Trans)/')

model_folder = "./fusion_model/"
#writer = SummaryWriter("train_logs/ "+model_folder)
def save_checkpoint(model, epoch):  # save model function

    model_out_path = model_folder + "{}.pth".format(epoch)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr":lr
    }
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, start_epoch=0,RESUME=False):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=2)
    print('Start training...')

    if RESUME:
        path_checkpoint = model_folder+"{}.pth".format(2000)
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    time_s = time.time()
    for epoch in range(start_epoch, epochs, 1):

        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            image1, image2 = batch[0].cuda(), batch[1].cuda()

            optimizer.zero_grad()  # fixed

            SR = model(image1, image2)

            time_e = time.time()
            Pixelwise_Loss =PLoss(image1, SR)+PLoss(image2, SR) +(1-ssim_loss(image1, SR))+(1-ssim_loss(image2, SR))
            # Pixelwise_Loss  = PLoss(Rec_LRHSI, LRHSI) + PLoss(Rec_HRMSI,HRMSI)
            # Sparse = Sparse_loss(A) + Sparse_loss(A_LR)
            # ASC_loss = Sum2OneLoss(A) + Sum2OneLoss(A_LR)

            Myloss = Pixelwise_Loss
            epoch_train_loss.append(Myloss.item())  # save all losses into a vector for one epoch

            Myloss.backward()  # fixed
            optimizer.step()  # fixed

            if iteration % 10 == 0:
                # log_value('Loss', loss.data[0], iteration + (epoch - 1) * len(training_data_loader))
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader),
                                                                   Myloss.item()))
                # for name, parameters in model.named_parameters():
                #     print(name, ':', parameters.size(), parameters)
            # for name, layer in model.named_parameters():
            #     # writer.add_histogram('torch/'+name + '_grad_weight_decay', layer.grad, epoch*iteration)
            #     writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*iteration)
        print("learning rate:º%f" % (optimizer.param_groups[0]['lr']))
        lr_scheduler.step()  # update lr

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        #writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('EpocI: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch
        print(time_e - time_s)
        
        # ============Epoch Validate=============== #
        if epoch % ckpt_step == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)
       

 
    
def test_real():
    testreal_set = DatasetFromFolder('I:/水文+SR/SR+Fusion/实验结果/production/IMERG-Early_UP/2019', 'I:/水文+SR/SR+Fusion/实验结果/production/SM2RAIN_UP/2019')

    num_testing = testreal_set.__len__()

    scale = 10
    testing_data_loader = DataLoader(dataset=testreal_set, num_workers=0, batch_size=1)
    wid = 4011
    hei = 7011
    output_HRone = np.zeros((184, wid, hei, 1))
    output_HRtwo = np.zeros((num_testing-184, wid, hei, 1))

    #output_HRbicone = np.zeros((num_testing, wid, hei, 1))
    model = ImageFusionNetwork1().cuda()
    #model = torch.nn.DataParallel(model)
    #path_checkpoint = model_folder + "{}.pth".format(200)
    path_checkpoint = model_folder + "200.pth"
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()


    for iteration, (batch1, image_name1, batch2, image_name2) in enumerate(testing_data_loader, 1):
        print(iteration)


        imag1= batch1[0].cuda()
#
        imag1 = torch.unsqueeze(imag1, dim=0)
        imag1 = torch.unsqueeze(imag1, dim=0)
        
        imag2= batch2[0].cuda()
#
        imag2 = torch.unsqueeze(imag2, dim=0)
        imag2 = torch.unsqueeze(imag2, dim=0)
 
        with torch.no_grad():
             output_HR= model(imag1, imag2)


        #output_HR = F.interpolate(GT, scale_factor=(10, 10),mode ='bicubic')
        if iteration<=184:


            output_HRone[iteration-1,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRone)  # 去除单维度
   
            a = np.squeeze(output[iteration-1, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
    
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2019/" "fusion_out"+ image_name2[0])
        else:
            output_HRtwo[iteration-185,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRtwo)  # 去除单维度
            a = np.squeeze(output[iteration-185, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2019/" "fusion_out"+ image_name2[0]) 
    
def test_real1():
    testreal_set = DatasetFromFolder('I:/水文+SR/SR+Fusion/实验结果/production/IMERG-Early_UP/2018', 'I:/水文+SR/SR+Fusion/实验结果/production/SM2RAIN_UP/2018')

    num_testing = testreal_set.__len__()

    scale = 10
    testing_data_loader = DataLoader(dataset=testreal_set, num_workers=0, batch_size=1)
    wid = 4011
    hei = 7011
    output_HRone = np.zeros((184, wid, hei, 1))
    output_HRtwo = np.zeros((num_testing-184, wid, hei, 1))

    #output_HRbicone = np.zeros((num_testing, wid, hei, 1))
    model = ImageFusionNetwork1().cuda()
    #model = torch.nn.DataParallel(model)
    #path_checkpoint = model_folder + "{}.pth".format(10)
    path_checkpoint = model_folder + "200.pth"
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()


    for iteration, (batch1, image_name1, batch2, image_name2) in enumerate(testing_data_loader, 1):
        print(iteration)


        imag1= batch1[0].cuda()
#
        imag1 = torch.unsqueeze(imag1, dim=0)
        imag1 = torch.unsqueeze(imag1, dim=0)
        
        imag2= batch2[0].cuda()
#
        imag2 = torch.unsqueeze(imag2, dim=0)
        imag2 = torch.unsqueeze(imag2, dim=0)
 
        with torch.no_grad():
             output_HR= model(imag1, imag2)


        #output_HR = F.interpolate(GT, scale_factor=(10, 10),mode ='bicubic')
        if iteration<=184:


            output_HRone[iteration-1,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRone)  # 去除单维度
   
            a = np.squeeze(output[iteration-1, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
    
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2018/" "fusion_out"+ image_name2[0])
        else:
            output_HRtwo[iteration-185,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRtwo)  # 去除单维度
            a = np.squeeze(output[iteration-185, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2018/" "fusion_out"+ image_name2[0]) 
            
def test_real2():
    testreal_set = DatasetFromFolder('I:/水文+SR/SR+Fusion/实验结果/production/IMERG-Early_UP/2017', 'I:/水文+SR/SR+Fusion/实验结果/production/SM2RAIN_UP/2017')

    num_testing = testreal_set.__len__()

    scale = 10
    testing_data_loader = DataLoader(dataset=testreal_set, num_workers=0, batch_size=1)
    wid = 4011
    hei = 7011
    output_HRone = np.zeros((184, wid, hei, 1))
    output_HRtwo = np.zeros((num_testing-184, wid, hei, 1))

    #output_HRbicone = np.zeros((num_testing, wid, hei, 1))
    model = ImageFusionNetwork1().cuda()
    #model = torch.nn.DataParallel(model)
    #path_checkpoint = model_folder + "{}.pth".format(10)
    path_checkpoint = model_folder + "200.pth"
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()


    for iteration, (batch1, image_name1, batch2, image_name2) in enumerate(testing_data_loader, 1):
        print(iteration)


        imag1= batch1[0].cuda()
#
        imag1 = torch.unsqueeze(imag1, dim=0)
        imag1 = torch.unsqueeze(imag1, dim=0)
        
        imag2= batch2[0].cuda()
#
        imag2 = torch.unsqueeze(imag2, dim=0)
        imag2 = torch.unsqueeze(imag2, dim=0)
 
        with torch.no_grad():
             output_HR= model(imag1, imag2)


        #output_HR = F.interpolate(GT, scale_factor=(10, 10),mode ='bicubic')
        if iteration<=184:


            output_HRone[iteration-1,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRone)  # 去除单维度
   
            a = np.squeeze(output[iteration-1, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
    
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2017/" "fusion_out"+ image_name2[0])
        else:
            output_HRtwo[iteration-185,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRtwo)  # 去除单维度
            a = np.squeeze(output[iteration-185, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2017/" "fusion_out"+ image_name2[0]) 
  
def test_real3():
    testreal_set = DatasetFromFolder('I:/水文+SR/SR+Fusion/实验结果/production/IMERG-Early_UP/2016', 'I:/水文+SR/SR+Fusion/实验结果/production/SM2RAIN_UP/2016')

    num_testing = testreal_set.__len__()

    scale = 10
    testing_data_loader = DataLoader(dataset=testreal_set, num_workers=0, batch_size=1)
    wid = 4011
    hei = 7011
    output_HRone = np.zeros((184, wid, hei, 1))
    output_HRtwo = np.zeros((num_testing-184, wid, hei, 1))

    #output_HRbicone = np.zeros((num_testing, wid, hei, 1))
    model = ImageFusionNetwork1().cuda()
    #model = torch.nn.DataParallel(model)
    #path_checkpoint = model_folder + "{}.pth".format(10)
    path_checkpoint = model_folder + "200.pth"
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()


    for iteration, (batch1, image_name1, batch2, image_name2) in enumerate(testing_data_loader, 1):
        print(iteration)


        imag1= batch1[0].cuda()
#
        imag1 = torch.unsqueeze(imag1, dim=0)
        imag1 = torch.unsqueeze(imag1, dim=0)
        
        imag2= batch2[0].cuda()
#
        imag2 = torch.unsqueeze(imag2, dim=0)
        imag2 = torch.unsqueeze(imag2, dim=0)
 
        with torch.no_grad():
             output_HR= model(imag1, imag2)


        #output_HR = F.interpolate(GT, scale_factor=(10, 10),mode ='bicubic')
        if iteration<=184:


            output_HRone[iteration-1,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRone)  # 去除单维度
   
            a = np.squeeze(output[iteration-1, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
    
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2016/" "fusion_out"+ image_name2[0])
        else:
            output_HRtwo[iteration-185,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRtwo)  # 去除单维度
            a = np.squeeze(output[iteration-185, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2016/" "fusion_out"+ image_name2[0]) 
            
            
def test_real4():
    testreal_set = DatasetFromFolder('I:/水文+SR/SR+Fusion/实验结果/production/IMERG-Early_UP/2015', 'I:/水文+SR/SR+Fusion/实验结果/production/SM2RAIN_UP/2015')

    num_testing = testreal_set.__len__()

    scale = 10
    testing_data_loader = DataLoader(dataset=testreal_set, num_workers=0, batch_size=1)
    wid = 4011
    hei = 7011
    output_HRone = np.zeros((184, wid, hei, 1))
    output_HRtwo = np.zeros((num_testing-184, wid, hei, 1))

    #output_HRbicone = np.zeros((num_testing, wid, hei, 1))
    model = ImageFusionNetwork1().cuda()
    #model = torch.nn.DataParallel(model)
    #path_checkpoint = model_folder + "{}.pth".format(10)
    path_checkpoint = model_folder + "200.pth"
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net'])
    model = model.cuda()


    for iteration, (batch1, image_name1, batch2, image_name2) in enumerate(testing_data_loader, 1):
        print(iteration)


        imag1= batch1[0].cuda()
#
        imag1 = torch.unsqueeze(imag1, dim=0)
        imag1 = torch.unsqueeze(imag1, dim=0)
        
        imag2= batch2[0].cuda()
#
        imag2 = torch.unsqueeze(imag2, dim=0)
        imag2 = torch.unsqueeze(imag2, dim=0)
 
        with torch.no_grad():
             output_HR= model(imag1, imag2)


        #output_HR = F.interpolate(GT, scale_factor=(10, 10),mode ='bicubic')
        if iteration<=184:


            output_HRone[iteration-1,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRone)  # 去除单维度
   
            a = np.squeeze(output[iteration-1, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
    
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2015/" "fusion_out"+ image_name2[0])
        else:
            output_HRtwo[iteration-185,:,:,:] = output_HR.permute([0, 2, 3, 1]).cpu().detach().numpy()
            output = np.squeeze(output_HRtwo)  # 去除单维度
            a = np.squeeze(output[iteration-185, :, :])
           # print(output.shape)
           # output = np.clip(output, 0, 255).astype(np.uint8)  # 将值限制在0-255之间，并转为无符号整型          
            image = Image.fromarray(a)
            image.save("I:/水文+SR/SR+Fusion/实验结果/production/fusion_ours_up/2015/" "fusion_out"+ image_name2[0]) 
 


            

            
# ------------------- Main Function  -------------------
###################################################################

if __name__ == "__main__":
    train_or_not = 0
    test_or_not =  1

    if train_or_not:
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.device_count())
        train_set = DatasetFromHdf5('./data/train_IMERG-Early_SM2RAIN_small.h5')  # creat data for training
        training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                          pin_memory=True, drop_last=True)  # put training data to DataLoader for batches
     
        train(training_data_loader)#, start_epoch=200)  # call train function (call: Line 53)

    if test_or_not:
        print("----------------------------testing-------------------------------")
        # test_real_visualization()

        # test_real()
        test_real1()
        test_real2()
        test_real3()
        test_real4()


   
   
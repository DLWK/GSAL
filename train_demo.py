from torch import optim
from losses import *
from data.dataloader import FJJ_Loader, FJJ_Loadertest
from data.dataloader2 import COVD19_Loader, COVD19_Loadertest
import torch.nn as nn
import torch
from models import ModelBuilder, SegmentationModule, SAUNet, VGG19UNet, VGG19UNet_without_boudary,VGGUNet
from torchvision import transforms
from utils.metric import *
from evaluation import *
from models.InfNet_Res2Net import  Inf_Net
from models.unet import UNet
from models.fcn import get_fcn8s
from models.UNet_2Plus import  UNet_2Plus
from models.AttU_Net_model  import AttU_Net
from models.BaseNet import CPFNet  
from models.cenet import CE_Net
from models.denseunet_model import DenseUnet
from models.F3net import F3Net
from models.LDF import LDF
from models.LDunet import LDUNet
# from SETR.transformer_seg import SETRModel
# from SETR.transformer_seg2 import NetS, NetC
from model.transformer_GSL import NetC, NetS
from data.dataloader import XSDataset, XSDatatest
import torch.nn.functional as F

import tqdm

def iou_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    loss_total= (wbce+wiou).mean()/wiou.size(0)
    return loss_total

# def iou_loss(pred, mask):
#     pred  = torch.sigmoid(pred)
#     inter = (pred*mask).sum(dim=(2,3))
#     union = (pred+mask).sum(dim=(2,3))
#     iou  = 1-(inter+1)/(union-inter+1)
#     return iou.mean()

# def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power)
#     lr = base_lr * (1-epoch/num_epochs)**power
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr






def test(testLoader,fold, nets, device):
    nets.to(device)
    sig = torch.nn.Sigmoid()
    nets.eval()
    with torch.no_grad():
         # when in test stage, no grad
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        count = 0
        for image, label  in tqdm.tqdm(testLoader):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # p1,p2,p3,p4= net(image)
            pred =nets(image)
            sig = torch.nn.Sigmoid()
            pred = sig(pred)
            # print(pred.shape)
            acc += get_accuracy(pred,label)
            SE += get_sensitivity(pred,label)
            SP += get_specificity(pred,label)
            PC += get_precision(pred,label)
            F1 += get_F1(pred,label)
            JS += get_JS(pred,label)
            DC += get_DC(pred,label)
            count+=1
        acc = acc/count
        SE = SE/count
        SP = SP/count
        PC = PC/count
        F1 = F1/count
        JS = JS/count
        DC = DC/count
        score = JS + DC
        
        return  acc, SE, SP, PC, F1, JS, DC, score
           

def train_net(nets, netc, device, train_data_path,test_data_path, fold, epochs=100, batch_size=4, lr=0.00001):
    isbi_train_dataset = COVD19_Loader(train_data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    test_dataset = COVD19_Loadertest(test_data_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1, 
                                               shuffle=False)
    
    # setup optimizer
    # beta1 = 0.5
    # optimizerG = optim.Adam(nets.parameters(), lr=lr, betas=(beta1, 0.999))
    # optimizerD = optim.Adam(netc.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.RMSprop(nets.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizerD = optim.RMSprop(netc.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
                    

    # criterion2 = nn.BCEWithLogitsLoss()
    #criterion3 = structure_loss()
    # criterion3 = BCEDiceLoss() 
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = BCEDiceLoss() 
    # criterion2 =LovaszHingeLoss()
    print('===> Starting training\n')
    best_loss = float('inf')
    result = 0
    # f = open('./finall_loss_unet'+str(fold)+'.csv', 'w')
    # f.write('epoch,loss'+'\n')
    for epoch in range(1, epochs+1):
        i=0
        nets.train()
        for image, mask, body, detail in train_loader:
            netc.zero_grad()

            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32) 
            # body = body.to(device=device, dtype=torch.float32) 
            # detail = detail.to(device=device, dtype=torch.float32)
            # edge = edge.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            # outb1, outd1, out1, outb2, outd2, out2 = net(image)
            # lossb1 = F.binary_cross_entropy_with_logits(outb1, body)
            # lossd1 = F.binary_cross_entropy_with_logits(outd1, detail)
            # loss1  = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)

            # lossb2 = F.binary_cross_entropy_with_logits(outb2, body)
            # lossd2 = F.binary_cross_entropy_with_logits(outd2, detail)
            # loss2  = F.binary_cross_entropy_with_logits(out2, mask) + iou_loss(out2, mask)
            # loss   = (lossb1 + lossd1 + loss1 + lossb2 + lossd2 + loss2)/2
            # p1,p2,p3,p4 = net(image)
            # loss1= iou_loss(p1,mask)
            # loss2= iou_loss(p2,body)
            # loss3= iou_loss(p3,detail)
            # loss4= iou_loss(p4,mask)
            # loss=loss1+loss2+loss3+loss4
            output = nets(image)
            output = F.sigmoid(output)
            output = output.detach() ### detach G from the network

            input_mask = image.clone()
            output_masked = image.clone()
            output_masked = input_mask * output
            # output_masked = output
            if cuda:
                output_masked = output_masked.cuda()

            target_masked = image.clone()
            target_masked = input_mask * mask
            # target_masked = mask
            if cuda:
                target_masked = target_masked.cuda()

            output_D = netc(output_masked)
            # print(output_D.shape)
            target_D = netc(target_masked)
            # print(target_D.shape)
            loss_D = 1 - torch.mean(torch.abs(output_D - target_D))
            loss_D.backward()
            optimizerD.step()

            ### clip parameters in D
            for p in netc.parameters():
                p.data.clamp_(-0.05, 0.05)
            #################################
            ### train Generator/Segmentor ###
            #################################
            nets.zero_grad()

            output = nets(image)
        
            output = F.sigmoid(output)

            loss_dice = iou_loss(output,mask)   ####修改

            output_masked = input_mask * output
            if cuda:
                output_masked = output_masked.cuda()

            target_masked = input_mask * mask
            if cuda:
                target_masked = target_masked.cuda()

            output_G = netc(output_masked)
            target_G = netc(target_masked)
            loss_G = torch.mean(torch.abs(output_G - target_G))
            loss_G_joint = loss_G + loss_dice
            loss_G_joint.backward()
            optimizerG.step()

            



            if(i % 4 == 0):

                print("\nEpoch[{}/{}]\tBatch({}/{}):\tBatch Dice_Loss: {:.4f}\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                            epoch, epochs, i, len(train_loader), loss_dice.item(), loss_G.item(), loss_D.item()))
            i+=1
          


       
        if epoch>0:
            acc, SE, SP, PC, F1, JS, DC, score=test(test_loader,fold, nets, device)
            if result < score:
                result = score
                # best_epoch = epoch
                torch.save(nets.state_dict(), './LDFUNet/COVD_19/New11_TSGANet_best_'+str(fold)+'.pth')
                with open("./LDFUNet/COVD_19/New11_TSGANet_"+str(fold)+".csv", "a") as w:
                    w.write("epoch="+str(epoch)+",acc="+str(acc)+", SE="+str(SE)+",SP="+str(SP)+",PC="+str(PC)+",F1="+str(F1)+",JS="+str(JS)+",DC="+str(DC)+",Score="+str(score)+"\n")


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fold = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True
    if cuda and not torch.cuda.is_available():

        raise Exception(' [!] No GPU found, please run without cuda.')
    
    # net = VGG19UNet(n_channels=1, n_classes=1)
    # net = UNet(n_channels=1, n_classes=1)
    # net =get_fcn8s(n_class=1)
    # net = UNet_2Plus(in_channels=1, n_classes=1)
    nets = NetS(patch_size=(32, 32), 
                    in_channels=1, 
                    out_channels=1, 
                    hidden_size=1024, 
                    num_hidden_layers=8, 
                    num_attention_heads=16, 
                    decode_features=[512, 256, 128, 64])

    netc = NetC(patch_size=(32, 32), 
                    in_channels=1, 
                    out_channels=1, 
                    hidden_size=1024, 
                    num_hidden_layers=8, 
                    num_attention_heads=16, 
                    decode_features=[512, 256, 128, 64])
   
    # net = Inf_Net()
    # net = AttU_Net(img_ch=1, output_ch=1)
    # net = CE_Net(num_classes=1, num_channels=1)
    # net = CPFNet()
    # net = VGGUNet(n_channels=1, n_classes=1)
    # net = VGG19UNet_without_boudary(n_channels=1, n_classes=1)
    # net =  DenseUnet(in_ch=1, num_classes=1)
    # net = LDUNet(n_channels=1, n_classes=1)
    # net = LDF()
    nets.to(device=device)
    netc.to(device=device)
    
    data_path = "/home/wangkun/COVD-19/train_512_"+str(fold)
    test_data_path = "/home/wangkun/COVD-19/test_512_"+str(fold)
    train_net(nets,netc, device, data_path,test_data_path, fold)


# by kun wang 
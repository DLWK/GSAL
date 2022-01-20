import glob
import numpy as np
import torch
import os
import time
import cv2
import csv
import tqdm
# from models import ModelBuilder, SegmentationModule, SAUNet, VGG19UNet,VGG19UNet_without_boudary
from models.unet import UNet
from models.fcn import get_fcn8s
from models.AttU_Net_model  import AttU_Net
from models.R2U_Net_model  import R2U_Net
from models.denseunet_model import DenseUnet
from models.cenet import CE_Net
from models.UNet_2Plus import  UNet_2Plus
from models.BaseNet import CPFNet 
# from models.vggunet import  VGGUNet
from MGmodels.mglnet import MGLNet

from model.transformer_GSL import NetC, NetS
import torch.nn.functional as F
#ablation

from medpy import metric
from evaluation import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from data.dataloader import XSDataset, XSDatatest





def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd





if __name__ == "__main__":




    fold=3
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net =Backbone(n_channels=3, n_classes=1)
    # net = UNet(n_channels=3, n_classes=1)
    # net =  UNet_3Plus(in_channels=1, n_classes=1)
    # net = SeResUNet(n_channels=1, n_classes=1)
    # net =  DilatedResUnet(n_channels=1, n_classes=1)
    # net =  SegNet(input_nbr=1,label_nbr=1)
    # net = UNet_2Plus(in_channels=3, n_classes=1)
    # net =  DenseUnet(in_ch=1, num_classes=1)
    # net = SceResUNet(n_channels=1, n_classes=1)
    # net = myChannelUnet(in_ch=1, out_ch=1)
    # net = ResUNet(n_channels=1, n_classes=1)
    # net =  DenseUnet(in_ch=1, num_classes=1)
    # net = UNet_2Plus(in_channels=1, n_classes=1)
    # net =get_fcn8s(n_class=1)
    # net = VGG19UNet_without_boudary(n_channels=1, n_classes=1)
    # net = R2U_Net(img_ch=1, output_ch=1)
    # net = VGGUNet(n_channels=1, n_classes=1)
    # net = CPFNet(nc=3)
    # net = CE_Net(num_classes=1, num_channels=3)
    # net = AttU_Net(img_ch=1, output_ch=1)       # 加载网络..........***************
    # 将网络拷贝到deivce中
    net = NetS(n_channels=3, n_classes=1)
    # net = MGLNet()
    # net = VGG19UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('/home/wangkun/data/LDFGAN/XS/GANet_best_'+str(fold)+'.pth', map_location=device))
    # net.load_state_dict(torch.load('/home/wangkun/data/LDFGAN/XS/Ablation/Aspp_Backbone_best_'+str(fold)+'.pth', map_location=device))
    # 测试模式
    # 
    # net.eval()
    # 读取所有图片路径
    # tests_path = glob.glob('/home/wangkun/shape-attentive-unet/data/test_96/image/*.jpg')
    # mask_path = "/home/wangkun/shape-attentive-unet/data/test_96/label/"
    # save_path = "/home/wangkun/shape-attentive-unet/data/test_96/MyNet-baseline/"

    # 
    # image_path  = "/home/wangkun/data/XS/Test/CVC-ClinicDB/images/"   Kvasir
    test_data_path = "/home/wangkun/data/XS/Test/Kvasir/"
   
    

    
    
    save_path = "/home/wangkun/data/XS/Test/CVC-ClinicDB/samBackbone/" 
    save_Pro_path = "/home/wangkun/data/XS/Test/CVC-ClinicDB/Prop/SegTrGAN/" 

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_Pro_path):
        os.mkdir(save_Pro_path)

    
    
         
    # 遍历素有图片
    test_dataset = XSDatatest(test_data_path)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False) 


   
    net.to(device)
    sig = torch.nn.Sigmoid()
    print('start test!')
    net.eval()
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
        dice=0
        jc=0
        hd=0
        asd=0
        # f = open('./test_time/CPFNet2.csv', 'w')
        # f.write('name,time'+'\n')
        for image, label, image_path in tqdm.tqdm(test_loader):
            # print(image)
            for test_path in tqdm.tqdm(image_path):
                
                name = test_path.split('/')[-1][0:-4]
                save_path1 = save_path + name+ ".png"
                save_path2 = save_Pro_path + name+ ".png"
                
            
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # pred,p1,p2,p3,p4,e= net(image)
            start_t = time.time()


            b,d,pred= net(image)
            end_t = time.time()

            cost_t =end_t-start_t
        

            pred = sig(pred)
            pred1 = np.array(pred.data.cpu()[0])[0]
            pred1[pred1 >= 0.5] = 255
            pred1[pred1 < 0.5] = 0
            img = pred1
            

            pred2 = np.array(pred.data.cpu()[0])[0]
          
            label2= label[0][0].cpu().numpy()
            label3= label.cpu().numpy()



        # 保存图片
            # cv2.imwrite(save_path1, img)
            # cv2.imwrite(save_path2,  pred2*255)   #保存概率图
            
            acc += get_accuracy(pred,label)
            SE += get_sensitivity(pred,label)
            SP += get_specificity(pred,label)
            PC += get_precision(pred,label)
            F1 += get_F1(pred,label)
            JS += get_JS(pred,label)
            DC += get_DC(pred,label)
           
       

           
            
            
            dice += metric.dc(label3, pred1) #####正确的表达
        
            jc += metric.jc(label3, pred1)
            hd += metric.binary.hd(pred2, label2)
            asd += metric.binary.asd(pred2, label2)
            count += 1
            # f.write(str(name)+","+str(cost_t)+"\n")
        
            



        acc = acc/count
        SE = SE/count
        SP = SP/count
        PC = PC/count
        F1 = F1/count
        JS = JS/count
        DC = DC/count
        dice=dice/count
        jc=jc/count
        hd=hd/count
        asd=asd/count

        score = JS + DC
        
    print('ACC:%.4f' % acc)
    print('SE:%.4f' % SE)
    print('SP:%.4f' % SP)
    print('PC:%.4f' % PC)
    print('F1:%.4f' % F1)
    print('JS:%.4f' % JS)
    print('DC:%.4f' % DC)

    print("**************************************")
    print('dice:%.4f' % dice)
    print('jc:%.4f' % jc)
    print('hd:%.4f' % hd)
    print('asd:%.4f' % asd)

    # f = open('./Ablation/vggu19+SAM.csv', 'w')
    # f.write('name,dice,iou,sen,pp'+'\n')
#     for test_path in tqdm.tqdm(tests_path):
        
#         name = test_path.split('/')[-1][0:-4]
       
    
#         mask = mask_path + name+".png"
       
#         image_path = image_path + name+".png"
        
#         # pred_name =  name+"_mask.png"
 
#         mask = cv2.imread(mask,0)

        
#         mask = torch.from_numpy(mask).cuda()
        


#         mask  = mask  / 255
     
       

#         # save_res_path = save_path+name + '_res.jpg'
#         save_mask_path = save_path+ name + '.png'

#         # 读取图片
#         img = cv2.imread(image_path ,1)
#         img = cv2.resize(img,(352,352))
#         mask = cv2.resize(mask,(352,352), interpolation=cv2.INTER_NEAREST)
#         # 转为灰度图
#         # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         # 转为batch为1，通道为1，大小为96*96的数组
#         img = img.reshape(1, 3, img.shape[0], img.shape[1])
#         # 转为tensor
#         img_tensor = torch.from_numpy(img)
#         # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
#         img_tensor = img_tensor.to(device=device, dtype=torch.float32)
#         # 预测
#         # pred = net(img_tensor)
#         e, pred = net(img_tensor)
#         sig = torch.nn.Sigmoid()
#         pred = sig(pred)
       

#         # 提取结果
#         pred1 = np.array(pred.data.cpu()[0])[0]
#         # # 处理结果
#         pred1[pred1 >= 0.5] = 255
#         pred1[pred1 < 0.5] = 0
#         img = pred1
#         # 保存图片
#         cv2.imwrite(save_mask_path, img)
        
#         # hd_s = metric.hd(mask, pred, voxelspacing= 0.3515625)
#         # f.write(name+","+str(dice_s)+","+str(iou_s)+","+str(sen_s)+","+str(ppv_s)+"\n")
#         acc += get_accuracy(pred,mask)
#         SE += get_sensitivity(pred,mask)
#         SP += get_specificity(pred,mask)
#         PC += get_precision(pred,mask)
#         F1 += get_F1(pred,mask)
#         JS += get_JS(pred,mask)
#         DC += get_DC(pred,mask)
#         count+=1
#     acc = acc/count
#     SE = SE/count
#     SP = SP/count
#     PC = PC/count
#     F1 = F1/count
#     JS = JS/count
#     DC = DC/count
# # hd_score = hd/count
    # print('ACC:%.4f' % acc)
    # print('SE:%.4f' % SE)
    # print('SP:%.4f' % SP)
    # print('PC:%.4f' % PC)
    # print('F1:%.4f' % F1)
    # print('JS:%.4f' % JS)
    # print('DC:%.4f' % DC)
   





# python<predict.py>sce2.txts
#by kun wang
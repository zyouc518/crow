import sys
sys.path.insert(0, '.')
sys.path.append('../../')
#from PCBModel_concat_norelu import PCBModel as Model
from PCBModel_concat_res101_norelu import PCBModel as Model
from torch.nn.parallel import DataParallel
from utils import set_devices
from utils import load_state_dict
from torch.utils.data import Dataset, DataLoader
import torch
from PreProcessImage import PreProcessIm
import numpy as np
from PIL import Image
from torch.autograd import Variable
from distance import normalize
import os
import cv2
import torch._utils
import time
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class ImageListDataset(Dataset):
    def __init__(self, list_name):
        if isinstance(list_name, basestring):
            self.img_list = open(list_name).readlines()
        else:
            self.img_list = list_name
        #print self.img_list
        self.pre_process_im = PreProcessIm(
            prng = np.random, 
            resize_h_w = (384, 128), 
            im_mean = [0.486, 0.459, 0.408],
            im_std = [0.229, 0.224, 0.225], 
            mirror_type=None,
            batch_dims='NCHW'
            )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx].strip()
#        im = np.asarray(Image.open(img_name))
        #img_name = img_name.replace(" ", "\\ ")
        im = cv2.imread("/data1/home/jinghaofeng/youchenz/crow/toxuemei/temp2/" + img_name)
        #print img_name, im.shape
        if im is None:
            print (img_name, 'image open error')
        im = im[:, :, ::-1 ]
        '''
        im1 = cv2.imread(img_name)
        #im1 = im1[:, :, [2,1,0]]
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        for m, m1 in zip(im, im1):
            for mm, mm1 in zip(m, m1):
                print np.mean(mm-mm1), mm, mm1, 'fooo'
        '''
        im, _ = self.pre_process_im(im)
        return im, img_name

def extract_feature(
    ckpt_file = 'models/ckpt.pth', 
    img_list = '/data1/home/jinghaofeng/youchenz/crow/toxuemei/temp2.txt/',
    resize_h_w = (384, 128), 
    batch_size = 128, 
    output_list = None
    ):
    #########
    # Model #
    #########
    
    #TVT, TMO = set_devices(sys_device_ids)
    model = Model(
      last_conv_stride = 1,
      num_stripes = 1,
      local_conv_out_channels = 768,
      num_classes=0
    )
    # Model wrapper
    
    # May Transfer Model to Specified Device.
    #TMO([model])
    
    #####################
    # Load Model Weight #
    #####################
    
    # To first load weights to CPU
    map_location = (lambda storage, loc: storage)
    ckpt = torch.load(ckpt_file, map_location=map_location)
    
    if 'state_dicts' in ckpt:
        state_dict = ckpt['state_dicts'][0]
    else:
        state_dict = ckpt['state_dict']

    #print loaded
    load_state_dict(model, state_dict)
    model = model.cuda()
    model = DataParallel(model)
    
    dataset = ImageListDataset(img_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    model.eval()
    
    ret = []
    ret_input = [] 
    #f = open(output_list, 'w')
    for i, (im, img_name) in enumerate(dataloader):
    #    print i
        im_out = im.float()
        im = Variable(im.float())
        start = time.clock()
        feats = model(im) 
        elapsed = (time.clock() - start)
        print ('time used', elapsed)
        feats = [lf.data.cpu().numpy() for lf in feats]
        feats = np.concatenate(feats, axis=1)
        feats = normalize(feats, axis = 1)
        if i % 50 == 0: print (i)
#        ret.append(feats)
        for feat, n in zip(feats, img_name):
            name = n.strip()
            string = ''
            string += name + ' '
            for ff in feat:
                string += str(ff) + ' '
            #feat = feat.reshape(768, -1, -1)
            if i % 50 == 0: 
                print (n)
                print (feat.shape)
            np.save("/data1/home/jinghaofeng/youchenz/crow/toxuemei/temp2_res/" + n, feat)
            #f.write(string + '\n')


#    ret = np.concatenate(ret, axis = 0)
#    return ret#, ret_input
if __name__ == '__main__' :
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3,4,5,6,7'
    #ckpt_file = 'logs/sdk_0.1_ckpt.pth'
    #ckpt_file = 'logs/run45_37ep_ckpt.pth'
    #ckpt_file = 'logs/pretrain_model/run176_baili_28ep_ckpt.pth'
    #ckpt_file = 'logs/pretrain_model/run72_10ep_ckpt.pth'
    #ckpt_file = 'logs/pretrain_model/task_2019-01-05-11-19-31_n43x1kw7_ep30.pth'
    #ckpt_file = 'logs/pretrain_model/run61_1ep_step1000_ckpt.pth'
    #ckpt_file = 'logs/pretrain_model/task_2019-01-11-10-56-17_2kljca4r_ep4.pth'
    #ckpt_file = 'logs/pretrain_model/task_2019-01-11-15-44-49_bx91yph7_ep23.pth'
    #ckpt_file = 'logs/pretrain_model/task_2019-01-18-10-08-57_6zekw5la_ep50.pth'
    #ckpt_file = 'logs/pretrain_model/run177_11ep_step4000_ckpt.pth'
    #ckpt_file = 'logs/pretrain_model/task_2019-03-05-10-19-29_ghfpcoyj_ep5.pth'
    
    
    
    ##############old
    #ckpt_file = 'task_2019-03-27-20-36-26_64yobeaw_ep12.pth'
    ################new
    ckpt_file = 'task_2019-04-20-17-01-49_ddyx0dbw_ep0.pth'
    #ckpt_file = 'logs/pretrain_model/task_2019-03-05-10-19-29_ghfpcoyj_ep15.pth'
    #ckpt_file = 'logs/pretrain_model/task_2018-12-27-17-18-37_y2drsokr_ep30.pth'
    #img_list = '/data2/sevjiang/beyond-part-models/data/reid_samples/lena_list'
    #img_list = './file_list/g_list'
    #img_list = './merge/merge_gen_q.list'
    #img_list = './file_list/0418_gallery.list'
    #img_list = './file_list/0419_mergeq_gen.list'
    img_list = '/data1/home/jinghaofeng/youchenz/crow/toxuemei/temp2.txt'
    #img_list = './file_list/0427_merge_q_gen.list'
    #img_list = './file_list/test_0418.list'
    #img_list = './file_list/0419_query.list'
    #img_list = './file_list/gallery.list'
    #img_list = 'int8/img_list'
    #img_list = 'data/1231_full/gallery_list'
    #img_list = '/data2/sevjiang/beyond-part-models/data/sampled_list_2000.txt'
    #img_list = '/data2/sevjiang/beyond-part-models/data/badcase_q_g_1210/gallery_list' 
    #img_list = 'data/1206/img_list'
    #img_list = 'data/0109/6488789133858181123/gallery_list'
    #img_list = 'data/1206/img_list'
    #img_list = '/data2/sevjiang/beyond-part-models/data/match_miss_0304/11572830426407362843/query_list'
    #img_list = '/data2/sevjiang/beyond-part-models/data/match_fail_coyj_list'
    #img_list = 'data/match/0305_miss_2_list'
    #img_list = 'data/wanda_online_badcase/0307/gallery_list'
    #img_list = 'data/match/0305_miss_1_2_3_list'
    resize_h_w = (384, 128)
    #output_list = 'data/0103_1600/feats_query_run177'
    #output_list = 'data/0109/6488789133858181123/feats_gallery_ca4r'
    #output_list = 'data/1231_full/feats_gallery'
    #output_list = '/data2/sevjiang/beyond-part-models/data/feats_0305_miss_2_yph7'
    #output_list = 'data/match/feats_0305_miss_1_2_3_beaw'
    #output_list = './feat_list/feats_merge_gen_q'
    #output_list = './feat_list/0418_feats_gallery'
    #output_list = './feat_list/0419_feats_merge_query'
    output_list = '/data1/home/jinghaofeng/youchenz/crow/toxuemei/temp2_res'
    #output_list = './feat_list/new_feat_0427_feats_merge_q_gen'
    #output_list = './feat_list/newfeat_test_0418_feats_merge_q_gen'
    
    #output_list = './feat_list/feats_gallery'
    batch_size = 8
  
    
    feats = extract_feature(ckpt_file = ckpt_file, 
        img_list = img_list, 
        resize_h_w = resize_h_w, 
        batch_size = batch_size, 
        output_list = output_list)
    
    '''
    img_list = open(img_list).readlines()
    print "img_list length:"+str(len(img_list))
    f = open(output_list, 'w')
    print feats.shape
    for feat, name in zip(feats, img_list):
        print 'write feature', name
        name = name.strip()
        string = ''
        string += name + ' '
        for ff in feat:
            string += str(ff) + ' '
        f.write(string + '\n')
    '''
   # for im , name in zip(pre_im, img_list):
    #    name = name.strip()
     #   print im
      #  im = im.flatten()
       # print im
       # print output_pre_im + '/' + os.path.basename(name).replace('.jpg', '')
       # np.savetxt(open(output_pre_im + '/' + os.path.basename(name).replace('.jpg', ''), 'w'), im, fmt = '%.18f')
       # print im.shape, name

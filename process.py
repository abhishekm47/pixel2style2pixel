import torch
import torchvision.transforms as transforms

from argparse import Namespace

import numpy as np
import cv2
from PIL import Image
import sys
sys.path.append(".")
sys.path.append("..")

from alignment import read_cv2_img, dlib_detect_face, face_recover, tensor2im
from models.psp import pSp
import time
import pprint


try:
    from hprams import Pixel2Style2PixelConfig as config
except:
    class Pixel2Style2PixelConfig:
        padding=0.25
        size=256
        moving=0.0
        model_path = "service/scripts/pixel2style2pixel/pretrained_models/psp_celebs_super_resolution.pt"
        resize_factors = 1
        output_size = 1024
        resize_outputs = False
        
    config = Pixel2Style2PixelConfig


def load_model(model_path, network):
    test_opts  = {
        'resize_factors':1,
        'resize_outputs':False
    }
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(test_opts)
    print("-----------Pixel2Style2Pixel-Inf-Config------------")
    pprint.pprint(opts)
    print("-----------****************************------------")
    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024

    opts['resize'] = False
    
    opts = Namespace(**opts)
    model = network(opts)
    model.eval()
    model.cuda()
    print('Model successfully loaded! ✅')
    
    return model



model = load_model(config.model_path, pSp)

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch



def psp_forward(frame_path, landmark, latent_mask = None):
    #image transformer
    img_transforms =transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    #read image
    img = read_cv2_img(frame_path)
    #align image
    img_aligned, M = dlib_detect_face(img, landmark)
    #transform image
    transformed_image = img_transforms(img_aligned)
    with torch.no_grad():
        tic = time.time()
        result_image = run_on_batch(transformed_image.unsqueeze(0), model, latent_mask)[0]
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))
        
    output_image = tensor2im(result_image)
    output_img = np.clip((np.transpose(output_image, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    rec_img = face_recover(output_img, M * 4, img)
    return output_img, rec_img
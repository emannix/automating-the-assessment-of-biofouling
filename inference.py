
from torchvision import transforms
import torch
from PIL import Image
from pdb import set_trace as pb
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

images = ['000015-gbEGjSAk6r.jpg', '000018-dqrONtdjbt.jpg', 
    '000022-x6lwoJexxL.jpg', '000112-Eg0gnVJjIm.jpg']

model_save_path = 'dinov2_vitB14/'


img_orig = [Image.open('sample_images/'+x) for x in images]
img = [transform(x) for x in img_orig]
img = torch.stack(img)

# =====================================================================

input_sample_run = img.cpu().numpy()

import onnxruntime
ort_session = onnxruntime.InferenceSession(model_save_path+'model.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider', 'TensorrtExecutionProvider'])
# input_name = ort_session.get_inputs()[0].name
ort_inputs = {'input': input_sample_run}
ort_outs = ort_session.run(None, ort_inputs)
# =====================================================================


"""
The return object ort_outs will have the following outputs

ort_outs[0] 
# The global class token from the underling DINOv2 ViT-B/14 backbone

ort_outs[1] 
# The output confidence of an underwater structure being detected.
# [Conf(Hull present), Conf(Hull not present), Conf(Non-informative patch present)]

ort_outs[2] 
# The output confidence of biofouling being present.

ort_outs[3] 
# The proportion of the image covered by patches with biofouling.

ort_outs[4] 
# The output confidence of paint damage being present.

ort_outs[5] 
# The proportion of the image covered by patches with paint damage.

ort_outs[6] 
# The confidence niche areas are present in the image.
# [Conf(Grating present), Conf(Plug/Discharge/Ropeguard present), Conf(Anode present)]

The next set of outputs will be 16x16 patch layers which correspond to the local predictions.
The output of their last dimension will be

ort_outs[7] 
# The output confidence of an underwater structure being detected.
# [Conf(Hull present), Conf(Hull not present), Conf(Non-informative patch present)]

ort_outs[8] 
# The confidence niche areas are present in the image.
# [Conf(No niche area), Conf(Grating present), Conf(Plug/Discharge/Ropeguard present), Conf(Anode present)]

ort_outs[9] 
# The output confidence of biofouling being present.
# [Conf(No fouling), Conf(Fouling)]

ort_outs[10] 
# The output confidence of biofouling being present.
# [Conf(No paint damage), Conf(Paint damage)]

# we use the following thresholds for detecting
"underwater_structure_detected": 0.25,
"fouling_present": 0.25,
"paint_damaged": 0.50,
"niche_grating": 0.25,
"niche_discharge_hole": 0.25,
"niche_anode": 0.25
"""

# visualizing outputs

threshold = 0.25
alpha = 0.5

fouling_map = torch.tensor(ort_outs[9][:,:,:,1:])
fouling_map = fouling_map.permute(0,3,1,2)

import torch.nn.functional as F
import torchvision
resized_tensor = F.interpolate(fouling_map, size=img_orig[0].size, mode='bilinear', align_corners=False)

for i in range(len(img_orig)):
    img_overlay = torchvision.transforms.ToPILImage()(resized_tensor[i]).convert("RGBA")
    img_overlay = np.array(img_overlay)
    img_overlay[:,:,0] = 0
    img_overlay[:,:,3] = int(255 * alpha)*(resized_tensor[i] > threshold)
    img_overlay = Image.fromarray(img_overlay)
    combined_image = Image.alpha_composite(img_orig[i].convert("RGBA"), img_overlay).show()



pb()
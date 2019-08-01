import argparse

import cv2
import numpy as np
import torch
from torch.autograd import Variable

TEST_IMG_PATH = '/media/data/users/master/2018/zhanghan/data/voc2012/val'
VAL_LABEL_PATH = '/media/data/users/master/2018/zhanghan/data/voc2012/val_labels.txt'
RESULT_IMG_PATH = 'test_results'


class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.resnet_layer._modules.items():
            x = module(x)
            if (name == '5'):
                low_map = x
            elif (name == '6'):
                mid_map = x

        x = self.model.merge(x, low_map, mid_map)
        x.register_hook(self.save_gradient)
        outputs += [x]

        return outputs, x


class ModelOutputs():
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(
            self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = self.model.classifier.avg(output)
        output = output.view(output.size(0), -1)
        output = self.model.classifier.fc(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(
        np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, cam, img_name, result_img_path):
    # heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    # cam =0.5*cam+0.5*img
    print(os.path.join(result_img_path, img_name))
    cv2.imwrite(os.path.join(result_img_path, img_name), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index, height, width):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        output_cam = []
        for idx in index:
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][idx] = 1
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            
            if self.cuda:
                one_hot = torch.sum(one_hot.cuda() * output)
            else:
                one_hot = torch.sum(one_hot * output)

            self.model.zero_grad()
            
			# one_hot.backward(retain_variables=True)
            one_hot.backward(retain_graph=True)
            grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
            
            target = features[-1]
            target = target.cpu().data.numpy()[0, :]
            weights = np.mean(grads_val, axis=(2, 3))[0, :]
            cam = np.zeros(target.shape[1:], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

            if np.sum(cam > 0) == 0:
                cam = cam * -1

            cam = np.maximum(cam, 0)
            
            cam = cv2.resize(cam, (width, height))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cam[cam > 0.7] = 1
            cam[cam < 1] = 0
            cam = cam.astype(np.uint8)

            output_cam.append((np.sum(cam), cam))

        return output_cam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    from res50_pm import get_cam_model

    """ python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

    args = get_args()
    gen_num = 2913

    model = get_cam_model(8)
    print(model)
    grad_cam = GradCam(model=model,
                       target_layer_names="upsample", use_cuda=args.use_cuda)
    import os

    with open(VAL_LABEL_PATH, 'r') as f:
        lines = f.readlines()

    img_labels = {}
    for line in lines:
        name_label_list = line.strip().split()
        img_name = name_label_list[0]
        labels = name_label_list[1:]
        img_labels[img_name] = [i for i, e in enumerate(labels) if e == '1']

    for idx, img_name in enumerate(os.listdir(TEST_IMG_PATH)):
        if idx >= gen_num:
            break
        img_path = os.path.join(TEST_IMG_PATH, img_name)
        img = cv2.imread(img_path, 1)
        h, w, c = img.shape
        input_img = img
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        img_name_nb = img_name[:-4]

        mask = grad_cam(input, img_labels[img_name_nb], h, w)
        label = img_labels[img_name_nb]
        mask_label = list(zip(mask, label))
        mask_label.sort(reverse=True)

        # for i in range(0, len(mask)):
        #     for j in range(i, len(mask)):
        #         if(np.sum(mask[i]) < np.sum(mask[j])):
        #             max_mask = mask[j]
        #             mask[j] = mask[i]
        #             mask[i] = max_mask

        #             max_label = label[j]
        #             label[j] = label[i]
        #             label[i] = max_label
        
        result_labelmap = np.zeros((h, w), dtype=np.uint8)
        for cam_label in mask_label:
            size_cam, label = cam_label
            size, cam = size_cam
            result_labelmap[cam==1] = label + 1

		# result_labelmap[mask[0] > 0] = (label[0]+1)
        # for i, cam in enumerate(mask):
        #     if(i > 0):
        #         result_cam[mask[i] > 0] = (label[i]+1)

        cam_img_name = '{}'.format(img_name[:-4])
        result_img_path = RESULT_IMG_PATH
        print (cam_img_name)
        np.save(os.path.join(result_img_path, cam_img_name),result_labelmap)
        # cam_img_name = '{}.png'.format(img_name[:-4])
        # result_segmap = label2img(result_labelmap)
        # result_segmap = dcrf(input_img,result_segmap)
        #
        # show_cam_on_image(input_img, result_segmap, cam_img_name,
        #                   result_img_path=RESULT_IMG_PATH)

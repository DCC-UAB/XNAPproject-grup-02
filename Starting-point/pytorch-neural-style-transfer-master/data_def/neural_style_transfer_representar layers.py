import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results
#API key: 977562db4dc023790ad117d9a62ec93e4792b1a6
# API MIMI; 70ea322aed851116c7091b20c2f61b7e2e47e88d
# python3 neural_style_transfer.py --content_img_name figures.jpg --style_img_name impresionismo.jpg

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse
import wandb
import matplotlib.pyplot as plt

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Estableciendo los máximos observados
    max_content_loss = 15000.0
    max_style_loss = 15000000.0
    max_tv_loss = 70000000.0
    
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    # Normalizando las pérdidas
    normalized_content_loss = content_loss / max_content_loss
    normalized_style_loss = style_loss / max_style_loss
    normalized_tv_loss = tv_loss / max_tv_loss
    
    total_loss = config['content_weight'] * normalized_content_loss + config['style_weight'] * normalized_style_loss + config['tv_weight'] * normalized_tv_loss

    wandb.log({
        "content_loss": content_loss,
        "style_loss": style_loss,
        "tv_loss": tv_loss,
        "total_loss": total_loss
        })
    
    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    # Define a function to normalize and save the averaged feature maps as images
    def save_averaged_feature_map(feature_maps, layer_name, output_dir, name):
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate the mean of the feature maps along the channel dimension
        averaged_feature_map = feature_maps.mean(dim=1).squeeze()
        
        # Normalize the averaged feature map to range [0, 1]
        averaged_feature_map -= averaged_feature_map.min()
        averaged_feature_map /= averaged_feature_map.max()
        
        # Convert the tensor to a numpy array
        averaged_feature_map_np = averaged_feature_map.cpu().numpy()
        
        # Plot the averaged feature map
        plt.imshow(averaged_feature_map_np, cmap='viridis')
        plt.axis('off')
        
        # Save the averaged feature map as an image
        plt.savefig(os.path.join(output_dir, f"{name}_{layer_name}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Output directory
    output_dir = 'feature_maps_output'

    # Save the averaged feature map for each layer
    for layer_name in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']:
        output_dir = 'feature_maps_output_style'
        feature_maps = getattr(style_img_set_of_feature_maps, layer_name)
        save_averaged_feature_map(feature_maps, layer_name, output_dir, config['style_img_name'])
        output_dir = 'feature_maps_output_content'
        feature_maps = getattr(content_img_set_of_feature_maps, layer_name)
        save_averaged_feature_map(feature_maps, layer_name, output_dir, config['content_img_name'])

    return dump_path


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data_def')
    content_images_dir = os.path.join(default_resource_dir, 'content')
    style_images_dir = os.path.join(default_resource_dir, 'style')
    output_img_dir = os.path.join(default_resource_dir, 'output')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    # sorted so that the ones on the top are more likely to be changed than the ones on the bottom
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, help="height of content and style images", default=400)

    parser.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    parser.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    parser.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'], default='adam')
    parser.add_argument("--model", type=str, choices=['vgg16', 'vgg19'], default='vgg19')
    parser.add_argument("--init_method", type=str, choices=['random', 'content', 'style'], default='content')
    parser.add_argument("--num_of_iterations", type=int, help="num of iterations", default=3000)
    parser.add_argument("--learning_rate", type=int, help="learning_rate", default=1e1)
    parser.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)
    args = parser.parse_args()

    # some values of weights that worked for figures.jpg, vg_starry_night.jpg (starting point for finding good images)
    # once you understand what each one does it gets really easy -> also see README.md

    # lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
    # lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
    # lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

    # adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
    # adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
    # adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

    # just wrapping settings into a dictionary
    optimization_config = dict()
    for arg in vars(args):
        optimization_config[arg] = getattr(args, arg)
    optimization_config['content_images_dir'] = content_images_dir
    optimization_config['style_images_dir'] = style_images_dir
    optimization_config['output_img_dir'] = output_img_dir
    optimization_config['img_format'] = img_format

    for style_image in os.listdir(style_images_dir):
        for content_image in os.listdir(content_images_dir):
            optimization_config['content_img_name'] = content_image
            optimization_config['style_img_name'] = style_image
            results_path = neural_style_transfer(optimization_config)


    # wandb.login(key="977562db4dc023790ad117d9a62ec93e4792b1a6")

    # run = wandb.init(
    # project="Style Transfer",
    # notes="",
    # tags=[f"content_image: {optimization_config['content_img_name']}", 
    #       f"style_image: {optimization_config['style_img_name']}",
    #       f"content_weight: {optimization_config['content_weight']}",
    #       f"style_weight: {optimization_config['style_weight']}",
    #       f"tv_weight: {optimization_config['tv_weight']}",
    #       f"optimizer: {optimization_config['optimizer']}",
    #       f"model: {optimization_config['model']}",
    #       f"saving_freq: {optimization_config['saving_freq']}",
    #       f"learning_rate: {optimization_config['learning_rate']}" 
    #     ]
    # )

    # config = wandb.config
    # config.learning_rate = optimization_config["learning_rate"]
    # config.epochs = optimization_config["num_of_iterations"]

    # original NST (Neural Style Transfer) algorithm (Gatys et al.)
    

    # uncomment this if you want to create a video from images dumped during the optimization procedure
   # create_video_from_intermediate_results(results_path, img_format)

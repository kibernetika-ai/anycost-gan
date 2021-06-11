import argparse
import os

import cv2
import numpy as np
import torch

import models
from models import dynamic_channel

n_style_to_change = 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--image')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output')

    return parser.parse_args()


def get_vector(device='cuda', n_styles=18):
    return torch.randn([1, n_styles, 512], device=device)


def deform_vector(old_vector, direction_dict, direction_values, max_value=0.6):
    edited_code = old_vector.clone()
    for direction_name, value in direction_values.items():
        edited_code[:, :n_style_to_change] = (
            edited_code[:, :n_style_to_change]
            + value * direction_dict[direction_name] / 100 * max_value
        )

    return edited_code


def get_new_face(gen, input_kwargs, vector=None, device='cuda', get_styles=False):
    if vector is None:
        vector = get_vector(device)
    input_kwargs['styles'] = vector
    input_kwargs['return_styles'] = get_styles
    with torch.no_grad():
        # scale = 2.0 * float(i_scale) / (n_steps - 1) - 1.
        img, styles = gen(**input_kwargs)
        img = img.cpu().numpy()[0]
        if get_styles:
            return to_img(img), styles
        return to_img(img)


def to_img(img):
    return (img.transpose([1, 2, 0]) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)


def show(img, direction_idx, direction_i):
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    bar = np.zeros([100, small.shape[1], 3], dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    columns = 4
    for k, v in direction_idx.items():
        color = (200, 200, 200) if k != direction_i else (0, 250, 0)
        x = (k % columns) * 120
        y = 20 + 25 * (k // columns)
        cv2.putText(
            bar,
            v,
            (x, y),
            font,
            0.6,
            color, thickness=1, lineType=cv2.LINE_AA
        )
    pic = np.vstack([small, bar])
    cv2.imshow('Face', pic[:, :, ::-1])


def main():
    args = parse_args()
    g_path = args.model

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    anycost_channel = 1.0
    anycost_resolution = 1024
    pretrained_type = 'generator'  # choosing from ['generator', 'encoder', 'boundary']
    config_name = 'anycost-ffhq-config-f'  # replace the config name for other models
    gen = models.get_pretrained(pretrained_type, config=config_name).to(device)
    encoder = models.get_pretrained('encoder', config=config_name).to(device)
    # mean_latent = gen.mean_style(10000)
    boundaries = models.get_pretrained('boundary', config_name)

    '''
    possible keys:
    ['00_5_o_Clock_Shadow', '01_Arched_Eyebrows', '02_Attractive', '03_Bags_Under_Eyes', '04_Bald', '05_Bangs',
     '06_Big_Lips', '07_Big_Nose', '08_Black_Hair', '09_Blond_Hair', '10_Blurry', '11_Brown_Hair', '12_Bushy_Eyebrows',
     '13_Chubby', '14_Double_Chin', '15_Eyeglasses', '16_Goatee', '17_Gray_Hair', '18_Heavy_Makeup', '19_High_Cheekbones',
     '20_Male', '21_Mouth_Slightly_Open', '22_Mustache', '23_Narrow_Eyes', '24_No_Beard', '25_Oval_Face', '26_Pale_Skin',
     '27_Pointy_Nose', '28_Receding_Hairline', '29_Rosy_Cheeks', '30_Sideburns', '31_Smiling', '32_Straight_Hair',
     '33_Wavy_Hair', '34_Wearing_Earrings', '35_Wearing_Hat', '36_Wearing_Lipstick', '37_Wearing_Necklace',
     '38_Wearing_Necktie', '39_Young']
    '''

    direction_map = {
        'smiling': '31_Smiling',
        'young': '39_Young',
        'wavy hair': '33_Wavy_Hair',
        'gray hair': '17_Gray_Hair',
        'blonde hair': '09_Blond_Hair',
        'eyeglass': '15_Eyeglasses',
        'mustache': '22_Mustache',
        'bald': '04_Bald',
        'attractive': '02_Attractive',
        'big nose': '07_Big_Nose',
        'big lips': '06_Big_Lips',
        'male': '20_Male',
    }
    direction_idx = {k: v for k, v in zip(range(len(direction_map)), direction_map)}

    direction_dict = dict()
    direction_i = 0
    direction_values = {k: 0.0 for k in direction_map}
    direction_step = 5.
    for k, v in direction_map.items():
        direction_dict[k] = boundaries[v].view(1, 1, -1).to(device)

    vector = get_vector(device_str)
    input_kwargs = {
        'styles': vector,
        'noise': None,
        'randomize_noise': False,
        'input_is_style': True
    }

    dynamic_channel.set_uniform_channel_ratio(gen, 1)  # set channel
    gen.target_res = 1024  # set resolution
    # out, _ = gen(...)  # generate image
    # dynamic_channel.reset_generator(gen)  # restore the generator

    if args.image:
        img = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

        for_encoder = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        for_encoder = ((for_encoder.astype(np.float32) - 127.5) / 127.5).transpose([2, 0, 1])
        vector = encoder(torch.tensor(for_encoder).unsqueeze(0).to(device))
        img = get_new_face(gen, input_kwargs, vector=vector)
    else:
        img = get_new_face(gen, {}, vector=vector)
    if args.output:
        print(f'Saved to {args.output}')
        cv2.imwrite(args.output, img[:, :, ::-1])

    if args.show:
        show(img, direction_idx, direction_i)

    if not args.interactive and args.show:
        cv2.waitKey(0)
    elif args.interactive and args.show:
        while True:
            key = cv2.waitKey(1)
            if key in [27, ord('q')]:
                break
            if key == ord('n'):
                # next direction
                direction_i += 1
                direction_i = direction_i % len(direction_dict)
            if key == ord('['):
                direction_name = direction_idx[direction_i]
                direction_values[direction_name] -= direction_step
                new_vector = deform_vector(
                    vector,
                    direction_dict,
                    direction_values
                )

                img = get_new_face(gen, input_kwargs, vector=new_vector)
            if key == ord(']'):
                direction_name = direction_idx[direction_i]
                direction_values[direction_name] += direction_step
                new_vector = deform_vector(
                    vector,
                    direction_dict,
                    direction_values
                )

                img = get_new_face(gen, input_kwargs, vector=new_vector)
            if key == 32:
                direction_values = {k: 0.0 for k in direction_map}

                style_vec = get_vector(device_str, n_styles=1)
                img, vector = get_new_face(gen, {'styles': style_vec}, vector=style_vec, get_styles=True)
                img = get_new_face(gen, input_kwargs, vector=vector)

            show(img, direction_idx, direction_i)


if __name__ == '__main__':
    main()

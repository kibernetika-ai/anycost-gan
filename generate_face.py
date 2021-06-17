import argparse
import logging
import time
import threading
import uuid

import cv2
import numpy as np
import torch

import models
from models import dynamic_channel

n_style_to_change = 12
LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator')
    parser.add_argument('--encoder')
    parser.add_argument('--boundary')
    parser.add_argument('--image')
    parser.add_argument('--scale', type=float, default=0.0)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output')

    return parser.parse_args()


def to_img(img):
    return (img.transpose([1, 2, 0]) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)


def show(img, direction_idx, direction_i):
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    bar = np.zeros([small.shape[0], 450, 3], dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    rows = 20
    for k, v in direction_idx.items():
        color = (200, 200, 200) if k != direction_i else (0, 250, 0)
        y = 20 + (k % rows) * 25
        x = 20 + 180 * (k // rows)
        cv2.putText(
            bar,
            v,
            (x, y),
            font,
            0.6,
            color, thickness=1, lineType=cv2.LINE_AA
        )
    pic = np.hstack([small, bar])
    cv2.imshow('Face', pic[:, :, ::-1])


class CachedVector:
    def __init__(self, vector, time):
        self.vector = vector
        self.time = time


class FaceGen:
    def __init__(self, config_name='anycost-ffhq-config-f', gen_path=None, enc_path=None,
                 bound_path=None):
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)
        self.device = device
        self.gen = models.get_pretrained('generator', config=config_name, path=gen_path).to(device)
        self.encoder = models.get_pretrained('encoder', config=config_name, path=enc_path).to(device)
        # mean_latent = gen.mean_style(10000)
        self.boundaries = models.get_pretrained('boundary', config_name, path=bound_path)
        self.keep_cache_sec = 3600
        self.cache = {}
        self.lock = threading.Lock()
        self.resolutions = [128, 256, 512, 1024]
        self.channel_ratio = [0.25, 0.5, 0.75, 1]

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
            '5_o_clock_shadow': '00_5_o_Clock_Shadow',
            'arched_eyebrows': '01_Arched_Eyebrows',
            'attractive': '02_Attractive',
            'bags_under_eyes': '03_Bags_Under_Eyes',
            'bald': '04_Bald',
            'bangs': '05_Bangs',
            'big_lips': '06_Big_Lips',
            'big_nose': '07_Big_Nose',
            'black_hair': '08_Black_Hair',
            'blonde_hair': '09_Blond_Hair',
            'blurry': '10_Blurry',
            'brown_hair': '11_Brown_Hair',
            'bushy_eyebrows': '12_Bushy_Eyebrows',
            'chubby': '13_Chubby',
            'double_chin': '14_Double_Chin',
            'eyeglasses': '15_Eyeglasses',
            'goatee': '16_Goatee',
            'gray_hair': '17_Gray_Hair',
            'heavy_makeup': '18_Heavy_Makeup',
            'high_cheekbones': '19_High_Cheekbones',
            'male': '20_Male',
            'mouth_slightly_open': '21_Mouth_Slightly_Open',
            'mustache': '22_Mustache',
            'narrow_eyes': '23_Narrow_Eyes',
            'no_beard': '24_No_Beard',
            'oval_face': '25_Oval_Face',
            'pale_skin': '26_Pale_Skin',
            'pointy_nose': '27_Pointy_Nose',
            'receding_hairline': '28_Receding_Hairline',
            'rosy_cheeks': '29_Rosy_Cheeks',
            'sideburns': '30_Sideburns',
            'smiling': '31_Smiling',
            'straight_hair': '32_Straight_Hair',
            'wavy_hair': '33_Wavy_Hair',
            'wearing_earrings': '34_Wearing_Earrings',
            'wearing_hat': '35_Wearing_Hat',
            'wearing_lipstick': '36_Wearing_Lipstick',
            'wearing_necklace': '37_Wearing_Necklace',
            'wearing_necktie': '38_Wearing_Necktie',
            'young': '39_Young'
        }
        self.direction_idx = {k: v for k, v in zip(range(len(direction_map)), direction_map)}

        self.direction_dict = dict()
        self._direction_values = {k: 0.0 for k in direction_map}

        for k, v in direction_map.items():
            self.direction_dict[k] = self.boundaries[v].view(1, 1, -1).to(device)

        self.input_kwargs = {
            'noise': None,
            'randomize_noise': False,
            'input_is_style': True
        }

    def get_new_face(self, input_kwargs=None, vector=None, get_styles=False,
                     new_face=False, resolution=0, channel_ratio=0.):
        """Generates face.
        """
        if vector is None:
            vector = self.get_vector(n_styles=1)
        if input_kwargs is None and not new_face:
            input_kwargs = self.input_kwargs.copy()
        elif new_face:
            input_kwargs = {}
        input_kwargs['styles'] = vector
        input_kwargs['return_styles'] = get_styles
        with torch.no_grad():
            with self.lock:
                if resolution:
                    if resolution not in self.resolutions:
                        raise RuntimeError(
                            f'Inappropriate resolution: {resolution}, '
                            f'possible choices are {self.resolutions}'
                        )
                    LOG.info(f'Set resolution to {resolution}')
                    self.gen.target_res = resolution
                if channel_ratio is not None and channel_ratio > 0:
                    if channel_ratio not in self.channel_ratio:
                        raise RuntimeError(
                            f'Inappropriate channel ratio: {channel_ratio}, '
                            f'possible choices are {self.channel_ratio}'
                        )
                    LOG.info(f'Set channel ratio to {channel_ratio}')
                    dynamic_channel.set_uniform_channel_ratio(self.gen, channel_ratio)
                img, styles = self.gen(**input_kwargs)

                if resolution or (channel_ratio and channel_ratio > 0):
                    dynamic_channel.reset_generator(self.gen)

            img = img.cpu().numpy()[0]
            if get_styles:
                return to_img(img), styles
            return to_img(img)

    def cache_vector(self, vector):
        vector_id = str(uuid.uuid4())
        self.cache[vector_id] = CachedVector(vector, time.time())
        self._invalidate_cache()
        return vector_id

    def get_cached_vector(self, key):
        result = self.cache[key]
        self._invalidate_cache()
        return result.vector

    def _invalidate_cache(self):
        keys = list(self.cache.keys())
        now = time.time()
        for key in keys:
            if now - self.cache[key].time >= self.keep_cache_sec:
                LOG.info(f'[Cache] Expired vector ID={key}')
                del self.cache[key]

    def get_vector(self, n_styles=18):
        return torch.randn([1, n_styles, 512]).to(self.device)

    def get_styles(self):
        with torch.no_grad():
            rand = self.get_vector(1)
            styles = self.gen.get_style(rand)
        return styles

    def get_direction_values(self):
        return self._direction_values.copy()

    def encode_image_get_vector(self, img):
        for_encoder = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        for_encoder = ((for_encoder.astype(np.float32) - 127.5) / 127.5).transpose([2, 0, 1])
        with torch.no_grad():
            vector = self.encoder(torch.tensor(for_encoder).unsqueeze(0).to(self.device))
            img = self.get_new_face(self.input_kwargs, vector=vector)
        return img, vector

    def deform_vector(self, old_vector, direction_values, max_value=0.6):
        edited_code = old_vector.clone()
        for direction_name, value in direction_values.items():
            edited_code[:, :n_style_to_change] = (
                    edited_code[:, :n_style_to_change]
                    + value * self.direction_dict[direction_name] / 100 * max_value
            )

        return edited_code


def main():
    args = parse_args()
    generator_path = args.generator
    encoder_path = args.encoder
    boundary_path = args.boundary

    # dynamic_channel.set_uniform_channel_ratio(gen, 1)  # set channel
    # gen.target_res = 1024  # set resolution
    # out, _ = gen(...)  # generate image
    # dynamic_channel.reset_generator(gen)  # restore the generator
    face_gen = FaceGen(
        config_name='anycost-ffhq-config-f',
        gen_path=generator_path,
        enc_path=encoder_path,
        bound_path=boundary_path
    )
    direction_i = 0
    direction_step = 5.
    direction_values = face_gen.get_direction_values()
    direction_idx = {k: v for k, v in zip(range(len(direction_values)), direction_values)}
    vector = face_gen.get_vector()

    if args.image:
        img = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
        img, vector = face_gen.encode_image_get_vector(img)
    else:
        img = face_gen.get_new_face({}, vector=vector)
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
                direction_i = direction_i % len(direction_values)
            if key == ord('b'):
                # prev direction
                direction_i = direction_i + len(direction_values) - 1
                direction_i = direction_i % len(direction_values)
            if key == ord('['):
                direction_name = direction_idx[direction_i]
                direction_values[direction_name] -= direction_step
                new_vector = face_gen.deform_vector(vector, direction_values)

                img = face_gen.get_new_face(vector=new_vector)
            if key == ord(']'):
                direction_name = direction_idx[direction_i]
                direction_values[direction_name] += direction_step
                new_vector = face_gen.deform_vector(vector, direction_values)

                img = face_gen.get_new_face(vector=new_vector)
            if key == 32:
                direction_values = face_gen.get_direction_values()

                img, vector = face_gen.get_new_face(get_styles=True, new_face=True)
                img = face_gen.get_new_face(vector=vector)

            show(img, direction_idx, direction_i)


if __name__ == '__main__':
    main()

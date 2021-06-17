import logging

import cv2
from ml_serving.utils import helpers

import generate_face


LOG = logging.getLogger(__name__)
PARAMS = {
    'generator': None,
    'encoder': None,
    'boundary': None,
}


def init_hook(ctx, **params):
    PARAMS.update(params)

    face_gen = generate_face.FaceGen(
        config_name='anycost-ffhq-config-f',
        gen_path=PARAMS['generator'],
        enc_path=PARAMS['encoder'],
        bound_path=PARAMS['boundary']
    )
    return face_gen


def process(inputs, ctx, **kwargs):
    """Generate face

    Possible inputs:
    - parameters values (random face)
    - parameters + cached vector id (edit earlier generated face)
    - parameters + vector [18, 512] (face with given parameters for style vectors)
    - parameters + image (re-generated face with given parameters)

    Additional parameters:
    - resolution: 128, 256, 512 or 1024
    - channel_ratio: 0.25, 0.5, 0.75 or 1
    """
    face_gen: generate_face.FaceGen = ctx.global_ctx
    image = None
    vector = None
    is_video = False
    direction_values = face_gen.get_direction_values()

    if 'image' in inputs and inputs['image']:
        image, is_video = helpers.load_image(inputs, 'image', rgb=True)
    generate_image = helpers.boolean_string(
        helpers.get_param(inputs, 'generate_image', default=True)
    )
    vector_id = helpers.get_param(inputs, 'vector_id')
    method = helpers.get_param(inputs, 'method')
    resolution = helpers.get_param(inputs, 'resolution')
    channel_ratio = helpers.get_param(inputs, 'channel_ratio')

    if method == 'get_face_params':
        return {'face_params': list(face_gen.get_direction_values().keys())}

    if 'vector' in inputs and inputs['vector']:
        vector = inputs['vector']
    if vector_id:
        # Try to load vector
        vector = face_gen.get_cached_vector(vector_id)
        LOG.info(f'Using cached vector ID={vector_id}')

    # if image supplied, then generate a vector from it and re-generate image
    if image is not None:
        image, styles = face_gen.encode_image_get_vector(image)
        vector_id = face_gen.cache_vector(vector)

    for name in direction_values:
        if name in inputs:
            val = helpers.get_param(inputs, name, 0.0)
            direction_values[name] = val

    # If there is still no vector, then generate one to get random face
    new_face = vector is None
    gen_kwargs = {'resolution': resolution, 'channel_ratio': channel_ratio}

    result = {}
    if not new_face:
        styles = face_gen.deform_vector(vector, direction_values)
        if generate_image:
            img, styles = face_gen.get_new_face(
                vector=styles,
                get_styles=True,
                **gen_kwargs
            )
    else:
        # rand = face_gen.get_vector(1)
        styles = face_gen.get_styles()
        styles = face_gen.deform_vector(styles, direction_values)
        if generate_image:
            img = face_gen.get_new_face(vector=styles, **gen_kwargs)
        cache_vector_id = face_gen.cache_vector(styles)
        LOG.info(f'New cached vector ID={cache_vector_id}')
        result['vector_id'] = cache_vector_id

    if not is_video and generate_image:
        img = cv2.imencode('.jpg', img[:, :, ::-1])[1].tobytes()
        result['image'] = img

    if vector_id:
        result['vector_id'] = vector_id
    result['vector'] = styles.cpu().numpy().tolist()
    return result

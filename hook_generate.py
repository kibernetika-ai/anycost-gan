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
    - parameters dict (random face)
    - parameters + vector [18, 512] (face with given parameters for style vectors)
    - parameters + image (re-generated face with given parameters)
    """
    face_gen: generate_face.FaceGen = ctx.global_ctx
    image = None
    vector = None
    is_video = False
    direction_values = face_gen.get_direction_values()

    if 'image' in inputs and inputs['image']:
        image, is_video = helpers.load_image(inputs, 'image', rgb=True)
    if 'vector' in inputs and inputs['vector']:
        vector = inputs['vector']
    if 'vector_id' in inputs and inputs['vector_id']:
        # Try to load vector
        vector_id = helpers.get_param(inputs, 'vector_id')
        vector = face_gen.get_cached_vector(vector_id)
        LOG.info(f'Using cached vector ID={vector_id}')

    # if image supplied, then generate a vector from it and re-generate image
    if image is not None:
        image, vector = face_gen.encode_image_get_vector(image)

    for name in direction_values:
        if name in inputs:
            val = helpers.get_param(inputs, name, 0.0)
            direction_values[name] = val

    # If there is still no vector, then generate one to get random face
    new_face = vector is None

    if not new_face:
        vector = face_gen.deform_vector(vector, direction_values)

    img, styles = face_gen.get_new_face(
        vector=vector,
        get_styles=True,
        new_face=new_face
    )
    result = {}
    if new_face:
        styles = face_gen.deform_vector(styles, direction_values)
        img = face_gen.get_new_face(vector=styles)
        cache_vector_id = face_gen.cache_vector(styles)
        LOG.info(f'New cached vector ID={cache_vector_id}')
        result['vector_id'] = cache_vector_id

    if not is_video:
        img = cv2.imencode('.jpg', img[:, :, ::-1])[1].tobytes()

    result['image'] = img
    result['vector'] = styles.cpu().numpy().tolist()
    return result

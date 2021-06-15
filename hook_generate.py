import cv2
from ml_serving.utils import helpers

import generate_face


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

    if 'image' in inputs:
        image, is_video = helpers.load_image(inputs, 'image', rgb=True)
    if 'vector' in inputs:
        vector = inputs['vector']

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
    if new_face:
        new_styles = face_gen.deform_vector(styles, direction_values)
        img = face_gen.get_new_face(vector=new_styles)

    if not is_video:
        img = cv2.imencode('.jpg', img[:, :, ::-1])[1].tobytes()

    return {
        'image': img,
        'vector': styles.cpu().numpy().tolist(),
    }

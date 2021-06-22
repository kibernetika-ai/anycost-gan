import logging

import cv2
from ml_serving.utils import helpers
import numpy as np

import generate_face


LOG = logging.getLogger(__name__)
PARAMS = {
    'generator': None,
    'encoder': None,
    'boundary': None,
}


def init_hook(ctx, **params):
    PARAMS.update(params)

    head_pose = None
    if len(ctx.drivers) > 1:
        head_pose = ctx.drivers[1]
    face_gen = generate_face.FaceGen(
        config_name='anycost-ffhq-config-f',
        gen_path=PARAMS['generator'],
        enc_path=PARAMS['encoder'],
        bound_path=PARAMS['boundary'],
        head_pose=head_pose,
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
    face_driver = ctx.drivers[0]
    if len(ctx.drivers) > 1:
        head_pose_driver = ctx.drivers[1]
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
    face_detect = helpers.boolean_string(
        helpers.get_param(inputs, 'face_detect', default=True)
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
        # If not squared - need crop.
        if image.shape[0] != image.shape[1] and face_detect:
            LOG.info('Trigger face detection')
            # Crop image by face.
            boxes = get_boxes(face_driver, image[:, :, ::-1], threshold=.5)
            if len(boxes) != 1:
                raise RuntimeError('Image must contain exactly one face.')
            image = crop_by_box(image, boxes[0], margin=[0.25, 0.12, 0.31, 0.47])
        cv2.imwrite('ff.jpg', image[:, :, ::-1])
        _, vector = face_gen.encode_image_get_vector(image)
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


def get_boxes(face_driver, frame, threshold=0.5, offset=(0, 0)):
    input_name, input_shape = list(face_driver.inputs.items())[0]
    output_name = list(face_driver.outputs)[0]
    inference_frame = cv2.resize(frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = face_driver.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    xmin = boxes[:, 0] * frame.shape[1] + offset[0]
    xmax = boxes[:, 2] * frame.shape[1] + offset[0]
    ymin = boxes[:, 1] * frame.shape[0] + offset[1]
    ymax = boxes[:, 3] * frame.shape[0] + offset[1]
    xmin[xmin < 0] = 0
    xmax[xmax > frame.shape[1]] = frame.shape[1]
    ymin[ymin < 0] = 0
    ymax[ymax > frame.shape[0]] = frame.shape[0]

    boxes[:, 0] = xmin
    boxes[:, 2] = xmax
    boxes[:, 1] = ymin
    boxes[:, 3] = ymax
    return boxes


def crop_by_boxes(img, boxes):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0.):
    h = (box[3] - box[1])
    w = (box[2] - box[0])
    if isinstance(margin, list) and len(margin) == 4:
        margin_top, margin_bottom, margin_left, margin_right = margin
    else:
        margin_top = margin_bottom = margin_left = margin_right = margin
    ymin = int(max([box[1] - h * margin_top, 0]))
    ymax = int(min([box[3] + h * margin_bottom, img.shape[0]]))
    xmin = int(max([box[0] - w * margin_left, 0]))
    xmax = int(min([box[2] + w * margin_right, img.shape[1]]))
    return img[ymin:ymax, xmin:xmax]

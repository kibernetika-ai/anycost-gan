import cv2
import numpy as np

NORMALIZATION_STANDARD = "standard"
NORMALIZATION_FIXED = "fixed"
NORMALIZATION_PREWHITEN = "prewhiten"
NORMALIZATION_HOPENET = "hopenet"

DEFAULT_NORMALIZATION = NORMALIZATION_FIXED


def get_images(image, bounding_boxes, face_crop_size=160, face_crop_margin=32, normalization=None,
               face_crop_margin_coef: float = None):

    images = []

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        # img_size = np.asarray(image.shape)[0:2]
        if nrof_faces > 1:
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            cropped = crop_by_box(image, det, margin=face_crop_margin, margin_coef=face_crop_margin_coef)

            scaled = cv2.resize(cropped, (face_crop_size, face_crop_size), interpolation=cv2.INTER_AREA)
            if normalization == NORMALIZATION_PREWHITEN:
                images.append(prewhiten(scaled))
            elif normalization == NORMALIZATION_STANDARD:
                images.append(normalize(scaled))
            elif normalization == NORMALIZATION_FIXED:
                images.append(fixed_normalize(scaled))
            elif normalization == NORMALIZATION_HOPENET:
                images.append(hopenet(scaled))
            else:
                images.append(scaled)

    return images


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def crop_by_boxes(img, boxes, margin_coef=None):
    crops = []
    for box in boxes:
        cropped = crop_by_box(img, box, margin_coef=margin_coef)
        crops.append(cropped)
    return crops


def crop_by_box(img, box, margin=0, margin_coef=None):
    if margin_coef is not None:
        h = (box[3] - box[1])
        w = (box[2] - box[0])
        ymin = int(max([box[1] - h * margin_coef, 0]))
        ymax = int(min([box[3] + h * margin_coef, img.shape[0]]))
        xmin = int(max([box[0] - w * margin_coef, 0]))
        xmax = int(min([box[2] + w * margin_coef, img.shape[1]]))
    else:
        ymin = max([box[1] - margin, 0])
        ymax = min([box[3] + margin, img.shape[0]])
        xmin = max([box[0] - margin, 0])
        xmax = min([box[2] + margin, img.shape[1]])

    return img[ymin:ymax, xmin:xmax]


# images normalization

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def fixed_normalize(x):
    return (x - 127.5) / 128.0


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])


# from fastai.vision.data
# imagenet_stats
def hopenet(face):
    face = face.transpose([2, 0, 1])
    face = face / 255.0
    return (face - mean) / std

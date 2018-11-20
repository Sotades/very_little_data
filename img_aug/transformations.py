import random
import cv2

def random_null(images):
    for img in images:
        yield img


def random_stretch_x(images):
    for img in images:
        scale_factor = random.uniform(1, 3)
        height = img.shape[0] # keep original height
        width = int(img.shape[1] * scale_factor)
        dim = (width, height)
        transformed_image = cv2.resize(img, dim)
        yield transformed_image


def random_stretch_y(images):
    for img in images:
        yield img


def random_shear(images):
    for img in images:
        yield img


def random_rotate(images):
    for img in images:
        yield img


def random_scale(images):
    for img in images:
        yield img

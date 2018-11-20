import unittest
import cv2
import tubo
import numpy as np
from transformations import random_null
from transformations import random_stretch_x
from transformations import random_stretch_y


class TestDefectTransformations(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread('images/edge_defect1.jpg')
        # Make list of base defects
        self.images = []
        self.images.append(self.image)

    def test_no_transform(self):

        self.output = tubo.pipeline(
            self.images,
            random_null,
        )

        for output_image in self.output:
            np.testing.assert_array_equal(self.image, output_image)

    def test_random_stretch_x(self):
        self.output = tubo.pipeline(
            self.images,
            random_stretch_x,
        )

        input_image_width = self.image.shape[1]
        input_image_height = self.image.shape[0]
        for output_image in self.output:
            output_image_width = output_image.shape[1]
            output_image_height = output_image.shape[0]
            self.assertGreaterEqual(output_image_width, input_image_width)
            self.assertEqual(output_image_height, input_image_height)

    def test_random_stretch_y(self):
        self.output = tubo.pipeline(
            self.images,
            random_stretch_y,
        )

        input_image_width = self.image.shape[1]
        input_image_height = self.image.shape[0]
        for output_image in self.output:
            output_image_width = output_image.shape[1]
            output_image_height = output_image.shape[0]
            self.assertGreaterEqual(output_image_height, input_image_height)
            self.assertEqual(output_image_width, input_image_width)


if __name__ == '__main__':
    unittest.main()

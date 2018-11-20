import tubo
from transformations import random_stretch_x
from transformations import random_stretch_y
from transformations import random_shear
from transformations import random_scale
from transformations import random_rotate


class Defect:

    # Initializer / Instance Attributes
    def __init__(self, defect_image, pipeline):

        self.pipeline = pipeline            # Pipeline that distorts the defect
        self.defect_image = defect_image    # original image of defect

        self.x: int                         # x coordinate of top left corner of defect box
        self.y: int                         # y coordinate of top left corner of defect box
        self.h: int                         # height of surrounding defect box
        self.w: int                         # width of surrounding defect box


    def distort_defect(self):
        # apply the pipeline to the defect to distort it
        output = tubo.pipeline(
            self.defect_image,
            random_stretch_x,
            random_stretch_y,
            random_shear,
            random_scale,
            random_rotate,
        )


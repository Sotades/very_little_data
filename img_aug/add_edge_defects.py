import random
import image as im

# Specify how many 'bottles' will be used (it's actually the same image always, but clearly we only put a few
# augmented defects on each 'bottle' - bottle_count
bottle_count = 4

# Repeat bc times
for bottle_count in range(bottle_count):

#   Specify how many times to try to add defect (n) to each bottle
    add_defect_count = random.randint(1, 5)

#   Get coordinates of edges
    label_edges[] = im.

#   Repeat n times

#       Decide which edge(s) to apply distortion to

#       Generate defect; randomly read defect image, apply random pipeline transforms

#       Given size of defect and orientation, decide where to put it along the label edge

#       Overlay defect at location specified


#   Determine which box(es) contain defect

#   Label boxes as defective NOK

#   All boxes NOT labelled NOK label as OK

#   Write all NOK boxes as files in NOK folder format 'bc' + bc + 'box'x'_'y+'NOK'.jpg

#   Write all OK boxes as files in OK folder format 'bc' + bc + 'box'x'_'y+'OK'.jpg





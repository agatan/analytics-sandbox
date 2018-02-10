from typing import List

import colorsys
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(image: np.ndarray, boxes: np.ndarray, box_classes: List[int],
               class_names: List[str], scores: List[float] = None):
    """Draw bounding boxes on image.

    Args:
        image (np.ndarray): Array of shape (width, height, 3)
        boxes (np.ndarray): Array of shape(num_boxes, 4) containing box corners
                            as (y_min, x_min, y_max, x_max).
        box_classes (List[int]): list of indices to class_names.
        class_names (List[str]): list of string class names.
        scores (List[float]): list of scores for each box.
    """
    image = Image.fromarray(np.floor(image * 255 + 0.5).astype('uint8'))

    font = ImageFont.truetype(
        font='RictyDiminished-Regular.ttf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    colors = _get_colors_for_classes(len(class_names))

    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        if isinstance(scores, np.ndarray):
            score = scores[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)


def _get_colors_for_classes(num_classes):
    """Return list of random colors for number of classes given."""
    # Use previously generated colors if num_classes is the same.
    if (hasattr(_get_colors_for_classes, "colors") and
            len(_get_colors_for_classes.colors) == num_classes):
        return _get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    _get_colors_for_classes.colors = colors  # Save colors for future calls.
    return colors


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = np.asarray(Image.open('images/sample.png')) / 255.
    new_image = draw_boxes(image, [[10, 10, 50, 50]], [0], ['foo'])
    plt.imshow(new_image, interpolation='nearest')
    plt.show()

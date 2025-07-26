import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def parse_cvat(xml_path, output_mask_dir, image_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    os.makedirs(output_mask_dir, exist_ok=True)

    for image in root.findall(".//image"):
        img_name = image.attrib["name"]
        mask = np.zeros(image_shape, dtype=np.uint8)

        for poly in image.findall("polygon"):
            if poly.attrib.get("label") != "lane":
                continue
            points = np.array(
                [
                    [float(x), float(y)]
                    for x, y in [
                        point.split(",")
                        for point in poly.attrib["points"].split(";")
                    ]
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [points], 255)

        cv2.imwrite(
            os.path.join(
                output_mask_dir, os.path.splitext(img_name)[0] + ".png"
            ),
            mask,
        )

"""Parse Pascal VOC XML annotations (the format the Roboflow export uses)
into plain Python objects the rest of the pipeline works with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree


@dataclass
class BoxAnnotation:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class ImageAnnotation:
    filename: str
    width: int
    height: int
    boxes: list[BoxAnnotation]


def parse_voc_xml(xml_path: str | Path) -> ImageAnnotation:
    """Parse one Pascal VOC XML file into an ImageAnnotation."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        label = obj.findtext("name")
        bnd = obj.find("bndbox")
        boxes.append(
            BoxAnnotation(
                label=label,
                x1=int(round(float(bnd.findtext("xmin")))),
                y1=int(round(float(bnd.findtext("ymin")))),
                x2=int(round(float(bnd.findtext("xmax")))),
                y2=int(round(float(bnd.findtext("ymax")))),
            )
        )

    return ImageAnnotation(filename=filename, width=width, height=height, boxes=boxes)


def load_annotations(annotations_dir: str | Path) -> list[ImageAnnotation]:
    """Parse every .xml file in a directory."""
    annotations_dir = Path(annotations_dir)
    return [parse_voc_xml(p) for p in sorted(annotations_dir.glob("*.xml"))]

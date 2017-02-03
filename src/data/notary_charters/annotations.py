import xml.etree.ElementTree as ET

# Namespace for xml parsing
ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def _bounding_box(coords):
    """Finds bounding box for a set of coordinates"""
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def parse_annotations(files, keys=None):
    """Returns a generator over all annotations contained in the passed files

    Args:
    files: list of annotation xml files
    keys (optional): list of annotation keys to filter annotations by

    Generates: tuples of the form (image, bbox), where bbox is a tuple 
    of the form (left, upper, right, lower).
    """
    def strip_namespace(tag):
        return tag.split('}')[-1]

    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        page = root.find('ns:Page', ns)
        image = page.get('imageFilename')
        for child in page:
            if strip_namespace(child.tag) == 'ReadingOrder':
                continue

            if keys is not None and strip_namespace(child.tag) not in keys:
                continue

            coords = child.find('ns:Coords', ns).get('points').split(' ')
            coords = [(int(x), int(y)) 
                      for x, y in (c.split(',') for c in coords)]
            
            bbox = _bounding_box(coords)
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                print('Warning: invalid bounding box in {}'.format(file))
                continue
            yield image, bbox


def parse_labeled_annotations(file):
    """Returns a generator over all labeled annotations contained in the file

    Args:
    file: labeled annotations file

    Generates: tuples of the form (image, bbox, label), where bbox is a tuple 
    of the form (left, upper, right, lower).
    """
    with open(file, 'r') as f:
        for line in f:
            name, bbox, label = line.split(';')
            bbox = tuple((int(c) for c in bbox.split(' ')))
            yield name, bbox, int(label)


def write_labeled_annotations(file, labeled_annotations):
    """Writes a list of labeled annotations to a file, such that it can 
    be later recovered with parse_labeled_annotations

    Args:
    file: file to output
    labeled_annotations: list of tuples of the form (image, bbox, label), 
        where bbox is a tuple of the form (left, upper, right, lower).
    """
    with open(file, 'w') as f:
        for image, bbox, label in labeled_annotations:
            bbox = '{} {} {} {}'.format(bbox[0], bbox[1], bbox[2], bbox[3])
            f.write('{};{};{}\n'.format(image, bbox, label))

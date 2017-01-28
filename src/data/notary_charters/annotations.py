import xml.etree.ElementTree as ET

# Namespace for xml parsing
ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def _bounding_box(coords):
    """Finds bounding box for a set of coordinates"""
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return (min(xs), min(ys), max(xs), max(ys))

def parse(files, keys=None):
    """Returns a generator over all annotations contained in the passed files

    Args:
    files: list of annotation xml files
    keys (optional): list of annotation keys to filter annotations by

    Returns:
    Tuples of the form (image_name, (bounding_box))
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
            
            yield image, _bounding_box(coords)

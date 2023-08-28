
def xyxy_convert_xywh(size, box):
    # size: width, height
    # box: xmin, ymin, xmax,  ymax
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h  # center x, y;  w, h


def xywh_convert_xxyy(size, box):
    """
    xywh: list ->  center x, y;  w, h
    size: tuple -> (width, height)
    xxyy: top-left, bottom-right
    """
    xmin = size[0] * (float(box[0]) - float(box[2]) / 2)
    ymin = size[1] * (float(box[1]) - float(box[3]) / 2)
    xmax = xmin + size[0] * float(box[2])
    ymax = ymin + size[1] * float(box[3])

    return int(xmin), int(ymin), int(xmax), int(ymax)  # where xmin,ymin=top-left, xmax,ymax=bottom-right


def xxyy2xywh(size, box):
    """
    size: width, height
    box: [xmin, ymin, xmax, ymax], where xmin,ymin=top-left, xmax,ymax=bottom-right
    """
    newx = box[0] / size[0]
    newy = box[1] / size[1]
    neww = (box[2] - box[0]) / size[0]
    newh = (box[3] - box[1]) / size[1]

    return [newx, newy, neww, newh]  # min vertex coordinates: x, y, w, h


def xywh2xxyy(size, box):
    """
    size: width, height
    box: [x, y, w, h], where x,y=top-left

    xmin = float(x) * float(width)
    ymin = float(y) * float(height)
    xmax = xmin + float(w) * float(width)
    ymax = ymin + float(h) * float(height)
    """
    xmin = box[0] * box[2]
    ymin = box[1] * box[3]
    xmax = xmin + box[2] * size[0]
    ymax = ymin + box[3] * size[1]

    return [xmin, ymin, xmax, ymax]  # min vertex coordinates: x, y, w, h
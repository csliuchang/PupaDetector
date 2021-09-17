try:
    from .rbbox_geo import rbbox_iou_iof
except Exception as err:
    print(err)
    rbbox_iou_iof = None

__all__ = ['rbbox_iou_iof']

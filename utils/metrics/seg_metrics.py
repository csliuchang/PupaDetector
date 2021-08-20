import torch


class SegEval(object):
    def __init__(self, ignore_label=255):
        pass
        self.ignore_label = ignore_label

    def __call__(self, collections, n_classes):
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        for collection in collections:
            logits = collection['predicts']
            label = collection['gt_masks']
            probs = torch.softmax(logits, dim=0)
            preds = torch.argmax(probs, dim=0)
            keep = label != self.ignore_label
            hist += torch.bincount(
                label[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
            ).view(n_classes, n_classes).float()
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()

from models.base.backbone.stdc import STDCNet813
import torch
import time




if __name__ == "__main__":
    a = torch.randn(1, 3, 512, 512).cuda()
    backbone = STDCNet813(3, (512, 512)).cuda()
    start = time.time()
    out = backbone(a)
    end = time.time()-start
    print('each image use %5f seconds, and image size is 512' % end, )
    print([i.shape for i in out])

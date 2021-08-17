from models.base.backbone.pvt import pvt_v2_b0
import torch
import time




if __name__ == "__main__":
    a = torch.randn(1, 3, 512, 512).cuda()
    backbone = pvt_v2_b0().cuda()
    start = time.time()
    out = backbone(a)
    end = time.time()-start
    print('each image use %5f seconds, and image size is 512' % end, )
    print([i.shape for i in out])

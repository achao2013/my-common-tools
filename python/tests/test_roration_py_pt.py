from pytorch import  eulerAnglesToRotationMatrix_pt
from pythonic import  eulerAnglesToRotationMatrix_np
if __name__ == '__main__' :
    e0 = np.random.rand(3) * math.pi * 2 - math.pi
    e1 = np.random.rand(3) * math.pi * 2 - math.pi
    e=np.stack([e0,e1],axis=0)
    R_np0 = eulerAnglesToRotationMatrix_np(e0)
    R_np1 = eulerAnglesToRotationMatrix_np(e1)
    R_np = np.stack((R_np0,R_np1), axis=2)
    R = eulerAnglesToRotationMatrix_pt(torch.from_numpy(e))
    print(R)
    print(R_np)
    print(torch.allclose(torch.from_numpy(R_np), R))

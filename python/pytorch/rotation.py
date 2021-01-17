import pytorch

def eulerAnglesToRotationMatrix_batch(theta) :

    rot_sin = torch.sin(theta[:,0])
    rot_cos = torch.cos(theta[:,0])
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    R_x = torch.stack([
        torch.stack([ones, zeros, zeros]),
        torch.stack([zeros, rot_cos, -rot_sin]),
        torch.stack([zeros, rot_sin, rot_cos])
    ])

    rot_sin = torch.sin(theta[:,1])
    rot_cos = torch.cos(theta[:,1])
    R_y = torch.stack([
        torch.stack([rot_cos, zeros, rot_sin]),
        torch.stack([zeros, ones, zeros]),
        torch.stack([-rot_sin, zeros, rot_cos])
    ])

    rot_sin = torch.sin(theta[:,2])
    rot_cos = torch.cos(theta[:,2])
    R_z = torch.stack([
        torch.stack([rot_cos, -rot_sin, zeros]),
        torch.stack([rot_sin, rot_cos, zeros]),
        torch.stack([zeros, zeros, ones])
    ])
    #print(R_y.shape, R_x.shape)
    R=torch.einsum('ija,jka->ika', (R_y, R_x))
    R=torch.einsum('ija,jka->ika', (R_z, R))
    return R

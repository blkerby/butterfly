import torch

def simplex_projection(x):
    xv, xi = x.sort(descending=True, dim=1)
    cs = torch.cumsum(xv, dim=1)
    ts = (cs - 1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1)
    d = ts - xv
    d1 = torch.where(d > 0, torch.full_like(d, -float('inf')), d)
    js = torch.argmax(d1, dim=1)
    outs = torch.clamp_min(xv - ts[torch.arange(ts.shape[0]), js].view(-1, 1), 0.0)
    out = torch.empty_like(outs)
    out.scatter_(dim=1, index=xi, src=outs)
    return out

def l1_sphere_projection(x):
    sgn = torch.sign(x)
    return sgn * simplex_projection(torch.abs(x))

out = simplex_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
                                       [0.8, -0.2, 0.5, 0.3]]))
print(out)

out = l1_sphere_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
                                       [-0.8, -0.2, -0.5, 0.3]]))
print(out)
import torch

def simplex_projection(x):
    xv, xi = x.sort(descending=True, dim=1)
    cs = torch.cumsum(xv, dim=1)
    ts = (cs - 1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1)
    d = ts - xv
    d1 = torch.where(d > 0, torch.full_like(d, -float('inf')), d)
    js = torch.argmax(d1, dim=1)
    outs = torch.clamp_min(xv - ts[torch.arange(ts.shape[0]), js], 0.0)
    out = torch.empty_like(outs)
    out.scatter_(dim=1, index=xi, src=outs)
    return out

def l1_sphere_projection(x):
    sgn = torch.sign(x)
    return sgn * simplex_projection(torch.abs(x))

def simplex_tangent_projection(xz, g):
    gv, gi = g.sort(descending=True, dim=1)
    xs = torch.gather(xz, dim=1, index=gi)
    zero = torch.zeros_like(gv)
    gz = torch.where(xs, gv, zero)
    gnz = torch.where(xs, zero, gv)
    snz = torch.sum(gnz, dim=1)
    nz1 = torch.sum(~xz, dim=1)
    cs = torch.cumsum(gz, dim=1)
    c1 = torch.cumsum(xs, dim=1) + nz1.view(-1, 1)
    ts = (snz.view(-1, 1) + cs) / c1.type(g.dtype)
    d = ts - gv
    d1 = torch.where(d > 0, torch.full_like(d, -float('inf')), d)
    js = torch.argmax(d1, dim=1)
    gd = gv - ts[torch.arange(ts.shape[0]), js].view(-1, 1)
    outs = torch.where(xs, torch.clamp_min(gd, 0.0), gd)
    out = torch.empty_like(outs)
    out.scatter_(dim=1, index=gi, src=outs)
    return out

def l1_sphere_tangent_projection(x, g):
    xz = x == 0
    sgn = torch.where(xz, torch.sign(g), torch.sign(x))
    return simplex_tangent_projection(xz, g * sgn) * sgn

b = torch.randn([8, 16]) < 0.0
A = torch.randn([8, 16])

# out = simplex_tangent_projection(
#     torch.tensor([[False, True, False, False]]),
#     torch.tensor([[1.5, 0.7, 0.6, 0.2]]))
out = simplex_tangent_projection(b, A)
print(torch.sum(out, dim=1))
#
# out = simplex_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
#                                        [0.8, -0.2, 0.5, 0.3]]))
# print(out)
#
# out = l1_sphere_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
#                                        [-0.8, -0.2, -0.5, 0.3]]))
# print(out)
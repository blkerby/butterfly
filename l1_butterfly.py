import torch

def l1_rotation(x, y, c):
    r = torch.abs(x) + torch.abs(y)
    c0 = c % 1
    c1 = torch.where(y >= 0,
                     torch.where(x >= 0, torch.full_like(x, 0), torch.full_like(x, 1)),
                     torch.where(x >= 0, torch.full_like(x, 3), torch.full_like(x, 2)))
    x1 = torch.where(c1 == 0, x,
                     torch.where(c1 == 1, y,
                                 torch.where(c1 == 2, -x, -y)))
    y1 = torch.where(c1 == 0, y,
                     torch.where(c1 == 1, -x,
                                 torch.where(c1 == 2, -y, x)))
    x2 = x1 - c0 * r
    y2 = y1 + c0 * r
    y3 = torch.where(y2 >= r, 2 * r - y2, y2)
    c2 = torch.floor(c1 + c) % 4

    x_out = torch.where(c2 == 0, x2,
                        torch.where(c2 == 1, -y3,
                                    torch.where(c2 == 2, -x2, y3)))
    y_out = torch.where(c2 == 0, y3,
                        torch.where(c2 == 1, x2,
                                    torch.where(c2 == 2, -y3, -x2)))
    return x_out, y_out

# import matplotlib.pyplot as plt
# import time
#
# xs = []
# ys = []
#
# plt.ion()
# for i in range(100):
#     x = torch.tensor([3.0])
#     y = torch.tensor([-2.0])
#     x1, y1 = l1_rotation(x, y, 4 + i / 20)
#     xs.append(x1)
#     ys.append(y1)
#     xv = torch.cat(xs)
#     yv = torch.cat(ys)
#     plt.scatter(xv, yv)
#     plt.draw()
#     plt.pause(0.1)
#     plt.clf()


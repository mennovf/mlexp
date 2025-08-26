import torch

def unsqueeze_n(x, at, n):
    for _ in range(n):
        x = x.unsqueeze(at)
    return x

class InsidePolygons():
    def __init__(self, polies):
        self.fro = torch.cat(polies, dim=-2)
        self.to = torch.cat([poly.roll(-1, dims=0) for poly in polies], dim=-2)

        self.v1 = (self.to - self.fro).unsqueeze(0)
        self.fro = self.fro.unsqueeze(0)

    def inside(self, x):
        fro, to, v1 = self.fro, self.to, self.v1
        x = x.unsqueeze(-2)

        t1 = ( x - fro).select(-1, 1) / v1.select(-1, 1)
        t0 = (x - fro - v1*t1.unsqueeze(-1)).select(-1, 0)

        hits = (t0 > 0) * (t1 > 0) * (t1 <= 1)

        nhits = hits.sum(dim=-1, keepdim=True)
        inside = nhits.remainder(2) == 1
        y = torch.cat([inside, inside.logical_not()], dim=-1).to(torch.float32)

        return y

def inside_any(polies, x):
    return InsidePolygons(polies).inside(x)

def inside(poly, x):
    return inside_any(poly.unsqueeze(0), x)


triangle = [
    [-0.5, 0.5],
    [-0.5, -0.5],
    [0.5, 0]
]

M = [
  [ -0.900,   0.900],   # Left leg bottom-left
  [ -0.900,  -0.900],   # Left leg top-left
  [ -0.600,  -0.900],   # Left leg top-right
  [ -0.200,  -0.403],
  [  0.000,  -0.187],   # Center valley
  [  0.200,  -0.393],
  [  0.600,  -0.900],
  [  0.900,  -0.900],
  [  0.900,   0.900],   # Right leg bottom-right
  [  0.600,   0.900],
  [  0.600,  -0.400],
  [  0.300,  -0.100],
  [  0.000,   0.200],   # Center peak
  [ -0.300,  -0.100],
  [ -0.600,  -0.400],
  [ -0.600,  0.900],
]

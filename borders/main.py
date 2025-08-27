import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import multinomial
import math


class Layer(nn.Module):
    def __init__(self, fro, to) -> None:
        super().__init__()

        self.parts = nn.Sequential(*[
            nn.Linear(fro, to),
            nn.ReLU(),
        ])

    def forward(self, x):
        return self.parts(x)


class Model(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super().__init__()
        self.layers = nn.Sequential(*[
            Layer(2, n_hidden),
            *[Layer(n_hidden, n_hidden) for _ in range(n_layers)],
            nn.Linear(n_hidden, 2)
        ])

    def forward(self, x, y = None):
        # x.shape() == (B, 2)

        # logits.shape() == (B, 2)
        logits = self.layers(x)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits, y)
            #print("crossing gvies", logits, y, loss)

        return logits, loss
        



import matplotlib.pyplot as plt
import show

def make_viewer(H: int, W: int, title="PyTorch 2D tensor viewer"):
    plt.ion()
    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    ax.set_axis_off()
    im = ax.imshow(np.zeros((H, W, 3), np.uint8), interpolation="nearest", origin="upper")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(ax.bbox)  # for blitting
    quit_flag = {"q": False}
    def on_key(e):
        if e.key in ("q", "escape"):
            quit_flag["q"] = True
    fig.canvas.mpl_connect("key_press_event", on_key)
    return fig, ax, im, bg, quit_flag

def blit_update(fig, ax, im, bg, img: np.ndarray):
    im.set_data(img)
    fig.canvas.restore_region(bg)
    ax.draw_artist(im)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()
    plt.pause(0.001)  # let GUI breathe

def circle(x):
    # x.shape() == (B, 2)
    
    d2 = x.square().sum(dim=-1)
    # d2.shape() == (B, 1)
    
    inside = d2 <= 1
    outside = d2 > 1
    
    y = torch.cat([
        inside.unsqueeze(-1),
        outside.unsqueeze(-1)
    ], dim=-1).to(torch.float32)
    
    return y
    

if __name__ == "__main__":
    import time, os, sys
    device = 'cuda'
    H, W = 320, 480
    B = 4096
    fig, ax, im, bg, quit_flag = make_viewer(H, W)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("medium")
    
    model = Model(128, 4)
    model.to(device)

    import polygons
    polies = polygons.smiley()
    polies = [tensor.detach().to(device) for tensor in polies]
    ps = polygons.InsidePolygons(polies)
    ground_truth = lambda x: ps.inside(x)

    # Display once first
    rr, cc = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1*W/H, 1*W/H, W, device=device), indexing="ij")
    windowx = torch.stack([cc, rr], dim=-1).detach().to(device)
    
    if os.environ.get("SHOW", False):
        ref = ground_truth(windowx)
        img = show.tensor_to_rgb_numpy(ref, eps=1e-2)
        blit_update(fig, ax, im, bg, img)
        time.sleep(5)
    
    iteration = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    while plt.fignum_exists(fig.number) and not quit_flag["q"]:
        start = time.time()
        optimizer.zero_grad()
        #x = (torch.rand(B, 2, device=device) - 0.5) * torch.tensor([2, 0.5], device=device)
        x = torch.rand(B, 2, device=device)*2 - 1
        y = ground_truth(x)
        
        pred, loss = model(x, y)
        loss.backward()
        optimizer.step()
        end = time.time()
        
        print(f"Iteration={iteration}, dt={(end - start) * 1000:.3f}ms, loss={loss:.4}")

        # Validation
        if iteration % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits, _ = model(windowx)
                probs = F.softmax(logits, dim=-1)
                img = show.tensor_to_rgb_numpy(probs, eps=1e-2)
                blit_update(fig, ax, im, bg, img)
                model.train()
        iteration += 1;


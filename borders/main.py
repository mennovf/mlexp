import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import multinomial



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

def tensor_to_rgb_numpy(t: torch.Tensor, eps: float = 1e-3) -> np.ndarray:
    """
    Map a 2-channel tensor to RGB:
      red   if p0 > p1 + eps
      green if p1 > p0 + eps
      blue  otherwise (≈ equal)
    Supports (H,W,2) or (2,H,W). Returns uint8 (H,W,3).
    """
    if t.ndim != 3:
        raise ValueError(f"expected 3D tensor, got {t.shape}")
    if t.shape[-1] == 2:
        hw2 = t
    elif t.shape[0] == 2:
        hw2 = t.permute(1, 2, 0)
    else:
        raise ValueError(f"last or first dim must be 2, got {t.shape}")

    hw2 = hw2.detach().to(torch.float32).cpu()
    p0, p1 = hw2[..., 0], hw2[..., 1]
    red   = p0 > (p1 + eps)
    green = p1 > (p0 + eps)
    blue  = ~(red | green)

    H, W = p0.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[red.numpy()]   = (255, 0, 0)
    img[green.numpy()] = (0, 255, 0)
    img[blue.numpy()]  = (0, 0, 255)
    return img

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
    import time
    device = 'cuda'
    H, W = 320, 480
    B = 1024
    fig, ax, im, bg, quit_flag = make_viewer(H, W)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    model = Model(2*100, 1)
    model.to(device)

    import polygons
    ptensor =  [torch.tensor(polygons.triangle) + torch.tensor([0.75, 0]), torch.tensor(polygons.triangle) + torch.tensor([-0.7, 0])]
    ptensor = [tensor.to(device) for tensor in ptensor]
    ground_truth = lambda x: polygons.inside_any(ptensor, x)

    # Display once first
    rr, cc = torch.meshgrid(torch.linspace(-2, 2, H, device=device), torch.linspace(-2*W/H, 2*W/H, W, device=device), indexing='ij')
    x = torch.stack([cc, rr], dim=-1).detach()
    ref = ground_truth(x)
    img = tensor_to_rgb_numpy(ref, eps=1e-2)
    blit_update(fig, ax, im, bg, img)
    time.sleep(5)
    
    iteration = 0
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    
    while plt.fignum_exists(fig.number) and not quit_flag["q"]:
        optimizer.zero_grad()
        x = torch.rand(B, 2, device=device) * 2 - 1
        y = ground_truth(x)
        
        pred, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        print(f"Iteration={iteration}, loss={loss:.4}")

        # Validation
        if iteration % 10 == 0:
            model.eval()
            rr, cc = torch.meshgrid(torch.linspace(-2, 2, H, device=device), torch.linspace(-2*W/H, 2*W/H, W, device=device), indexing="ij")
            x = torch.stack([cc, rr], dim=-1)
            logits, _ = model(x)
            probs = F.softmax(logits, dim=-1)
            img = tensor_to_rgb_numpy(probs, eps=1e-2)
            blit_update(fig, ax, im, bg, img)
            model.train()
        iteration += 1;


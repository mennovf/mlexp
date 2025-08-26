import numpy as np
import torch

# --- Palettes (sRGB 0..255) ---
_PALETTES = {
    "okabe-ito": {  # colorblind-friendly, high taste-to-effort ratio
                  "red":  (213,  94,   0),
                  "grn":  (  0, 158, 115),
                  "eq":   ( 86, 180, 233),  # sky blue
                  },
    "classic": {
        "red":  (255,   0,   0),
        "grn":  (  0, 255,   0),
        "eq":   (  0,   0, 255),
    },
    "pastel": {
        "red":  (244, 154, 151),
        "grn":  (152, 230, 179),
        "eq":   (173, 216, 230),
    },
}

def tensor_to_rgb_numpy(
    t: torch.Tensor,
    eps: float = 1e-3,
    *,
    sigma: float = 0.08,        # softness for red/green decision (≈ anti-alias)
    eq_sigma: float = 0.08,     # softness for the "equal" band around |p0-p1|<=eps
    palette: str = "okabe-ito", # "okabe-ito" | "classic" | "pastel"
    linear_mix: bool = True     # mix in linear-light for nicer gradients
) -> np.ndarray:
    """
    Smoothly map a 2-channel tensor to RGB with anti-aliased class edges.
      - Red when p0>>p1, Green when p1>>p0, Sky-blue near equality (|p0-p1|≈0).
    Supports (H,W,2) or (2,H,W). Returns uint8 (H,W,3).

    Notes:
      * 'sigma' & 'eq_sigma' are in the same units as p (assumes probabilities by default).
      * Set sigma=eq_sigma=0 for hard, old-school thresholds.
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
    d = p0 - p1                      # signed margin (+ => class 0 / red)
    a = torch.abs(d)

    # --- Smooth weights (anti-aliased classification) ---
    if sigma > 0:
        w_red_base = torch.sigmoid(d / sigma)      # favors red as d↑
    else:
        w_red_base = (d > 0).to(torch.float32)

    # distance from the equality band (|d|<=eps) with smooth falloff
    if eq_sigma > 0:
        delta = torch.clamp(a - eps, min=0.0)
        w_far = 1.0 - torch.exp(-0.5 * (delta / eq_sigma) ** 2)  # 0 near equal, →1 far
    else:
        w_far = (a > eps).to(torch.float32)

    w_eq  = 1.0 - w_far                    # "blue" weight near equality
    w_red = w_far * w_red_base
    w_grn = w_far * (1.0 - w_red_base)

    # normalize (numerical safety)
    S = (w_eq + w_red + w_grn).clamp_min(1e-12)
    w_eq, w_red, w_grn = w_eq/S, w_red/S, w_grn/S

    if palette not in _PALETTES:
        raise ValueError(f"unknown palette '{palette}'")

    pr = torch.tensor(_PALETTES[palette]["red"], dtype=torch.float32) / 255.0
    pg = torch.tensor(_PALETTES[palette]["grn"], dtype=torch.float32) / 255.0
    pb = torch.tensor(_PALETTES[palette]["eq"],  dtype=torch.float32) / 255.0

    # --- Mix in linear-light for smoother gradients (optional but nicer) ---
    def srgb_to_linear(c):
        return torch.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055) ** 2.4)
    def linear_to_srgb(c):
        return torch.where(c <= 0.0031308, 12.92*c, 1.055 * torch.clamp(c, 0, 1) ** (1/2.4) - 0.055)

    if linear_mix:
        pr_lin, pg_lin, pb_lin = srgb_to_linear(pr), srgb_to_linear(pg), srgb_to_linear(pb)
        mix = (w_red[..., None] * pr_lin +
               w_grn[..., None] * pg_lin +
               w_eq[...,  None] * pb_lin)
        rgb = linear_to_srgb(mix)
    else:
        mix = (w_red[..., None] * pr +
               w_grn[..., None] * pg +
               w_eq[...,  None] * pb)
        rgb = torch.clamp(mix, 0, 1)

    img = (rgb * 255.0 + 0.5).to(torch.uint8).numpy()
    return img

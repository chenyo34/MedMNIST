import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



def get_module_by_name(model: nn.Module, module_path: str) -> nn.Module:
    """
    Resolve a nested module by dotted path.

    Examples:
        'vgg16.features.28'
        'layer4.1.conv2'
        'features.28'
    """
    module: nn.Module = model
    for part in module_path.split('.'):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module



def replace_relu_inplace(module: nn.Module) -> None:
    """Recursively replace all in-place ReLU with non-inplace ReLU."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_inplace(child)


class GradCAM:
    """
    Minimal Grad-CAM implementation for CNN backbones.

    Args:
        model: PyTorch model.
        target_layer: Either a module object or a dotted module path string,
            e.g. 'vgg16.features.28' or 'layer4.1.conv2'.
    """

    def __init__(self, model: nn.Module, target_layer):
        self.model = model
        self.target_layer = (
            get_module_by_name(model, target_layer)
            if isinstance(target_layer, str)
            else target_layer
        )

        self.activations = None
        self.gradients = None
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradients)

    def _save_gradients(self, grad):
        self.gradients = grad

    def __call__(self, input_tensor: torch.Tensor, class_idx=None, retain_graph: bool = False):
        return self.forward(input_tensor, class_idx, retain_graph)

    def forward(self, input_tensor: torch.Tensor, class_idx=None, retain_graph: bool = False):
        """
        Args:
            input_tensor: Shape [1, C, H, W]
            class_idx: Optional target class index. If None, uses predicted class.
            retain_graph: Whether to retain the graph during backward.

        Returns:
            saliency_map: [1, 1, H, W]
            logits: [1, num_classes]
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"Expected input_tensor shape [1, C, H, W], got {tuple(input_tensor.shape)}")

        self.gradients = None
        self.activations = None

        _, _, h, w = input_tensor.size()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)

        if self.gradients is None:
            raise ValueError(
                "Gradients are None. Try input_tensor.requires_grad_(True), recreate the GradCAM object, "
                "and ensure the target layer participates in the computation graph."
            )
        if self.activations is None:
            raise ValueError("Activations are None. Forward hook did not capture the target layer output.")

        gradients = self.gradients
        activations = self.activations

        b, k, _, _ = gradients.shape
        alpha = gradients.view(b, k, -1).mean(dim=2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(dim=1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(
            saliency_map,
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        )

        saliency_min = saliency_map.min()
        saliency_max = saliency_map.max()
        saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency_map.detach(), logits.detach()

    def remove_hooks(self):
        self.forward_handle.remove()



def denormalize_img(img_tensor: torch.Tensor, mean, std) -> torch.Tensor:
    """
    Reverse channel normalization.

    Args:
        img_tensor: [1, 3, H, W] or [3, H, W]
        mean, std: length-3 sequences
    Returns:
        Tensor in [0, 1]
    """
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    mean_t = torch.tensor(mean, device=img_tensor.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=img_tensor.device).view(1, 3, 1, 1)

    img = img_tensor * std_t + mean_t
    return torch.clamp(img, 0, 1)



def show_gradcam_result(
    image_tensor: torch.Tensor,
    mask: torch.Tensor,
    mean,
    std,
    label_map=None,
    true_label=None,
    pred_label=None,
    alpha: float = 0.4,
):
    """
    Show original image, heatmap, and overlay side by side.

    Args:
        image_tensor: [1,3,H,W] or [3,H,W], normalized input image.
        mask: [1,1,H,W] or [H,W], Grad-CAM mask in [0,1].
        mean, std: normalization stats used for the model input.
        label_map: Optional dict mapping class ids to names. Supports int or str keys.
    """
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)

    image = denormalize_img(image_tensor, mean, std).squeeze(0).detach().cpu()
    image = image.permute(1, 2, 0).numpy()

    if mask.ndim == 4:
        heatmap = mask.squeeze().detach().cpu().numpy()
    else:
        heatmap = mask.detach().cpu().numpy()

    def _label_name(label):
        if label_map is None or label is None:
            return ""
        return label_map.get(label, label_map.get(str(label), "Unknown"))

    true_name = _label_name(true_label)
    pred_name = _label_name(pred_label)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    title = "Original"
    if true_label is not None:
        title += f"\nTrue: {true_label} ({true_name})"
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    title = 'Overlay'
    if pred_label is not None:
        title += f"\nPred: {pred_label} ({pred_name})"
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_gradcam_grid(
    image_tensors: list,
    heatmap_tensors: list,
    titles: list,
    mean,
    std,
    ncols: int = 3,
    overlay_alpha: float = 0.45,
    figscale: float = 3.6,
):
    """
    Plot a grid of Grad-CAM overlays (original + jet heatmap).

    Args:
        image_tensors: list of [1,3,H,W] or [3,H,W], normalized like model input.
        heatmap_tensors: list of [1,1,H,W] or [H,W], values in [0,1].
        titles: subplot titles.
        mean, std: ImageNet-style normalization (length 3).
        ncols: columns in the grid.
    """
    n = len(image_tensors)
    if n == 0:
        return
    nrows = (n + ncols - 1) // ncols
    fig_w = figscale * ncols
    fig_h = figscale * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        img_t = image_tensors[i]
        if img_t.ndim == 3:
            img_t = img_t.unsqueeze(0)
        image = denormalize_img(img_t, mean, std).squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        hm = heatmap_tensors[i]
        if hm.ndim == 4:
            hm = hm.squeeze()
        heatmap = hm.detach().cpu().numpy()

        ax.imshow(image)
        ax.imshow(heatmap, cmap="jet", alpha=overlay_alpha)
        ax.set_title(titles[i], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


__all__ = [
    'GradCAM',
    'get_module_by_name',
    'replace_relu_inplace',
    'denormalize_img',
    'show_gradcam_result',
    'show_gradcam_grid',
]

import torch
import torch.nn.functional as F
from torch import nn

def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None

class RMSNorm(nn.Module):
    """
    RMS  Normalization

    Args:
        dim (int): The dimension of the input.
        eps (float): The epsilon value.

    Attributes:
        scale (float): The scale value.
        gamma (nn.Parameter): The gamma parameter.

    Example:
        >>> module = RMSNorm(768)
        >>> x = torch.randn(2, 197, 768)
        >>> y = module(x)
        >>> y.shape
        torch.Size([2, 197, 768])

    """

    def __init__(self, dim, groups=1):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(groups, dim, 1))

    def forward(self, x):
        """Forward method implementation."""
        normed = F.normalize(x, dim=-2)
        return normed * self.scale * self.gamma

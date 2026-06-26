import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_base_grid(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Generates a normalized 2D grid in the range [-1, 1] for F.grid_sample.
    Returns shape: (1, h, w, 2)
    """
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    
    # Stack to create (h, w, 2) and add batch dimension
    base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return base_grid

def total_variation_loss(displacement: torch.Tensor) -> torch.Tensor:
    """
    Computes the Total Variation (TV) loss for the displacement field
    to enforce smoothness and prevent the grid from folding.
    displacement shape: (1, 2, grid_h, grid_w)
    """
    # Differences along the x-axis
    dx = displacement[..., :, 1:] - displacement[..., :, :-1]
    # Differences along the y-axis
    dy = displacement[..., 1:, :] - displacement[..., :-1, :]
    
    # Sum of absolute differences
    tv_loss = torch.sum(torch.abs(dx)) + torch.sum(torch.abs(dy))
    return tv_loss

def warp_image(image: torch.Tensor, base_grid: torch.Tensor, coarse_displacement: torch.Tensor) -> torch.Tensor:
    """
    Upsamples the coarse grid displacements, applies them to the base grid,
    and warps the image using bilinear interpolation.
    """
    _, _, h, w = image.shape
    
    # Upsample the coarse displacements to the full image resolution
    # Interpolation expects (B, C, H, W)
    upsampled_disp = F.interpolate(coarse_displacement, size=(h, w), mode='bilinear', align_corners=True)
    
    # Permute upsampled displacements to match grid_sample format: (B, H, W, 2)
    upsampled_disp = upsampled_disp.permute(0, 2, 3, 1)
    
    # Add displacements to the base coordinates
    warped_grid = base_grid + upsampled_disp
    
    # Warp the moving image
    warped_image = F.grid_sample(image, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return warped_image

def refine_homography_ffd(
    img_ref: torch.Tensor, 
    img_mov: torch.Tensor, 
    grid_size: tuple = (16, 16), 
    lr: float = 0.01, 
    lambda_tv: float = 0.5, 
    iterations: int = 200,
    verbose: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimizes a Free-Form Deformation grid to align img_mov to img_ref.
    Assumes images are equalized/normalized grayscale tensors of shape (1, 1, H, W).
    """
    device = img_ref.device
    _, _, h, w = img_ref.shape
    
    # 1. Initialize the base grid
    base_grid = generate_base_grid(h, w, device)
    
    # 2. Initialize the coarse displacement parameters (zero displacement initially)
    # Shape: (1, 2, grid_h, grid_w) -> 2 channels for dx and dy
    coarse_displacement = torch.zeros((1, 2, grid_size[0], grid_size[1]), device=device, requires_grad=True)
    
    # 3. Setup the Optimizer
    optimizer = torch.optim.Adam([coarse_displacement], lr=lr)
    
    mse_criterion = nn.MSELoss()
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Warp the moving image
        warped_img = warp_image(img_mov, base_grid, coarse_displacement)
        
        # Calculate Photometric Loss (MSE)
        mse_loss = mse_criterion(warped_img, img_ref)
        
        # Calculate Regularization Loss (TV)
        tv_loss = total_variation_loss(coarse_displacement)
        
        # Total Objective
        loss = mse_loss + lambda_tv * tv_loss
        
        # Backprop and step
        loss.backward()
        optimizer.step()
        
        if verbose and (i % 50 == 0 or i == iterations - 1):
            print(f"Iter {i:03d} | Total Loss: {loss.item():.6f} | MSE: {mse_loss.item():.6f} | TV: {tv_loss.item():.6f}")
            
    # Final forward pass to get the fully optimized image
    with torch.no_grad():
        final_warped_img = warp_image(img_mov, base_grid, coarse_displacement)
        
    return final_warped_img, coarse_displacement.detach()
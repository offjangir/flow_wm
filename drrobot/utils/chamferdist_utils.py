import torch

"""
Inefficient, but non pytorch3d implementation of Chamfer distance
"""

def chamfer_distance(x, y):
    """
    Compute the Chamfer distance between two point clouds.
    
    Args:
    x: (B, N, D) torch.Tensor, a batch of N points with D dimensions
    y: (B, M, D) torch.Tensor, a batch of M points with D dimensions
    
    Returns:
    chamfer_dist: (B,) torch.Tensor, the Chamfer distance for each batch
    """
    x = x.unsqueeze(2)  # (B, N, 1, D)
    y = y.unsqueeze(1)  # (B, 1, M, D)

    dist = torch.sum((x - y) ** 2, dim=-1)  # (B, N, M)

    min_dist_xy = torch.min(dist, dim=2)[0]  # (B, N)
    min_dist_yx = torch.min(dist, dim=1)[0]  # (B, M)

    chamfer_dist = torch.mean(min_dist_xy, dim=1) + torch.mean(min_dist_yx, dim=1)

    return chamfer_dist

def mean_chamfer_distance(x, y):
    """
    Compute the Chamfer distance between two point clouds.
    
    Args:
    x: (B, N, D) torch.Tensor, a batch of N points with D dimensions
    y: (B, M, D) torch.Tensor, a batch of M points with D dimensions
    
    Returns:
    chamfer_dist: (B,) torch.Tensor, the Chamfer distance for each batch
    """
    x = x.unsqueeze(2)  # (B, N, 1, D)
    y = y.unsqueeze(1)  # (B, 1, M, D)

    dist = torch.sum((x - y) ** 2, dim=-1)  # (B, N, M)

    min_dist_xy = torch.min(dist, dim=2)[0]  # (B, N)
    min_dist_yx = torch.min(dist, dim=1)[0]  # (B, M)

    chamfer_dist = torch.mean(min_dist_xy, dim=1) + torch.mean(min_dist_yx, dim=1)

    return chamfer_dist.mean()

# Example usage:
if __name__ == "__main__":
    import time

    # Create random point clouds
    batch_size, num_points_x, num_points_y, dim = 2, 1000, 800, 3
    x = torch.rand(batch_size, num_points_x, dim)
    y = torch.rand(batch_size, num_points_y, dim)

    # Number of iterations for benchmarking
    num_iterations = 10000

    # Benchmark non-compiled Chamfer distance
    start_time = time.time()  # Start timing before the loop
    for _ in range(num_iterations):
        distance_non_compiled = chamfer_distance(x, y)
    avg_non_compiled_time = (time.time() - start_time) / num_iterations  # Calculate average time

    print(f"Average non-compiled Chamfer distance time: {avg_non_compiled_time:.6f} seconds")

    # Benchmark compiled Chamfer distance
    compiled_chamfer_distance = torch.compile(chamfer_distance, fullgraph=True)
    start_time = time.time()  # Start timing before the loop
    for _ in range(num_iterations):
        distance_compiled = compiled_chamfer_distance(x, y)
    avg_compiled_time = (time.time() - start_time) / num_iterations  # Calculate average time

    print(f"Average compiled Chamfer distance time: {avg_compiled_time:.6f} seconds")

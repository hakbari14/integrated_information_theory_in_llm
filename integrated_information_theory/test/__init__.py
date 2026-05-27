import torch

def print_tensor_shape(tensor, name="Tensor", show_details=False):
    """
    User-friendly tensor shape printing
    """
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Dimensions: {tensor.dim()}")
    print(f"Total elements: {tensor.numel():,}")
    
    if show_details:
        print(f"Data type: {tensor.dtype}")
        if tensor.is_cuda:
            print(f"Device: GPU")
        elif tensor.device.type == 'cpu':
            print(f"Device: CPU")
        
        # Memory usage (approximate)
        memory_bytes = tensor.element_size() * tensor.numel()
        if memory_bytes < 1024:
            print(f"Memory: {memory_bytes} bytes")
        elif memory_bytes < 1024**2:
            print(f"Memory: {memory_bytes/1024:.2f} KB")
        elif memory_bytes < 1024**3:
            print(f"Memory: {memory_bytes/(1024**2):.2f} MB")
        else:
            print(f"Memory: {memory_bytes/(1024**3):.2f} GB")

# Example usage
x = torch.randn(2, 3, 224, 224)  # Batch of images
print_tensor_shape(x, "Input Images", show_details=True)
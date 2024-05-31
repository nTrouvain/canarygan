import torch
import socket


if __name__ == "__main__":

    print(f"\nPyTorch {torch.__version__}")

    cuda = torch.cuda.is_available()
    host = socket.gethostname()

    if not cuda:
        print(f"CUDA device not found on {host}.")
    else:
        print(f"CUDA {torch.version.cuda}")
        if not hasattr(torch._C, "_nccl_all_reduce"):
            print("PyTorch is not compiled with NCCL support.")
        else:
            print(f"NCCL {'.'.join([str(v) for v in torch.cuda.nccl.version()])}")
        if torch.backends.cudnn.is_available():
            print(f"CuDNN {torch.backends.cudnn.version()}")
        else:
            print("CuDNN not available in PyTorch.")

        n_device = torch.cuda.device_count()
        print(f"Found {n_device} devices on {socket.gethostname()}.")

        print("=" * 20)
        for i in range(n_device):
            print(f"Device {i}")
            print(torch.cuda.get_device_name(i))
            print("-" * 20)

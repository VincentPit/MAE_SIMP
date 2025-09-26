import torch
import subprocess


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("CUDA is not available. Using CPU.")
        try:
            subprocess.run(["nvidia-smi"], check=True)
        except subprocess.CalledProcessError:
            print(
                "nvidia-smi command failed. Make sure NVIDIA drivers are installed correctly."
            )
        except FileNotFoundError:
            print(
                "nvidia-smi command not found. Ensure the NVIDIA drivers and CUDA toolkit are installed."
            )


if __name__ == "__main__":
    check_cuda()

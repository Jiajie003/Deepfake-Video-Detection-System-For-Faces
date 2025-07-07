import torch

print("✅ PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("🎉 CUDA is available!")
    print("🖥️  GPU Name:", torch.cuda.get_device_name(0))
    print("💡 Total GPUs:", torch.cuda.device_count())
    print("🔥 CUDA Version:", torch.version.cuda)
else:
    print("❌ CUDA is NOT available.")
    print("Please make sure you are using a 64-bit Python environment and have installed the correct PyTorch GPU version.")

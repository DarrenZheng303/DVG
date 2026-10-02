# TAEHV checkpoints
Download the required files from the [upstream TAEHV repository](https://github.com/madebyollin/taehv) into `dvg/taehv/weights/`, keeping the original filenames:

```bash
mkdir -p dvg/taehv/weights
base_url="https://raw.githubusercontent.com/madebyollin/taehv/main"
curl -L "$base_url/taehv.pth" -o dvg/taehv/weights/taehv.pth
curl -L "$base_url/taehv1_5.pth" -o dvg/taehv/weights/taehv1_5.pth
curl -L "$base_url/taew2_1.pth" -o dvg/taehv/weights/taew2_1.pth
```

The files correspond to HunyuanVideo, HunyuanVideo-1.5, and Wan models using the Wan2.1 VAE (Wan2.1 1.3B and Wan2.2 A14B), respectively. Download only the checkpoint(s) needed by your backbone.

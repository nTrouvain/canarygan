import argparse
from pathlib import Path

from pmseq.viz.generation_umap import plot_generation_umap


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--gen_dir", type=str)
parser.add_argument("--reducer_ckpt", type=str)
parser.add_argument("--epoch", type=str)
parser.add_argument("--version", type=str)
parser.add_argument("--save_dir", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_file = (
        Path(args.save_dir) / f"umap_plot-version_{args.version}-epoch_{args.epoch}.pdf"
    )
    plot_generation_umap(
        data_dir=args.data_dir,
        gen_dir=args.gen_dir,
        epoch=args.epoch,
        version=args.version,
        save_file=save_file,
        reducer_ckpt=args.reducer_ckpt,
    )

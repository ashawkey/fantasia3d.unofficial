# Fantasia3D.unofficial

This is a quickly produced **unofficial** reproduction of [Fantasia3D](https://fantasia3d.github.io/) for text-to-3D generation, featuring the ability to generate high-quality meshes with PBR textures directly.

Thanks for the author's kind help in elaborating the details! 
This repo is also heavily based on [Nvdiffrec](https://github.com/NVlabs/nvdiffrec) (with extremely untidy modifications...).

**Important Notice: This repo is like a preview of this impressive method, and may not be actively maintained since the authors will release the [official implementation](https://github.com/Gorilla-Lab-SCUT/Fantasia3D). The performance seems not as good as the original paper, and there are still some details may be different, but it's still a start point if you are interested.**


https://user-images.githubusercontent.com/25863658/228754226-8f14f1e4-fc46-43ef-b815-497d6d79ca44.mp4


### Install
```bash
pip install -r requirements.txt
```

### Usage
First you need to create a config file under `configs`, you can copy and edit one of the provided examples like this:
```json
{
    "text": "a ripe strawberry",
    "random_textures": true,
    "iter": 5000,
    "save_interval": 10,
    "texture_res": [ 1024, 1024 ],
    "train_res": [512, 512],
    "batch": 8,
    "learning_rate": 0.001,
    "ks_min" : [0, 0.25, 0],
    "out_dir": "strawberry",
    "base_mesh": "./data/sphere.obj"
}
```
We provide two base meshes under `./data` (`sphere.obj` and `ellipsoid.obj`), but you can use any mesh to initialize the geometry.

The default settings are tested under a 32GB V100. 
Lower `batch` and increase `iter` if your GPU memory is limited.

Then you can run training by:
```bash
# single GPU
python train.py --config configs/strawberry.json

# multi GPU (experimental)
torchrun --nproc_per_node 4 train.py --config configs/strawberry.json
```

For single GPU (V100), it takes about 4 hours to train a single model (5000 iters at batch size of 8).

The validation/checkpoints/final mesh will be stored to `./out/<out_dir>`

### Implementation Notes

* the major logic of SDS is hard coded in `geometry/dmtet.py`.
* It seems important to apply antialiasing on the normal and visibility mask (`render/render.py`) so the gradient can be propagated to DMTet.

### Acknowledgement
* The awesome original paper:
```bibtex
@misc{chen2023fantasia3d,
      title={Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation}, 
      author={Rui Chen and Yongwei Chen and Ningxin Jiao and Kui Jia},
      year={2023},
      eprint={2303.13873},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

* [Nvdiffrec](https://github.com/NVlabs/nvdiffrec) codebase.

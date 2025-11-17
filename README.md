<div align="left">
<h1 align="center">PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image
</h1>
<p align="center"><a href="https://arxiv.org/abs/2507.12465"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://physx-anything.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=homepage&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/Caoza/PhysX-Mobility'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>
<a href='https://youtu.be/okMms-NdxMk'><img src='https://img.shields.io/youtube/views/okMms-NdxMk'></a>
<div align="center">
    <a href="https://ziangcao0312.github.io/" target="_blank">Ziang Cao</a><sup>1</sup>,
     <a href="https://hongfz16.github.io/" target="_blank">Fangzhou Hong</a><sup>1</sup>,
    <a href="https://frozenburning.github.io/" target="_blank">Zhaoxi Chen</a><sup>1</sup>,
    <a href="https://github.com/paul007pl" target="_blank">Liang Pan</a><sup>2</sup>,
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1</sup>
</div>
<div align="center">
    <sup>1</sup>S-Lab, Nanyang Technological University&emsp; <sup>2</sup>Shanghai AI Laboratory
</div>
<div>



<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="img/teaser.png">
</div>
## 🏆 News

- We release the code of PhysX-Anything and our new dataset PhysX-Mobility 🎉

## PhysX-Anything

### Installation

1. Clone the repo:

```
git clone --recurse-submodules https://github.com/ziangcao0312/PhysX-Anything.git
cd PhysX-Anything 
```

2. Create a new conda environment named `physxgen` and install the dependencies:

```bash
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

**Note**: The detailed usage of `setup.sh` can be found at [TRELLIS](https://github.com/microsoft/TRELLIS)

3. Install the dependencies for Qwen2.5:

```bash
pip install transformer==4.50.0
pip install qwen-vl-utils
pip install 'accelerate>=0.26.0'
```

### Inference

1. Download the pre-train model from [huggingface_v1](https://huggingface.co/Caoza/PhysX-Anything).

```bash
python download.py
```

2. Run the inference code

```bash
python 1_vlm_demo.py            # vlm inference
    --demo_path ./demo          # inputted image path
    --save_part_ply True        # save the geometry of parts 
    --remove_bg False           # Set this to false for RGBA images and true otherwise.
    --ckpt ./pretrain/vlm       # ckpt path
    
python 2_decoder.py             # decoder inference

python 3_split.py               # split the mesh

python 4_simready_gen.py        # convert to URDF & XML
    --voxel_define 32           # voxel resolution
    --basepath ./test_demo      # results path
    --process 0                 # use postprocess
    --fixed_base 0              # fix the basement of object or not
    --deformable 0              # introduce deformable parts or not
```

**Note**: Although our method can generate parts with physical deformable parameters, the deformable components are not stable in MuJoCo. Therefore, we recommend setting the deformable flag to 0 to obtain more reliable simulation results.

## PhysX-Mobility

For more details about our proposed dataset including dataset structure and annotation, please see this [PhysX-Mobility](https://huggingface.co/datasets/Caoza/PhysX-Mobility) and [PhysXNet](https://huggingface.co/datasets/Caoza/PhysX-3D).

## References

If you find PhysX-Anything useful for your work please cite:

```
@article{cao2025physx,
  title={PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image},
  author={Cao, Ziang and Hong, Fangzhou and Chen, Zhaoxi and Pan, Liang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2507.12465},
  year={2025}
}
```

### Acknowledgement

The data and code is based on [PartNet-mobility](https://sapien.ucsd.edu/browse), [Qwen](https://github.com/QwenLM/Qwen3-VL) and [TRELLIS](https://github.com/microsoft/TRELLIS). We would like to express our sincere thanks to the contributors.

## :newspaper_roll: License

Distributed under the S-Lab License. See `LICENSE` for more information.

<div align="center">
  <a href="https://info.flagcounter.com/x0BB"><img src="https://s01.flagcounter.com/map/x0BB/size_s/txt_000000/border_CCCCCC/pageviews_0/viewers_0/flags_0/" alt="Flag Counter" border="0"></a>
</div>

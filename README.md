<div align="center">

<h2>AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis</h2>

 <a href='https://arxiv.org/abs/2302.02088'><img src='https://img.shields.io/badge/ArXiv-2303.02088-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='https://liangsusan-git.github.io/project/avnerf/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


_**[Susan Liang](https://liangsusan-git.github.io/), [Chao Huang](https://wikichao.github.io/), [Yapeng Tian](https://www.yapengtian.com/), [Anurag Kumar](https://anuragkr90.github.io/), [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/)**_

</div>

### Abstract

<b>TL; DR: AV-NeRF enables joint audio-visual synthesis at novel positions and novel view directions.</b>

> Can machines recording an audio-visual scene produce realistic, matching audio-visual experiences at novel positions and novel view directions? We answer it by studying a new task---real-world audio-visual scene synthesis---and a first-of-its-kind NeRF-based approach for multimodal learning. Concretely, given a video recording of an audio-visual scene, the task is to synthesize new videos with spatial audios along arbitrary novel camera trajectories in that scene. We propose an acoustic-aware audio generation module that integrates prior knowledge of audio propagation into NeRF, in which we implicitly associate audio generation with the 3D geometry and material properties of a visual environment. Furthermore, we present a coordinate transformation module that expresses a view direction relative to the sound source, enabling the model to learn sound source-centric acoustic fields. To facilitate the study of this new task, we collect a high-quality Real-World Audio-Visual Scene (RWAVS) dataset. We demonstrate the advantages of our method on this real-world dataset and the simulation-based SoundSpaces dataset. We recommend that readers visit our project page for convincing comparisons.

### RWAVS Dataset
We provide the Real-World Audio-Visual Scene (RWAVS) Dataset.
1. The dataset can be downloaded from the Hugging Face repository: https://huggingface.co/datasets/susanliang/RWAVS.

2. After you download the dataset, you can decompress the `RWAVS_Release.zip`.
    ```
    unzip RWAVS_Release.zip
    cd release/
    ```

3. The data is organized with the following directory structure.
    ```
    ./release/
    ├── 1
    │   ├── binaural_syn_re.wav
    │   ├── feats_train.pkl
    │   ├── feats_val.pkl
    │   ├── frames
    │   │   ├── 00001.png
    |   |   ├── ...
    │   │   ├── 00616.png
    │   ├── source_syn_re.wav
    │   ├── transforms_scale_train.json
    │   ├── transforms_scale_val.json
    │   ├── transforms_train.json
    │   └── transforms_val.json
    ├── ...
    ├── 13
    └── position.json
    ```

    The dataset contains 13 scenes indexed from 1 to 13. For each scene, we provide
    * `transforms_train.json`: camera poses for training.
    * `transforms_val.json`: camera poses for evaluation. We split the data into `train` and `val` subsets with 80% data for training and the rest for evaluation.
    * `transforms_scale_train.json`: normalized camera poses for training. We scale 3D coordindates to $[-1, 1]^3$.
    * `transforms_scale_val.json`: normalized camera poses for evaluation.
    * `frames`: corresponding video frames for each camera pose.
    * `source_syn_re.wav`: single-channel audio emitted by the sound source.
    * `binaural_syn_re.wav`: two-channel audio captured by the binaural microphone. We synchronize `source_syn_re.wav` and `binaural_syn_re.wav` and resample them to $22050$ Hz.
    * `feats_train.pkl`: extracted vision and depth features at each camera pose for training. We rely on V-NeRF to synthesize vision and depth images for each camera pose. We then use a pre-trained encoder to extract features from rendered images.
    * `feats_val.pkl`: extracted vision and depth features at each camera pose for inference.
    * `position.json`: normalized 3D coordinates of the sound source.

    Please note that some frames may not have corresponding camera poses because COLMAP fails to estimate the camera parameters of these frames.

### Training & Evaluation
After downloading the dataset, please modify the `DATA_DIR` and `LOG_DIR` variables in the `run.sh` file. `DATA_DIR` should point to the path where you saved the dataset, and `LOG_DIR` will be used to store all checkpoints as well as results.

Then, you can train and evaluate the model by running:
```
bash run.sh
```
The `run.sh` contains both training and evaluation commands. During training, the script traverses all 13 scenes. Once the training has finished, the program will print the evaluation results for all environments and the overall performance.

### V-NeRF
We utilize the nerfacto model provided by `nerf-studio` as the V-NeRF. Please refer to [nerfstudio installation](https://docs.nerf.studio/quickstart/installation.html) for detailed guidance on installing `nerfstudio` and `tiny-cuda-nn`. You can train a NeRF for a given environment by running:
```
ns-train nerfacto --output-dir xxx --data xxx --max-num-iterations 100000 --viewer.quit-on-train-completion True
```

### Citation
```bib
@inproceedings{liang23avnerf,
 author = {Liang, Susan and Huang, Chao and Tian, Yapeng and Kumar, Anurag and Xu, Chenliang},
 booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
 title = {AV-NeRF: Learning Neural Fields for Real-World Audio-Visual Scene Synthesis},
 year = {2023}
}
```

### Acknowledgment
We borrowed a lot of code from [NAF](https://proceedings.neurips.cc/paper_files/paper/2022/file/151f4dfc71f025ae387e2d7a4ea1639b-Paper-Conference.pdf) and [INRAS](https://proceedings.neurips.cc/paper_files/paper/2022/file/35d5ad984cc0ddd84c6f1c177a2066e5-Paper-Conference.pdf). We thank the authors for sharing their code. If you use our codes, please also consider citing their nice works.

### Contact
If you have any comments or questions, feel free to contact [Susan Liang](mailto:sliang22@ur.rochester.edu) and [Chao Huang](mailto:chuang65@ur.rochester.edu).

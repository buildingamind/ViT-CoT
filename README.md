<h1 align="center">Are Vision Transformers More Data Hungry Than Newborn Visual Systems?</h1>
<p align="center"> Lalit Pandey, Samantha M. W. Wood, Justin N. Wood </p>

### Accepted Conference: NeurIPS 2023


<img src='./media/main.png'>


## Abstract
Vision transformers (ViTs) are top-performing models on many computer vision benchmarks and can accurately predict human behavior on object recognition tasks. However, researchers question the value of using ViTs as models of biological learning because ViTs are thought to be more “data hungry” than brains, with ViTs requiring more training data to reach similar levels of performance. To test this assumption, we directly compared the learning abilities of ViTs and animals, by performing parallel controlled-rearing experiments on ViTs and newborn chicks. We first raised chicks in impoverished visual environments containing a single object, then simulated the training data available in those environments by building virtual animal chambers in a video game engine. We recorded the first-person images acquired by agents moving through the virtual chambers and used those images to train self-supervised ViTs that leverage time as a teaching signal, akin to biological visual systems. When ViTs were trained “through the eyes” of newborn chicks, the ViTs solved the same view-invariant object recognition tasks as the chicks. Thus, ViTs were not more data hungry than newborn visual systems: both learned view-invariant object representations in impoverished visual environments. The flexible and generic attention-based learning mechanism in ViTs—combined with the embodied data streams available to newborn animals—appears sufficient to drive the development of animal-like object recognition.

## About ViT-CoT

<h3>Encoder (ViT):</h3><p>The image is first divided into smaller 8x8 patches and then reshaped into a sequence of flattened patches. A learnable positional embedding is added to each flattened patch, and a class token (CLS_Token) is added to the sequence. The resulting embedding is then sequentially processed by transformer blocks while also being analyzed in parallel by attention heads, which generate attention filters shown next to each head. The learned representation of the image is adjusted based on a contrastive learning through time loss function.</p>

<img src="./media/encoder.png" style="height:300px">

<h3>Contrastive Learning Through Time Loss Function (CoT):</h3><p>
ViT-CoT leverages the temporal structure of embodied data streams. Learning occurs by making successive views (images seen in a 300 ms temporal window) more similar in the embedding space.
</p>

<img src="./media/loss_func.png" style="height:200px">

## Code Base Organization
```
ViT-CoT
└── datamodules: directory containing python code to set up the datasets and dataloaders to train and test the model.
│   ├── image_pairs.py - set dataloader to train the model
│   ├── invariant_recognition.py - set dataloader to evaluate the model using a linear probe
│
└── models: directory containing python code to set up the architecture for training and evaluating the model.
│   ├── vit_contrastive.py - contains ViT-CoT architecture
│   ├── train_vit.py - contains model training loop
│   ├── common.py - contains linear probe architecture
│   ├── evaluate.py - contains encoder+linear probe evaluation code
│
└── notebooks: Jupyter notebook files used for creating the graphs and attention head visualizations (as shown in the paper).
│   └── plotGraphs.ipynb
│   └── visualizeAttentionHeads.ipynb
│    
└── scripts: directory containing scripts to train and test the model.
│   └── train_vit.sh
├── media: directory containing images and videos for the readme
```

## Environment Set Up
The proposed model in this paper is primarily implemented using <a href='https://lightning.ai/docs/pytorch/stable/'> PyTorch Lightning</a> and <a href="https://pytorch.org/"> Pytorch</a>. All the dependencies are listed in the <i><u>requirements.txt</u></i> file.

```python
pip install -r requirements.txt
```

## Model Training

The bash script 'train_vit.sh' is used to train the model. The bash script calls the python script which intializes the model and the dataloader.

```python
sh train_vit.sh
```

<p>Example:</p>

The following example shows the list of arguments in the bash script used to train the model.

<img src="./media/bash.png" style="height:200px">


## Model Testing

<p> To test the ViT-CoT model (encoder), a linear probe is attached to the frozen encoder. Script <b>'evaluate.py'</b> is used to freeze the encoder and attach the linear probe to evaluate the model using a k-fold analysis.</p>

```python
python evaluate.py --data_dir "PATH_TO_DATA" --exp_name "EXP_NAME" --model "vit" --model_path "PATH_TO_CKPT" --max_epochs 100 --num_folds 12 --identifier "12fold" --project_name "PROJECT_NAME" --shuffle True
```

## Plot Results

<ol>
<li> <u><b>plotGraphs.ipynb</b></u> - Jupyter notebook containing python code to create bar graphs and line graphs (as shown in the paper). </li>

<br>
<li> <u><b>visualizeAttentionHeads.ipynb</b></u> - Jupyter notebook containing python code to show the internal representations (saliency maps) for each attention head (as shown in the paper). </li>

<br>
<p> Example: </p>
<img src="./media/vis.png" style="height:250px">

</ol>


## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/L-Pandey"><img src="https://avatars.githubusercontent.com/u/90662028?v=4?s=100" width="100px;" alt=""/><br /><b>Lalit Pandey</b></td>
    <td align="center"><a href="https://github.com/smwwood"><img src="https://avatars.githubusercontent.com/u/90662028?v=4?s=100" width="100px;" alt=""/><br /><b> Samantha M. W. Wood</b></td>
    <td align="center"><a href="https://github.com/justinnwood"><img src="https://avatars.githubusercontent.com/u/90662028?v=4?s=100" width="100px;" alt=""/><br /><b> Justin N. Wood</b></td>
  </tr>

</table>

  </tr>
</table>

## Citation 

```
@inproceedings{
pandey2023are,
title={Are Vision Transformers More Data Hungry Than Newborn Visual Systems?},
author={Lalit Pandey and Samantha Marie Waters Wood and Justin Newell Wood},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=W23ZTdsabj}
}
```
# Geo-Neus
![Example 1](media/scan24.gif)
![Example 1](media/scan40.gif)

![Example 2](media/scan63.gif)
![Example 1](media/scan110.gif)

This is the official repo for the implementation of [Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction](https://arxiv.org/abs/2205.15848), Qiancheng Fu*, Qingshan Xu*, Yew-Soon Ong, Wenbing Tao (* Equal Contribution), NeurIPS 2022.  
We will release the files on sparse points, image pairs and pretrained models soon!

## Setup  
This code is built with pytorch 1.11.0 and pytorch3d 0.6.2. In addition, other packages listed in requirements.txt are required.  
You can create an anaconda environment called `geoneus` with the required dependencies by running:
```  
conda create -n geoneus python=3.7  
conda activate geoneus  
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch  
conda install fvcore iopath  
conda install -c bottler nvidiacub  
conda install pytorch3d -c pytorch3d  
pip install -r requirements.txt  
```  
## Running 
* Training  
```
python exp_runner.py --mode train --conf ./confs/womask.conf --case <case_name>
```  
* Extract surface from trained model
```
python exp_runner.py --mode validate_mesh --conf ./confs/womask.conf --case <case_name> --is_continue
```
* Evaluation
```
python eval.py --conf ./confs/womask.conf --case <case_name>
```
## Citation
If you find our work useful in your research, please consider citing:
```
@article{Fu2022GeoNeus,  
  title={Geo-Neus: Geometry-Consistent Neural Implicit Surfaces Learning for Multi-view Reconstruction}, 
  author={Fu, Qiancheng and Xu, Qingshan and Ong, Yew-Soon and Tao, Wenbing}, 
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
## Acknowlegement
Our code is partially based on [NeuS](https://github.com/Totoro97/NeuS) project and some code snippets are borrowed from [NeuralWarp](https://github.com/fdarmon/NeuralWarp). Thanks for these great projects.

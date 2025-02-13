# Non-Euclidean Spectral-Spatial feature mining network with Gated GCN-CNN for hyperspectral image classification


<a href="https://orcid.org/0000-0002-2300-7112">Zhang Zhen <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>,<a href="https://orcid.org/0009-0008-4239-2088"> Huang Lehao <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>,<a href="https://orcid.org/0000-0002-1918-5346"> Tang Bo-Hui <img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>,<a href="https://orcid.org/0000-0001-5820-5357"> Wang Qingwang<img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>,<a href="https://orcid.org/0000-0002-7671-3723"> Ge Zhongxi<img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>,<a href="https://orcid.org/0009-0006-7723-478X"> Jiang Linhuan<img alt="ORCID logo" src="https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png" width="16" height="16" /></a>


<h2>Overview</h2>

The proposed NESSGGCN framework for hyperspectral image classification is shown in Figure. The entire process comprises five key components: the Linear Discriminant Analysis-based Simple Linear Iterative Clustering (LDA-SLIC) superpixel segmentation, the Denoising Module, the CNN branch, the Spectral-Spatial Gated GCN, and the Classifier. \
Initially, the original hyperspectral image undergoes superpixel segmentation using the modified LDA-SLIC algorithm. This process yields the transformation matrix Q, which encodes the original image into a superpixel feature map. Additionally, this step also generates two adjacency matrices: the superpixel spectral adjacency matrix Aspe and the superpixel spatial adjacency matrix Aspa, which are subsequently input into the Spectral-Spatial Gated GCN branch. Parallelly, the original image is processed through a denoising module for noise reduction and dimensionality reduction. The denoised feature vectors are then fed into two separate branches: the pixel-level CNN branch and the super-pixel-level Spectral-Spatial Gated GCN branch for feature extraction, respectively. Ultimately, the two output feature vectors from both branches undergo weight fusion, and the fused feature are input into the classifier for final classification. 

![image](https://github.com/user-attachments/assets/895c7c3f-bf81-47ef-b58b-2c99641e2a45)


## Installation

This project is implemented with Pytorch and has been tested on version\
- Pytorch 2.3,
- numpy 1.24.3,
- matplotlib 3.7.2,
- scikit-learn 1.3.0

## Run
If you want to run this code, please change **'your path'** in **main.py** to your correct path, and enter the correct path and corresponding parameters in the **config** file.\
Before running the code, please create three empty folders to store the results, named *pic*, *weights*, and *results*, respectively.

> python main.py 
## Citation
Please kindly cite the papers [Non-Euclidean Spectral-Spatial feature mining network with Gated GCN-CNN for hyperspectral image classification](https://www.sciencedirect.com/science/article/abs/pii/S0957417425004336) if this code is useful and helpful for your research.
```
@ARTICLE{
  author={Zhen Zhang, Lehao Huang, Bo-Hui Tang, Qingwang Wang, Zhongxi Ge, Linhuan Jiang},
  journal={Expert Systems with Applications}, 
  title={Non-Euclidean Spectral-Spatial feature mining network with Gated GCN-CNN for hyperspectral image classification}, 
  year={2025},
  volume={},
  number={},
  pages={126811},
  doi={https://doi.org/10.1016/j.eswa.2025.126811}
}
```


<h2>Code acknowledgments</h2>

We acknowledge the following code repositories that helped to build the NESSGGCN repository :  

- https://github.com/quanweiliu/WFCG

- https://github.com/yuweihao/MambaOut



Thank you! If there are any that have not been mentioned, please contact me to add them.

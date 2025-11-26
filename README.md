# ML-project
## 使用conda配置环境
1. 创建新的conda环境
conda create -n rechorus python=3.10.4

2. 激活环境
conda activate rechorus

3. 安装PyTorch和CUDA工具包
conda install pytorch==1.12.1 torchvision cudatoolkit=10.2.89 -c pytorch

4. 安装其他依赖包
pip install numpy==1.22.3 ipython==8.10.0 jupyter==1.0.0 tqdm==4.66.1 pandas==1.4.4 scikit-learn==1.1.3 scipy==1.7.3 PyYAML

5. 安装ANS-Recbole的依赖
pip install hyperopt==0.2.5 scikit_learn>=0.23.2 pyyaml>=5.1.0 colorlog==4.7.2 colorama==0.4.4 tensorboard>=2.5.0 thop>=0.1.1.post2207130030 ray>=1.13.0 tabulate>=0.8.10 plotly>=4.0.0 texttable>=0.9.0
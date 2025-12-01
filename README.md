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

运行模型
（程序现在能在多处自动查找数据：默认会使用 `--path` 指定的位置；若找不到会尝试仓库中 `ANS-Rechorus/data/` 和 `src/data/`，因此大多数情况下你无需额外指定 `--path`。但是在我的环境中会出现莫名其妙的找不到，需要额外指定`--path`，如果无法运行就使用下面的指令。）
```bash
cd src
python main.py --model_name ANS --dataset Grocery_and_Gourmet_Food --path ../data/

python src/main.py --model_name ANS --dataset Grocery_and_Gourmet_Food

# 说明与容错
- 在 `src/` 下运行时，推荐显式传 `--path ../data/` 或使用默认（已经支持自动查找）。
- 若你从仓库根目录运行 `python src/main.py ...`，脚本也会自动尝试在 `ANS-Rechorus/data/` 找到数据文件。
```



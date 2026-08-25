# PCR-99: A Practical Method for Point Cloud Registration with 99 Percent Outliers

[Paper](https://arxiv.org/abs/2402.16598)

This is the Python implementation of our work **"PCR-99: A Practical Method for Point Cloud Registration with 99 Percent Outliers"**. 

It uses Pytorch for GPU parallelization.

This implementation specifically assumes that the scale is known to be 1.

For the unscaled problem, see our matlab code [here](https://github.com/sunghoon031/PCR-99).

## Quick start:
1. Download the processed datasets [here](https://drive.google.com/drive/folders/1a17qggTB9dEfqe6kldd9xuPOtNKbcJFA?usp=sharing).
2. On terminal, type
````
conda env create -f environment.yml
conda activate PCR99
````
3. To run, type
````
python run_pcr99c.py PATH_TO_TXT_FILE.txt   --sigmas CHOSEN_THRESHOLD    --thr1 0.1   --thr2 5   --n-hypo 1024
````

## Results on my laptop:
| Dataset        | Threshold to use | mAA at 5 deg | mAA at 10 deg | mAA at 15 deg | Average time on my laptop (s) |
|----------------|------------------|--------------|---------------|---------------|-------------------------------|
| 3DMatch + FPFH | 0.02             | 0.511029     | 0.631608      | 0.684822      | 0.684043                      |
| 3DMatch + FCGF | 0.01             | 0.623660     | 0.761861      | 0.817375      | 0.218386                      |
| KITTI + FPFH   | 0.25             | 0.952072     | 0.973514      | 0.981261      | 0.192549                      |
| KITTI + FCGF   | 0.4              | 0.987387     | 0.991892      | 0.993393      | 0.184949                      |

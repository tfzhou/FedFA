# FedFA: Federated Feature Augmentation

This is the PyTorch implementation of our [ICLR 2023](https://iclr.cc/) paper: [FedFA: Federated Feature Augmentation](https://openreview.net/pdf?id=U9yFP90jU0)
by [Tianfei Zhou](https://www.tfzhou.com/) and [Ender Konukoglu](https://scholar.google.com/citations?user=OeEMrhQAAAAJ&hl=en), from ETH Zurich.

## ChangeLog

* [2023-02-24] Code released with reimplementations of experiments

## Preparation

### Environment

* python/3.10.4
* pytorch/1.11.0
* cuda/11.3.1
* gcc/6.3.0
* numpy/1.22.4
* scipy/1.8.1
* opencv-python/4.5.5
* wandb (for experiment tracking)

### Dataset

* Office-Caltech 10: https://drive.google.com/file/d/1gKd0M2Z6gdFwHntgsMTq5ZF5tQIg_bUD/view?usp=share_link
* Prostate MRI: https://drive.google.com/file/d/1GHy3hWAIU288Lr-NF80lydoJQmC7zlmk/view?usp=share_link
* DomainNet: TBD

## Training

Bash training scripts for `prostate_mri` and `office-caltech` are provided in `run_scripts`. 
They are written for our cluster, but can be easily adapted
for any kind of training environments. 
For scripts of other datasets, they are very similar with existing ones; or I will update them later.


Specify data path in Line 11 or Line 12 in `config/prostate_mri/base.py`:
```python
self.DIR_ROOT = os.environ.get('TMPDIR')
self.DIR_DATA = os.path.join(self.DIR_ROOT, 'ProstateMRI')
```

Enter the directory of shell scripts, e.g., `run_scripts/euler/prostate_mri`:
```python
cd run_scripts/euler/prostate_mri
```

Run training:
```python
bash euler_train_fedfa.sh
```

## Benchmark

### Benchmark for ProstateMRI

|   Algorithm  | Round | BIDMC | HK |  I2CVB | BMC | RUNMC | UCL | Average | Log | Ckpt |
| :----------: | :---: | :----: | :-----: | :---: | :----: | :----: | :----: | :-----: | :-----: | :-----: | 
| FedAvg       |  500  |  82.60 |  91.59  | 89.55 | 82.00  |  90.44  | 86.27  | 87.08 | [log](https://drive.google.com/file/d/1tcRvmauf8M8i2yZAvUwA0N9HjLf7Us_R/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1R31tLI0thRbtgf6JV8fLzAeKYoXttzOy?usp=sharing) |
| FedAvgM      |  500  |  83.00 |  91.56  | 88.27 | 82.29  |  90.39  | 84.82  | 86.72 | [log](https://drive.google.com/file/d/1d499qOyZ769HNCp8jJ-sGxRYcOyznc33/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1hfXLvjnh-AhGkSYRXin8PuHIVVEp_sf0?usp=sharing) |
| FedProx      |  500  |  83.61 |  88.31  | 89.45 | 80.93  |  88.13  | 86.36  | 86.13 | [log](https://drive.google.com/file/d/1dRrnuRZiJk7YK_rXO8ZP8pb7sHd_TLe5/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1JZ9HCGQ46h0JnOC_rw6lCeq3LOUlRD7D?usp=sharing) |
| FedSAM       |  500  |  82.14 |  92.49  | 91.48 | 84.61  |  92.96  | 87.47  | 88.52 | [log](https://drive.google.com/file/d/1qErzee3sn6Zz09IqCbO1ERlsM83t4jaI/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1Ut1eecgUO-Wfn9IBOzkHvTZG0xE0emew?usp=sharing) |
| FedDyn       |  500  |  78.09 |  84.24  | 81.13 | 76.61  |  82.46  | 75.80  | 79.72 | [log](https://drive.google.com/file/d/1Ro5cpV7F0rcOH_bEYrDk9vY29THdQrDF/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1V-FoRruu-mpYhDWiD0AoniXNhj8fQn4M?usp=sharing) |
| FedFA        |  500  |  89.18 |  92.64  | 90.08 | 89.16  |  90.91  | 87.71  | 89.95 | [log](https://drive.google.com/file/d/1jWDVzjWdgc1L7xErR6nwDP6rBoatbkiF/view?usp=share_link) | [ckpt](https://drive.google.com/drive/folders/1gf9mv4614i-7HznClTFAeek2zoozrwRO?usp=sharing) |


### Benchmark for Office-Caltech 10

|   Algorithm  | Round | Amazon | Caltech |  DSLR | Webcam | Average | Log | Ckpt |
| :----------: | :---: | :----: | :-----: | :---: | :----: | :-----: | :-----: | :-----: | 
| FedAvg       |  400  |  84.38 |  64.44  | 75.00 | 91.53  |  78.84  | [log](https://drive.google.com/file/d/17_AX7Zqn3oQkdwU3GNVUkvMQ-54YUU8S/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1oFrMcg0V0YlWQbHQokd3BJVlWMzr_OYh?usp=sharing) |
| FedAvgM      |  400  |  80.21 |  65.78  | 75.00 | 91.53  |  78.13  | [log](https://drive.google.com/file/d/1-T2coqbhEzyP4J0NA11rajUiPp3jKDq-/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1tF2-YSK1kDHMadNX41xqBy1uDMxMO9Z-?usp=sharing) |
| FedProx      |  400  |  83.85 |  63.56  | 78.12 | 94.92  |  80.11  | [log](https://drive.google.com/file/d/18rPi5Xw4HvEgjoi7sDeGNu8Ryg6EUILE/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1pkv5V5NGyz248-Hnvs3C220PqGb7g7YT?usp=sharing) |
| FedSAM       |  400  |  81.25 |  61.78  | 68.75 | 83.05  |  73.71  | [log](https://drive.google.com/file/d/1cF73qHB7nwz0nJM562gkO4FBQ2wffxIP/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1KlXwopWq3gUtOWwGur-P5rUJ9CMPNT1T?usp=sharing) |
| FedDyn       |  400  |  79.17 |  60.89  | 65.62 | 88.14  |  73.45  | [log](https://drive.google.com/file/d/1ivip69PLeWISZQbXsFBef5s4LUetak-5/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/10644Vq15zBsIFf05nwgorB26TPv1M1I6?usp=sharing) |
| FedFA        |  400  |  82.81 |  67.11  | 90.62 | 93.22  |  83.44  | [log](https://drive.google.com/file/d/13qLmkNPGowU-3hItFVoq3k3UrLS88qRa/view?usp=sharing) | [ckpt](https://drive.google.com/drive/folders/1gf9mv4614i-7HznClTFAeek2zoozrwRO?usp=sharing) |





## Citation

If you find this work useful, please consider citing:

```
@inproceedings{zhou2023fedfa,
  title={Fed{FA}: Federated Feature Augmentation},
  author={Tianfei Zhou and Ender Konukoglu},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

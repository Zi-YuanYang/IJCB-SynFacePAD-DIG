# IJCB-SynFacePAD-DIG

2nd solution of IJCB 2023 Face Presentation Attack Detection based on Privacy-aware Synthteic Training Data Competition (SynFacePAD-2023).

The proposed Co-Former consists of three Transformer-based branches to extract different-level semantic features, including shallow, normal, and deep semantic features, by tiny, small, and normal-scale swin-transformer. Then the extracted features are concatenated and processed by two linear layers to cooperatively predict the final results. Besides, we also propose a reflection simulation method to augment the data for imitating the reflective effect caused by the material in practice. Concretely, we randomly select the coordinate index as the reflection center, and then utilize a 2-D Gaussian distribution to stimulate diffusion reflections.

The model can be found in ```model.py```, and the augmentation method can be found in ```dataset.py```.

#### Train

Our method can be directly trained by train_Yang_wh_aug.py  with our augmentation method.

```
python train_Yang_wh_aug.py --prefix Yang_new_trans_1024_aug --model_name Yang --batch_size 14 --intra_fe 1024
```

Or you can train without our augmentation method.

```
python train_Yang_wo_aug.py --prefix Yang_new_trans_1024_wo_aug --model_name Yang --batch_size 14 --intra_fe 1024
```

#### Pretrained Model

The pretrained models can be downloaded at https://pan.baidu.com/s/1gr-Km0McFIVbr-ee-QWncA Password：meof

#### Acknowledgments

Many thanks to the competition organizers, especially Dr. Meiling Fang, who contributed so much to this competition. Besides, I also want to thank all my cooperators, who helped this project a lot.

If you have any question or suggestion to our method, please feel free to contact us.

#### Citation

If you think our codes are valuable to your works, please cite the competition paper.

```
@inproceedings{ijcb2023synfacepad,
  title={SynFacePAD 2023: Competition on Face Presentation Attack Detection Based on Privacy-aware Synthetic Training Data},
  author={Fang, Meiling and Huber, Marco and Fierrez, Julian, Ramachandra, Raghavendra and Damer, Naser and Alkhaddour, Alhasan and Kasantcev, Maksim and Pryadchenko, Vasiliy and Yang, Ziyuan and Huangfu, Huijie and Chen, Yingyu and Zhang, Yi and others},
  booktitle={2023 IEEE International Joint Conference on Biometrics (IJCB)},
  year={2023},
  organization={IEEE}
}
```


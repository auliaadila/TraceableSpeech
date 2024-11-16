# TraceableSpeech
PyTorch Implementation of [TraceableSpeech: Towards Proactively Traceable Text-to-Speech with Watermarking](https://arxiv.org/abs/2406.04840)

Now we update the part of speech watermarking.

This is the watermark training pipeline. 

<!--
## Quick Started
### Dependencies
```
pip install -r requirement.txt
```

### Default Preparation
We are using the [LibriTTS](https://openslr.org/60/) dataset.

Modify the parameter `--input_training_file` `--input_validation_file` `--checkpoint_path` in `train.py`

Modify the parameter `--input_wavs_dir` `--output_dir` `--checkpoint_file` in `inference.py`

Modify the config.json

### Watermark config
The watermark configuration is in the `watermark.py` file, defaulting to 4-digit base-16.

### Train
```
python train.py
```

### Test
```
python inference.py
```

### Acknowledgements
This implementation uses parts of the code from the following Github repos: [AcademiCodec](https://github.com/yangdongchao/AcademiCodec)

### Citations
If you find this code useful in your research, please consider citing:
```
@misc{zhou2024traceablespeechproactivelytraceabletexttospeech,
      title={TraceableSpeech: Towards Proactively Traceable Text-to-Speech with Watermarking}, 
      author={Junzuo Zhou and Jiangyan Yi and Tao Wang and Jianhua Tao and Ye Bai and Chu Yuan Zhang and Yong Ren and Zhengqi Wen},
      year={2024},
      eprint={2406.04840},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2406.04840}, 
}
```
-->




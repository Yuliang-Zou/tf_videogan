# tf_videogan

A TensorFlow version of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) (Keep updating...)

Since the original dataset is 9 TB, which is almost impossible for any individual to use it directly, I here extract the pre-trained model parameter from t7 format files. 

For now I have implemented the generator only, which can be used to generate videos. They claimed that the generated videos are not photo-realistic, maybe we can add additional loss functions to improve the model in the future.

## Demos

The following videos were generated from sampled noise:

<table><tr><td>
<strong>Beach</strong><br>
<img src='./demo/beach/01.gif'>
<img src='./demo/beach/04.gif'>
<img src='./demo/beach/11.gif'>
<img src='./demo/beach/13.gif'><br>
<img src='./demo/beach/15.gif'>
<img src='./demo/beach/17.gif'>
<img src='./demo/beach/20.gif'>
<img src='./demo/beach/21.gif'><br>
<img src='./demo/beach/31.gif'>
<img src='./demo/beach/44.gif'>
<img src='./demo/beach/46.gif'>
<img src='./demo/beach/54.gif'>
</td><td>
<strong>Golf</strong><br>
<img src='./demo/golf/02.gif'>
<img src='./demo/golf/14.gif'>
<img src='./demo/golf/16.gif'>
<img src='./demo/golf/18.gif'><br>
<img src='./demo/golf/20.gif'>
<img src='./demo/golf/21.gif'>
<img src='./demo/golf/29.gif'>
<img src='./demo/golf/30.gif'><br>
<img src='./demo/golf/37.gif'>
<img src='./demo/golf/40.gif'>
<img src='./demo/golf/47.gif'>
<img src='./demo/golf/60.gif'>
</td></tr><tr><td>
<strong>Train</strong><br>
<img src='./demo/train/14.gif'>
<img src='./demo/train/15.gif'>
<img src='./demo/train/17.gif'>
<img src='./demo/train/24.gif'><br>
<img src='./demo/train/29.gif'>
<img src='./demo/train/35.gif'>
<img src='./demo/train/37.gif'>
<img src='./demo/train/42.gif'><br>
<img src='./demo/train/46.gif'>
<img src='./demo/train/51.gif'>
<img src='./demo/train/52.gif'>
<img src='./demo/train/55.gif'>
</td></tr></table>


## Models

You can find the 'beach' model in the repository. The remaining 2 models can be found [here](https://drive.google.com/drive/folders/0B2SnTpv8L4iLRzdWb2lQdjc2ZFE?usp=sharing)

And you can also extract other 2 models with the code `load_t7.py`. (You will need to install the package [torchfile](https://github.com/bshillingford/python-torchfile) first)

## Notes

You might want to see their original [torch implementation](https://github.com/cvondrick/videogan).

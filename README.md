# tf_videogan

A TensorFlow version of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) (Keep updating...)

Since the original dataset is 9 TB, which is almost impossible for any individual to use it directly, I here extract the pre-trained model parameter from t7 format files. 

For now I have implemented the generator only, which can be used to generate videos. They claimed that the generated videos are not photo-realistic, maybe we can add additional loss functions to improve the model in the future.

## Demos

<table><tr><td>
<strong>Beach</strong><br>
<img src='./demo/01.gif'>
<img src='./demo/04.gif'>
<img src='./demo/11.gif'>
<img src='./demo/13.gif'><br>
<img src='./demo/15.gif'>
<img src='./demo/17.gif'>
<img src='./demo/20.gif'>
<img src='./demo/21.gif'><br>
<img src='./demo/31.gif'>
<img src='./demo/44.gif'>
<img src='./demo/46.gif'>
<img src='./demo/54.gif'>
</td></tr></table>


## Models

You can find the 'beach' model in the repository. And you can also extract other 2 models with the code `load_t7.py`. (You will need to install the package [torchfile](https://github.com/bshillingford/python-torchfile) first)

# Rodino

# Adversarial Attack
```math
F_\theta(\text{Attack}(x ; G_\phi)) = F_\theta(x') \neq y
```
# Adversarial Training
## Dino Architecture 

![](images/dino.gif)


![](images/Rodino.png)



## How to run?


To evaluate the defense model on the classifier,

```
python eval_linear.py --arch vit_small --patch_size 16 --epochs 100 --num_labels 100 --data_path /data/sara/imagenet-100/ --output_dir ./save/ --linear_pretrained_weights ./checkpoint_Dino-reference_linear-010.pth.tar --evaluate
```

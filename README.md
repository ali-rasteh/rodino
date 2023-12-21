# Rodino

# Adversarial Attack
```math
F_\theta(\text{Attack}(x ; G_\phi)) = F_\theta(x') \neq y
```

## How to run?

## PGD Attack 
This function generates the PGD-Linf adversarial images 
```
def generate_attack(target_model, img, label):
    if args.pgd_attack == 'linf':
        adversary = LinfPGDAttack(target_model, loss_fn=None, eps=args.pgd_size, nb_iter=50, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
```

# Adversarial Training
## Dino Architecture 

![](images/dino.gif)


![](images/Rodino.png)

# SphereFace
PyTorch implementation of SphereFace, gradients MODIFIED to make it more stable following exactly Weiyang's Caffe implementation. 

### Notation from Weiyang's repository:
https://github.com/wy1iu/sphereface/blob/master/README.md

1.**Backward gradient**
	- In this implementation, we did not strictly follow the equations in paper. Instead, we normalize the scale of gradient. It can be interpreted as a varying strategy for learning rate to help converge more stably. Similar idea and intuition also appear in [normalized gradients](https://arxiv.org/pdf/1707.04822.pdf) and [projected gradient descent](https://www.stats.ox.ac.uk/~lienart/blog_opti_pgd.html).
	- More specifically, if the original gradient of ***f*** w.r.t ***x*** can be written as **df/dx = coeff_w \*  w + coeff_x \* x**, we use the normalized version **[df/dx] = (coeff_w \* w + coeff_x \* x) / norm_wx** to perform backward propragation, where **norm_wx** is **sqrt(coeff_w^2 + coeff_x^2)**. The same operation is also applied to the gradient of ***f*** w.r.t ***w***.
	- In fact, you do not necessarily need to use the original gradient, since the original gradient sometimes is not an optimal design. One important criterion for modifying the backprop gradient is that the new "gradient" (strictly speaking, it is not a gradient anymore) need to make the objective value decrease stably and consistently. (In terms of some failure cases for gradient-based back-prop, I recommand [a great talk](https://www.youtube.com/watch?v=jWVZnkTfB3c) by [Shai Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/))
	- If you use the original gradient to do the backprop, you could still make it work but may need different lambda settings, iteration number and learning rate decay strategy. 

import torch
import numpy as np
import random

def pool_mix(fake_t, real_t):
    real_t_noise = real_t + torch.randn(real_t.shape) * 0.01
    batch = fake_t.shape[0]
    randint = random.randint(0, batch-1)
    rand_indice = random.sample(range(batch), randint)
    for _ in rand_indice:
        fake_t[_,:,:,:] = real_t_noise[_,:,:,:]
    return fake_t

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images



'''
tensor.detach() creates a tensor that shares storage with tensor that does not require grad. tensor.clone()creates a copy of tensor that imitates the original tensor's requires_grad field.
You should use detach() when attempting to remove a tensor from a computation graph, and clone as a way to copy the tensor while still keeping the copy as a part of the computation graph it came from.

tensor.data returns a new tensor that shares storage with tensor. However, it always has requires_grad=False (even if the original tensor had requires_grad=True

You should try not to call tensor.data in 0.4.0. What are your use cases for tensor.data?

tensor.clone() makes a copy of tensor. variable.clone() and variable.detach() in 0.3.1 act the same as tensor.clone() and tensor.detach() in 0.4.0.
'''
# https://discuss.pytorch.org/t/clone-and-detach-in-v0-4-0/16861/13

import jax
import jax.numpy as jnp
from flax import linen as nn

class cnn(nn.Module):
    @nn.compact
    def __call__(self,images,training:bool):

        images = nn.Conv(features=32,kernel_size=(3,3),padding="SAME")(images)
        images = nn.BatchNorm(use_running_average=not training)(images)
        images = nn.relu(images)
        images = nn.max_pool(images,window_shape=(2,2),strides=(2,2))

        images = nn.Conv(features=64,kernel_size=(3,3),padding="SAME")(images)
        images = nn.BatchNorm(use_running_average=not training)(images)
        images = nn.relu(images)
        images = nn.max_pool(images,window_shape=(2,2),strides=(2,2))

        images = nn.Conv(features=128,kernel_size=(3,3),padding="SAME")(images)
        images = nn.BatchNorm(use_running_average=not training)(images)
        images = nn.relu(images)
        images = nn.max_pool(images,window_shape=(2,2),strides=(2,2))

        images = jnp.mean(images,axis=(1,2))

        images = nn.Dropout(rate=0.5)(images,deterministic=not training)

        images = nn.Dense(features=10)(images)

        return images


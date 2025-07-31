import jax
import jax.numpy as jnp
import time
import numpy as np

"""
a = jnp.array([1,2,3])
b = jnp.array([4,5,6])

print (a*b)
print(jnp.sqrt(a))
print(jnp.mean(a))
print(a.reshape(-1,1))
"""

@jax.jit
def myfunc(x):
    return jnp.where (x % 2 == 0, x / 2, 3 * x + 1)

def myfuncasnp(x):
    return np.where (x % 2 == 0, x / 2, 3 * x + 1)

""""
testing out jit compilation time: 0.047
testing out numpy compilation time: 0.0003

if we run myfunc before the timer to warm up, we can drop the runtime even more
jax: 1.7042038962244987e-05
np: 9.332899935543537e-06
""""
arr = jnp.arange(10)
start = time.perf_counter()
myfunc(arr)
end = time.perf_counter()
print(end - start)

arr2 = np.arange(10)
start = time.perf_counter()
myfuncasnp(arr2)
end = time.perf_counter()
print(end - start)



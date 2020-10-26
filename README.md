Python version 3.6+ required!

Pipenv doesn't allow for Python version range to be specified...may
change venv system due to that

## Debugging environment (internal notes, temporary...)

TensorFlow
```shell script
# CPU
CUDA_VISIBLE_DEVICES="" python ...
# GPU
LD_LIBRARY_PATH=/opt/cudnn7-cuda10.1/lib/ python ...

# shhh
TF_CPP_MIN_LOG_LEVEL=1 python ...
```

Theano

1. Install [`libgpuarray`](http://deeplearning.net/software/libgpuarray/installation.html)
2. Install `cython`: `pip install cython`
3. Install [`pygpu`](http://deeplearning.net/software/libgpuarray/installation.html):
`cd libgpuarray && pip install -e .`

```shell script
# CPU
THEANO_FLAGS='floatX=float32,device=cpu' python ...
# GPU
THEANO_FLAGS='floatX=float32,device=cuda0' python ...
```

You need a version (not available yet) that provides the `_print_Exp1` method to the theanocode
printer. If not, you'll need to modify `sympy/printing/theanocode.py` as done in
[this PR](https://github.com/sympy/sympy/pull/20335).

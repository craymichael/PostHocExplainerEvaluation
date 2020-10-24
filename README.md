Python version 3.6+ required!

Pipenv doesn't allow for Python version range to be specified...may
change venv system due to that

## Debugging environment (internal notes, temporary...)

```shell script
LD_LIBRARY_PATH=/opt/cudnn7-cuda10.1/lib/ python ...
```

Theano

1. Install [`libgpuarray`](http://deeplearning.net/software/libgpuarray/installation.html)
2. Install `cython`: `pip install cython`
3. Install [`pygpu`](http://deeplearning.net/software/libgpuarray/installation.html):
`cd libgpuarray && pip install -e .`

```shell script
THEANO_FLAGS='floatX=float32,device=cuda0'
```

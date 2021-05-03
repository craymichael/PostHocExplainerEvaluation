Python version 3.6+ required!

Pipenv doesn't allow for Python version range to be specified...may
change venv system due to that

FICO HELOC license prohibits distribution. Obtain from:
https://community.fico.com/s/explainable-machine-learning-challenge

## SHAPR Installation
Install R. Then, in an R shell:
```R
install.packages("shapr")
```

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

## Usage (TODO)
```bash
./generate_expressions.py --seed 42 --n-runs 1 \
    --n-feats-range 2,1024,10,log \
    --kwarg n_dummy              --kwarg-range 0.,0.95,5,linear \
    --kwarg n_interaction        --kwarg-range 0,0.5,4,linear \
    --kwarg interaction_ord      --kwarg-range 2,3,linear \
    --kwarg nonlinear_multiplier --kwarg-range 0.,1.5,5,linear

python generate_data.py experiment_data/expr/generated_expressions_n_dummy-vs-n_interaction-vs-interaction_ord-vs-nonlinear_multiplier_2020-12-24T18_58_01.pkl
python evaluate_explainers.py experiment_data/expr/generated_expressions_n_dummy-vs-n_interaction-vs-interaction_ord-vs-nonlinear_multiplier_2020-12-24T18_58_01.pkl --max-explain 1000

# === TAKE 2(MOST RECENT): ===
generate_expressions.job --> generated_expressions_n_dummy-vs-n_interaction-vs-interaction_ord-vs-nonlinear_multiplier_2021-01-01T22_05_20.pkl

# data
qrsh -pe smp 24
source venv/bin/activate
python generate_data.py \
    /afs/crc.nd.edu/group/cvrl/scratch_40/experiment_data/expr/generated_expressions_n_dummy-vs-n_interaction-vs-interaction_ord-vs-nonlinear_multiplier_2021-01-01T22_05_20.pkl \
    --n-samples 500 \
    --out-dir /afs/crc.nd.edu/group/cvrl/scratch_40/experiment_data/data/ \
    --n-jobs 24

# metrics
python aggregate_metrics.py /afs/crc.nd.edu/group/cvrl/scratch_40/experiment_data/expr/generated_expressions_n_dummy-vs-n_interaction-vs-interaction_ord-vs-nonlinear_multiplier_2021-01-01T22_05_20.pkl
python aggregate_metrics.py experiment_data/expr/generated_expressions_n_dummy-vs-n_interaction-vs-interaction_ord-vs-nonlinear_multiplier_2021-01-01T22_05_20.pkl

rsync -avh --ignore-existing zcarmich@crcfe02.crc.nd.edu:/afs/crc.nd.edu/group/cvrl/scratch_40/experiment_data/ /mnt/hdd1/posthoceval_data/experiment_data/
```

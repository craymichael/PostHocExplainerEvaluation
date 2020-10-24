import time

import numpy as np

from posthoceval.model_generation import tsang_iclr18_models


def make_data(n_samples, n_features):
    data = np.random.uniform(0, +1, size=(n_samples, n_features))
    data = data.astype('float32')
    return data


def print_stats(setup_times, stats):
    print('setup times')
    max_len = max(len(b) for b in setup_times.keys())
    setup_times = sorted(setup_times.items(), key=lambda x: x[1])
    for backend, dur in setup_times:
        print('  {{:{}}}'.format(max_len).format(backend), dur)
    print('stats')
    stats = sorted(stats.items(), key=lambda x: x[0])
    for n_runs, run_stats in stats:
        print('For {} samples.'.format(n_runs))
        run_stats = sorted(run_stats.items(), key=lambda x: x[1])
        print('  {{:{}}}'.format(max_len).format(''),
              ''.join('{:>10}'.format(s)
                      for s in ('mean', 'std', 'median', 'min', 'max')))
        for backend, durs in run_stats:
            print('  {{:{}}}'.format(max_len).format(backend),
                  ''.join('{: 10.3g}'.format(dur) for dur in durs))


def benchmark(gpu=False, debug=False):
    print('GPU' if gpu else 'CPU')
    cpu_backends = (
        'numpy',
        'theano',
        'tensorflow',
        'numexpr',
        'f2py',
        'cython',
        'ufuncify_numpy',
    )
    gpu_backends = (
        'theano',
        'tensorflow',
    )

    backends = gpu_backends if gpu else cpu_backends

    sample_sizes = (100,) if debug else (100, 1_000, 10_000, 100_000, 1_000_000)

    all_setup_times = {}
    all_stats = {}
    for i in range(1, 11):
        # Make model
        model_name = 'f' + str(i)
        model = tsang_iclr18_models(model_name)
        msg = f'Benchmark for Tsang et al. Equation {model_name}'
        print()
        print('=' * len(msg))
        print(msg)
        model.pprint()
        print('=' * len(msg))

        # Single dummy run to measure potential setup times
        setup_times = {}
        dummy_data = make_data(1, model.n_features)
        for backend in backends:
            print('drugs', backend)
            t0 = time.perf_counter()
            model(dummy_data, backend=backend)
            dur = time.perf_counter() - t0
            setup_times[backend] = dur

            all_durs = all_setup_times.get(backend, [])
            all_durs.append(dur)
            all_setup_times[backend] = all_durs

        # Benchmark
        stats = {}
        for n_samples in sample_sizes:
            # Make data
            data = make_data(n_samples, model.n_features)

            stats_backend = {}
            for backend in backends:
                # Average over this many times
                n_runs = max(10_000_000 // n_samples, 5)

                print(f'{backend} - {n_samples} samples - {n_runs} runs')

                durs = []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    model(data, backend=backend)
                    durs.append(time.perf_counter() - t0)
                durs = np.asarray(durs)
                stats_backend[backend] = (
                    durs.mean(), durs.std(), np.median(durs),
                    durs.min(), durs.max()
                )

                all_backend = all_stats.get(n_samples, {})
                all_durs = all_backend.get(backend, [])
                all_durs.extend(durs)
                all_backend[backend] = all_durs
                all_stats[n_samples] = all_backend
            stats[n_samples] = stats_backend

        print()
        print_stats(setup_times, stats)

    print()
    print_stats(all_setup_times, all_stats)


if __name__ == '__main__':
    def main():
        import argparse
        from textwrap import dedent

        parser = argparse.ArgumentParser(
            description=dedent('''\
            # CPU
            THEANO_FLAGS='floatX=float32,device=cpu' CUDA_VISIBLE_DEVICES="" python ...
            # GPU
            THEANO_FLAGS='floatX=float32,device=cuda0' LD_LIBRARY_PATH=/opt/cudnn7-cuda10.1/lib/ python ...\
            ''')
        )
        parser.add_argument('--gpu', action='store_true',
                            help='Assume GPU mode.')
        parser.add_argument('--debug', action='store_true',
                            help='Debug mode - use fewer samples.')
        args = parser.parse_args()

        benchmark(args.gpu, debug=args.debug)


    main()

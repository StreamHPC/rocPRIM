import alive_progress
import alive_progress.styles
import glob
import itertools
import json
import os
import scipy.optimize
import shutil
import subprocess

# Dual annealing is best for tuning:
# - "Benchmarking optimization algorithms for auto-tuning GPU kernels" by Schoonhoven et al, 2022.
# - "A methodology for comparing optimization algorithms for auto-tuning" by Willemsen et al, 2024.

parameter_spaces = {
    'device_segmented_radix_sort_keys': {
        'benchmark': 'benchmark_device_segmented_radix_sort_keys',
        'types':{
            'KeyType': ['int64_t', 'int', 'short', 'int8_t', 'double', 'float', 'rocprim::half'],
        },
        'params': {
            'LongBits' : [7, 8],
            'ShortBits' : [4, 5, 6],
            'BlockSize': [256],
            'ItemsPerThread': [7, 8, 13, 16, 17],
            'WarpSmallLWS': [8, 16, 32],
            'WarpSmallIPT': [4, 8],
            'WarpSmallBS': [256],
            'WarpPartition': [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            'WarpMediumLWS': [16, 32],
            'WarpMediumIPT': [4, 16],
            'WarpMediumBS': [256]
        }
    }
}

print_intermediate_configs = False

def get_result_from_json(filename:os.PathLike):
    with open(filename, 'r') as file:
        data = json.load(file)
        return data['benchmarks'][0]['bytes_per_second']

def spinner(title):
    return alive_progress.alive_bar(monitor=None, stats=None, title=title, bar=None, spinner=alive_progress.styles.SPINNERS['classic'])

def merge_jsons(source_filenames, target_filename):
    merged = {
        'context': {},
        'benchmarks': [],
    }

    # collect jsons
    for filename in source_filenames:
        with open(filename, 'r') as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError as e:
                print(f'Skipping file \'{filename}\' because of error: {e}')
            # HACK: we reuse the last context since we can only have one
            merged['context'] = data['context']
            # append benchmark data
            merged['benchmarks'].append(data['benchmarks'])

    # write out file
    with open(target_filename, 'w') as file:
        json.dump(merged, file, indent=2)

def tune(alg_name:str, arch:str, max_samples=200):
    script_dir  = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.join(script_dir, '../..')
    build_dir   = os.path.join(project_dir, 'build')
    result_dir  = os.path.join(script_dir, 'artifacts')

    # delete build directory
    try:
        shutil.rmtree(build_dir)
    except(FileNotFoundError):
        pass

    try:
        shutil.rmtree(result_dir)
    except(FileNotFoundError):
        pass

    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # get the context of the tuning run
    alg_space = parameter_spaces[alg_name]

    # types to tune, this can be a product of multiple types
    types = [dict(zip(alg_space['types'], ts)) for ts in itertools.product(*[alg_space['types'][type] for type in alg_space['types']])]

    # generate bounds by normalizing the parameter space from discrete to real numbers (relaxation)
    bounds = dict(zip(alg_space['params'], ((0, 1) for _ in alg_space['params'])))

    build_target = alg_space['benchmark']

    def param_from_normalized(name:str, value):

        params = alg_space['params'][name]
        index = min(int(value * len(params)), len(params) - 1)

        try:
            return str(params[index])
        except IndexError as e:
            print(f'Could not find parameter \'{name}\' at \'{index}\' derived from value \'{value}\' in {params}.')
            raise e

    for type in types:
        cache = {}

        def sample(xs):
            # delete *.parallel folder
            try:
                # HACK: we just delete the benchmark folder because it's easier
                shutil.rmtree(os.path.join(build_dir, 'benchmark'))
            except(FileNotFoundError):
                pass

            tune_param_names = list(type.keys()) + list(alg_space['params'])
            tune_param_vals = list(type.values()) + [param_from_normalized(name, value) for (name, value) in zip(alg_space['params'].keys(), xs)]

            result_id = '_'.join(tune_param_vals)
            if result_id in cache:
                return cache[result_id]

            result_filename = f'{arch}_{build_target}_{result_id}.json'

            tune_param_names = ';'.join(tune_param_names)
            tune_param_vals = ';'.join(tune_param_vals)

            # CMake configure
            with spinner('Configuring with CMake') as progress:
                configure = subprocess.call([
                    'cmake',
                    '-S', '.',
                    '-B', build_dir,
                    '-GNinja',
                    '-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++',
                    '-DBUILD_BENCHMARK=ON',
                    '-DBENCHMARK_CONFIG_TUNING=ON',
                    f'-DAMDGPU_TARGETS={arch}',
                    f'-DBENCHMARK_TUNE_PARAM_NAMES={tune_param_names}',
                    f'-DBENCHMARK_TUNE_PARAMS={tune_param_vals}',
                ], cwd=project_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if configure != 0:
                    progress.title = "Configuration failed"
                    cache[result_id] = 0
                    return 0
                progress.title = "Configuring succeeded"
                progress()

            # Build target
            with spinner('Building') as progress:
                build = subprocess.call([
                    'cmake',
                    '--build', '.',
                    '--target', build_target
                ], cwd=build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if build != 0:
                    progress.title = 'Build failed'
                    cache[result_id] = 0
                    return 0
                progress.title = "Build succeeded"
                progress()

            # Run benchmark
            with spinner('Running benchmark') as progress:
                bench = subprocess.call([
                    os.path.join(build_dir, 'benchmark', build_target),
                    '--name_format', 'json',
                    '--seed', 'random', # Random is better... I think? Otherwise we might overfit.
                    '--trials', '20', # 20 seems to be a fair balance between benchmarking and compile time. Needs verification though.
                    '--benchmark_out_format=json',
                    f'--benchmark_out={result_filename}',
                ], cwd=result_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)

                if bench != 0:
                    cache[result_id] = 0
                    return 0
                result_value = get_result_from_json(os.path.join(result_dir, result_filename))
                progress.title = f"Benchmark achieved {result_value / 1e9:.3f} GB/s"
                progress()

            if print_intermediate_configs:
                result_context = {
                    'config': dict(((name, param_from_normalized(name, val)) for name, val in zip(alg_space['params'], xs))),
                    'bytes_per_second': result_value,
                }
                print(json.dumps(result_context, indent=2))

            cache[result_id] = result_value

            # scipy.optimize does minimization, negate result for maximize
            return -result_value

        print(f'Begin tuning {alg_name} for {type}')
        result = scipy.optimize.dual_annealing(
            sample,
            bounds=bounds.values(),
            maxfun=max_samples)
        print(f'Done with tuning {alg_name} for {type}')
        print(result)

    with spinner('Collecting benchmark results') as progress:
        merge_jsons(glob.glob(os.path.join(result_dir, f'{arch}_{build_target}_*.json')), os.path.join(result_dir, f'{arch}_{build_target}.json'))
        progress()

tune('device_segmented_radix_sort_keys', 'gfx1030', 40)

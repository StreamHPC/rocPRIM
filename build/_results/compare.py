#!/usr/bin/env python3

# pip3 install matplotlib natsort numpy

import csv
import natsort
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
import numpy as np
import collections as col
import operator as op
import re
import json

def get_rocprim_indices(row):
    data_label = 'bytes_per_second'
    name_label = 'name'

    if not all(label in row for label in [data_label, name_label]):
        return None

    return ([row.index(name_label)], row.index(data_label))

def get_thrust_indices(row):
    data_label = 'Thrust Average Throughput'
    alg_label  = 'Algorithm'
    type_label = 'Element Type'
    size_label = 'Total Input Size'

    if not all(label in row for label in [data_label, alg_label, type_label, size_label]):
        return None

    return ([row.index(label) for label in [alg_label, type_label, size_label]], row.index(data_label))

def parse_csv(file_path):
    measurements = []
    indices = None

    with open(file_path) as file:
        reader = csv.reader(file)
        for row in reader:
            # detect indices of benchmark name and the data
            # we assume that the row above the actual data is 
            # truthful and the data does not contain similar
            # strings
            if not indices:
                # TODO: this switch is hardcoded
                indices = get_rocprim_indices(row)
                # indices = get_thrust_indices(row)
                continue
            if (row[0] != '') and indices:
                name_indices, data_index = indices
                # TODO: this does not account for '"foo, bar",baz" and will
                # output ['"foo', 'bar"', 'baz']
                name = [item for name_index in name_indices for item in re.sub('\/manual_time$', '', row[name_index]).split(',')]
                measurements.append((name, float(row[data_index])))
        
        if not indices:
            raise Exception("Error: could not find relevant indices in csv!")
    return measurements

def parse_json(file_path):
    with open(file_path) as file:
        measurements = []
        data = json.load(file)
        for benchmark in data['benchmarks']:
            json_name = json.loads(re.sub('\/manual_time$', '', benchmark['name']))
            # opinionated: use 'key=value' for combining json items. we're 
            # also stripping away quotation marks.
            names = [f"{key}={value}" for key, value in json_name.items()]
            bytes_per_second = float(benchmark["bytes_per_second"])
            measurements.append((names, bytes_per_second))
    return measurements

def distil_common(xss):
    '''
    Checks a list of lists for common items. Returns a tuple of list of lists
    with common items removed, and a list of common items.
    '''
    num_xs = len(xss)
    x_count = col.defaultdict(lambda: 0)
    for xs in xss:
        for x in xs:
            x_count[x] += 1
    common_xs = [x for (x, count) in x_count.items() if count == num_xs]
    other_xs  = [[x for x in xs if x not in common_xs] for xs in xss]
    return (other_xs, common_xs)

def format_bandwidth(x, pos=None):
    if  x >= 2 ** 40:
       return '%1.1fTiB/s' % (x / 2 ** 40)
    if  x >= 2 ** 30:
       return '%1.1fGiB/s' % (x / 2 ** 30)
    if  x >= 2 ** 20:
       return '%1.1fMiB/s' % (x / 2 ** 20)
    if  x >= 2 ** 10:
        return '%1.1fKiB/s' % (x / 2 ** 10)
    return  '%1.1fB/s' % x

def data_from_file(file, compare_indices, baseline_name=None, parser=parse_csv):
    groups = dict()
    for row in reversed(parser(file)):
        
        row_names = row[0]
        group_name = ",".join(row_names[i] for i in compare_indices)
        row_names  = [row_name for i, row_name in enumerate(row_names) if i not in compare_indices]
        
        row = list(row)
        row[0] = ','.join(row_names)
        row = tuple(row)

        if not group_name in groups:
            groups[group_name] = [row]
        else:
            groups[group_name].append(row)

    # Try to grab base line
    baseline = None
    if baseline_name:
        if baseline_name not in groups:
            raise Exception('Could not find baseline name!')
        baseline = groups[baseline_name]
        del groups[baseline_name]

    return dict({
        'results'       : list(groups.values()),
        'names'         : list(groups.keys()),
        'baseline'      : baseline,
        'baseline_name' : baseline_name
    })

def data_from_files(files, baseline_file=None, parser=parse_csv):
    return dict({
        'results'       : [parser(file) for file in files],
        'names'         : files,
        'baseline'      : parser(baseline_file) if baseline_file else None,
        'baseline_name' : baseline_file
    })

def plot(results, names, baseline=None, baseline_name=None, output='chart.png'):
    # opinionated: we're doing a natural sort so that numeric names are sorted properly
    names, results = list(zip(*natsort.natsorted(zip(names, results), key=op.itemgetter(0))))

    # make shortened names by looking for differing parts in the file name
    short_names, _ = distil_common([re.split(r'[,\/\\\.]', name) for name in names])
    short_names = [', '.join(short_name) for short_name in short_names]

    # opinionated: 'Set2' is a qualitative color set which is better at
    # showing unordered data sets (i.e. benchmark results)
    color_map = plt.get_cmap('Set2')
    bar_width = 0.9 / len(results)
    fig, ax = plt.subplots(layout='constrained', figsize=(12,len(results[0]) * len(results) * 0.3))
    ax.spines[['right', 'top']].set_visible(False)

    # parse ticks
    ticks, common_xs = distil_common([row[0] for row in results[0]])
    ticks = [', '.join(tick) for tick in ticks]
    x = np.arange(len(ticks))
    ax.set_yticks(x, ticks)

    # set titles
    ax.set_title(', '.join(common_xs))
    if baseline:   
        ax.set_xlabel('speed up (%)')
    else:
        ax.set_xlabel('bandwidth (bytes/s)')
    
    if baseline:
        ax.xaxis.set_major_formatter(mtk.PercentFormatter())
    else:
        loc = mtk.MaxNLocator(steps=[8], integer=True)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mtk.FuncFormatter(format_bandwidth))
    ax.grid(True, which='both', axis='x')

    if baseline:
        ref_line_offset = (1 + len(results)) * bar_width / 2
        ax.plot([0, 0], [-ref_line_offset, len(baseline) - 1 + ref_line_offset], color='black', label=baseline_name)

    # parse y-data
    for i, result in enumerate(results):
        # originally: 
        #   offset = (i - (len(results) - 1) / 2) * bar_width
        # reversed to make ordering similar to legend
        offset = (- i + (len(results) - 1) / 2) * bar_width
        
        # Get data from results
        if baseline:
            data = [(res_row[1] / base_row[1] - 1) * 100 for base_row, res_row in zip(baseline, result)]
        else:
            data = [row[1] for row in result]

        # Plot bar
        bars = ax.barh(x + offset, data, bar_width, color=color_map(i))
        if baseline:
            ax.bar_label(bars, fontsize=8, fmt='{0:.2f}%')
        elif short_names[i]:
            ax.bar_label(bars, fontsize=8, fmt=lambda v: f'{format_bandwidth(v)} - {short_names[i]}')
        else:
            ax.bar_label(bars, fontsize=8, fmt=lambda v: f'{format_bandwidth(v)}')

    legend_labels = list([baseline_name]) + list(names) if baseline else list(names)
    ax.legend(legend_labels, bbox_to_anchor=(0.5,-0.1), loc='upper center')
    fig.savefig(output, bbox_inches='tight', dpi=200)
    print(output)

def plot_2d(series, serie_names, baseline_key=None, output="chart.png"):
    # for data_item, name in zip(data, names):
    #     results = data_item['results']
    #     _, results = list(zip(*natsort.natsorted(zip(_, results), key=op.itemgetter(0))))
        
    data = {}
    color_maps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
    line_styles = ['-', '--', '-.', ':']
    
    # tranpose data
    for serie_name, serie in zip(serie_names, series):
        data[serie_name] = {}
        for benchmark_name, benchmark in zip(serie['names'], serie['results']):
            for benchmark_key, benchmark_value in benchmark:
                y_label = ', '.join(benchmark_key)
                x_label = benchmark_name # todo parse to numeric value
                x_label = benchmark_name.split('-')[2]

                if y_label not in data[serie_name]:
                    data[serie_name][y_label] = {}

                data[serie_name][y_label][x_label] = benchmark_value
    
    # setup figure
    fig, ax = plt.subplots(layout='constrained', figsize=(12, 16))
    ax.spines[['right', 'top']].set_visible(False)

    loc = mtk.MaxNLocator(steps=[4], integer=True)
    
    if baseline_key:
        ax.yaxis.set_major_formatter(mtk.PercentFormatter())
    else:
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(mtk.FuncFormatter(format_bandwidth))
        ax.set_ylabel('bandwidth (bytes/s)')
    ax.grid(True, which='both', axis='y')

    global_x_labels = list(list(data.values())[0].values())[0].keys()

    # collect labels
    legend_labels = []

    if baseline_key:
        ax.plot(global_x_labels, [0 for _ in  global_x_labels], color='black', label=baseline_key, linestyle=':')
        legend_labels.append(baseline_key)

        avg_baseline_data = np.array([list(x.values()) for x in data[baseline_key].values()])
        avg_baseline_data = np.average(avg_baseline_data, axis=0)

    for i, (serie_name, serie) in enumerate(data.items()):
        if serie_name == baseline_key:
            continue

        color_map = plt.get_cmap(color_maps[i], len(serie) * 4)
        line_style = line_styles[i]

        avg_data = np.array([list(x.values()) for x in serie.values()])
        avg_data = np.average(avg_data, axis=0)
        if baseline_key:
            avg_data = (np.divide(avg_data, avg_baseline_data) - 1) * 100
        avg_labels = list(serie.values())[0].keys()

        for j, (benchmark_key, benchmark_values) in enumerate(serie.items()):
            color = color_map(2 * j + len(serie))
            y_label = benchmark_key
            x_labels, x_values = zip(*benchmark_values.items())

            if baseline_key:
                x_values = (np.divide(x_values, list(data[baseline_key][benchmark_key].values())) - 1) * 100

            ax.plot(x_labels, x_values, color=color, linestyle=line_style)

            legend_labels.append(y_label)
        
        ax.plot(avg_labels, avg_data, color='black', linewidth=1.5, linestyle=line_style)
        legend_labels.append(f"average {serie_name}")

    ax.legend(legend_labels, bbox_to_anchor=(0.5,-0.1), loc='upper center')
    
    fig.savefig(output, bbox_inches='tight', dpi=200)
    print(output)

# ---

sizes = ['32K', '64K', '128K', '256K', '512K', '1024K', '2048K', '4096K', '8192K', '16384K', '32768K']
data_onesweep = data_from_files([f"device_radix_sort-0-{size}.json"   for size in sizes], parser=parse_json)
data_merge    = data_from_files([f"device_radix_sort-32M-{size}.json" for size in sizes], parser=parse_json)


# absolute plot (bandwidth)
plot_2d([data_onesweep, data_merge], ["onesweep", "merge"])

# relative plot (% perf increase)
plot_2d([data_onesweep, data_merge], ["onesweep", "merge"], baseline_key="merge", output="chart_rel.png")


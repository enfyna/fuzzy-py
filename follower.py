# https://github.com/saianurag96/Data-and-Artificial-Intelligence/blob/main/Mamdani%20Fuzzy%20Inference%20System.ipynb
# https://www.sciencedirect.com/science/article/pii/S2772662224001140
# https://ieeexplore.ieee.org/document/9261408

import numpy as np
import skfuzzy as fuzz
import math
import pandas as pd
from skfuzzy import control as ctrl
from sys import argv

import matplotlib.pyplot as plt

BISECTOR = 0
CENTROID = 1

TRIMF = 0
GAUSS = 1

if len(argv) == 3:
    SHAPE = int(argv[1])
    DEFUZZIFIER = int(argv[2])
elif len(argv) == 2:
    SHAPE = int(argv[1])
    DEFUZZIFIER = BISECTOR
else:
    SHAPE = TRIMF
    DEFUZZIFIER = BISECTOR

# inputs
anchor_ratio = ctrl.Antecedent(np.arange(10, 31, .1), 'anchor_ratio')
trans_range = ctrl.Antecedent(np.arange(12, 26, .1), 'trans_range(m)')
node_density = ctrl.Antecedent(np.arange(100, 400, .1), 'node_density')
iterations = ctrl.Antecedent(np.arange(14, 101, .1), 'iterations')

# output
ale = ctrl.Consequent(
    np.arange(0.39, 2.57, 0.001), 'ale',
    defuzzify_method='bisector' if DEFUZZIFIER == BISECTOR else 'centroid'
)

if SHAPE == TRIMF:

    anchor_ratio['low'] = fuzz.trimf(
        anchor_ratio.universe, [12.634, 14.635, 16.636])
    anchor_ratio['med'] = fuzz.trimf(
        anchor_ratio.universe, [16.902, 20.000, 23.098])
    anchor_ratio['high'] = fuzz.trimf(
        anchor_ratio.universe, [27.418, 29.571, 30.000])

    trans_range['low'] = fuzz.trimf(
        trans_range.universe, [13.575, 15.304, 17.032])
    trans_range['med'] = fuzz.trimf(
        trans_range.universe, [17.0, 19.692, 21.136])
    trans_range['high'] = fuzz.trimf(
        trans_range.universe, [22.000, 24.000, 25.000])

    node_density['low'] = fuzz.trimf(
        node_density.universe, [100.000, 100.000, 120.000])
    node_density['med'] = fuzz.trimf(
        node_density.universe, [138.200, 228.000, 328.000])

    iterations['low'] = fuzz.trimf(
        iterations.universe, [14.000, 27.709, 44.950])
    iterations['med'] = fuzz.trimf(
        iterations.universe, [42.526, 53.478, 64.431])
    iterations['high'] = fuzz.trimf(
        iterations.universe, [60.025, 81.724, 100.000])

    ale['low'] = fuzz.trimf(ale.universe, [0.424, 0.684, 0.943])
    ale['med'] = fuzz.trimf(ale.universe, [0.825, 1.184, 1.542])
    ale['high'] = fuzz.trimf(ale.universe, [1.489, 2.052, 2.568])

elif SHAPE == GAUSS:

    anchor_ratio['low'] = fuzz.gaussmf(
        anchor_ratio.universe, 14.635, 1.001)
    anchor_ratio['med'] = fuzz.gaussmf(
        anchor_ratio.universe, 20.000, 1.549)
    anchor_ratio['high'] = fuzz.gaussmf(
        anchor_ratio.universe, 29.571, 1.077)

    trans_range['low'] = fuzz.gaussmf(
        trans_range.universe, 15.304, 0.864)
    trans_range['med'] = fuzz.gaussmf(
        trans_range.universe, 19.692, 0.722)
    trans_range['high'] = fuzz.gaussmf(
        trans_range.universe, 24.000, 1.000)

    node_density['low'] = fuzz.gaussmf(node_density.universe, 100.000,  10.000)
    node_density['med'] = fuzz.gaussmf(node_density.universe, 228.000,  44.900)

    iterations['low'] = fuzz.gaussmf(iterations.universe, 27.709,  8.621)
    iterations['med'] = fuzz.gaussmf(iterations.universe, 53.478,  5.476)
    iterations['high'] = fuzz.gaussmf(iterations.universe, 81.724,  10.850)

    ale['low'] = fuzz.gaussmf(ale.universe, 0.684,  0.130)
    ale['med'] = fuzz.gaussmf(ale.universe, 1.184,  0.179)
    ale['high'] = fuzz.gaussmf(ale.universe, 2.052,  0.282)

# Plotting the membership functions
anchor_ratio.view()
trans_range.view()
node_density.view()
iterations.view()
ale.view()
plt.show()

rules = [

    # 4 - Input Rules:
    ctrl.Rule(
        anchor_ratio['high'] & trans_range['low'] &
        node_density['med'] & iterations['low'],
        ale['low']),
    ctrl.Rule(
        anchor_ratio['med'] & trans_range['high'] &
        node_density['low'] & iterations['low'],
        ale['med']),
    ctrl.Rule(
        anchor_ratio['high'] & trans_range['high'] &
        node_density['low'] & iterations['high'],
        ale['low']),
    ctrl.Rule(
        anchor_ratio['low'] & trans_range['low'] &
        node_density['med'] & iterations['low'],
        ale['low']),
    ctrl.Rule(
        anchor_ratio['low'] & trans_range['low'] &
        node_density['med'] & iterations['high'],
        ale['low']),

    # 3 - Input Rules:
    ctrl.Rule(
        anchor_ratio['low'] & trans_range['low'] & node_density['med'],
        ale['low']),
    ctrl.Rule(
        anchor_ratio['high'] & trans_range['low'] & iterations['low'],
        ale['low']),
    ctrl.Rule(
        anchor_ratio['low'] & node_density['med'] & iterations['high'],
        ale['low']),
    ctrl.Rule(
        trans_range['low'] & node_density['med'] & iterations['low'],
        ale['low']),

    # 2 - Input Rules:
    ctrl.Rule(anchor_ratio['low'] & trans_range['low'], ale['low']),
    ctrl.Rule(anchor_ratio['low'] & node_density['med'], ale['low']),
    ctrl.Rule(anchor_ratio['low'] & iterations['high'], ale['low']),
    ctrl.Rule(trans_range['low'] & node_density['med'], ale['low']),
    ctrl.Rule(trans_range['low'] & iterations['low'], ale['low']),
    ctrl.Rule(node_density['low'] & iterations['low'], ale['med']),

    # 1 - Input Rules:
    ctrl.Rule(anchor_ratio['low'], ale['low']),
    ctrl.Rule(anchor_ratio['high'], ale['low']),
    ctrl.Rule(trans_range['low'], ale['low']),
    ctrl.Rule(trans_range['high'], ale['med']),
    ctrl.Rule(node_density['med'], ale['low']),

    # Custom
    ctrl.Rule(node_density['med'], ale['low']),
    ctrl.Rule(node_density['low'], ale['med']),
    ctrl.Rule(node_density['low'], ale['high'] % 0.13),

    ctrl.Rule(node_density['low'] & iterations['low'], ale['high'] % 0.5),

    ctrl.Rule(
        anchor_ratio['med'] & trans_range['med'] &
        node_density['low'] & iterations['low'],
        ale['high']),

    ctrl.Rule(
        anchor_ratio['high'] & trans_range['low'] &
        node_density['low'] & iterations['low'],
        ale['high']),

]

ctrl_inputs = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(ctrl_inputs)

df = pd.read_csv("./data/mcs_ds_edited_iter_shuffled.csv")

input_vars = ['anchor_ratio', 'trans_range(m)', 'node_density', 'iterations']
output_var = 'ale(m)'

mae_acc = 0
rmse_acc = 0

count = len(df)
diff_count = 0
err_count = 0

for i in range(count):
    inps = df[input_vars].values[i]
    expected = df[output_var].values[i]

    for k, v in zip(input_vars, inps):
        sim.input[k] = v

    sim.compute()

    try:
        out = sim.output['ale']
    except Exception as e:
        err_count += 1
        print(f'out err: {e}')
        continue

    diff = expected - out
    mae_acc += abs(diff)
    rmse_acc += diff ** 2

    if diff > 0.1:
        diff_count += 1
        print(
            sim.input,
            'expected', expected, '\n',
            'ale: '+str(out), '\n',
            'diff: '+str(diff), '\n',
            '^^^^^^^^^^^^'
        )


mae = mae_acc / count
rmse = math.sqrt(rmse_acc / count)

print('mae: '+str(mae))
print('rmse: '+str(rmse))
print('data count: '+str(count))
print('diff count: '+str(diff_count))
print('trimf' if SHAPE == TRIMF else 'gauss', '=>',
      'bisector' if DEFUZZIFIER == BISECTOR else 'centroid')

# TRIMF
# bisector
# mae: 0.2240438862745578
# rmse: 0.3193308406159088
# centroid
# mae: 0.23894472389968255
# rmse: 0.3294523327311681
# GAUSS
# bisector
# mae: 0.21690811485562037
# rmse: 0.30548566430163204
# centroid
# mae: 0.22070068849316135
# rmse: 0.2971489827096147

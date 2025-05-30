# https://github.com/saianurag96/Data-and-Artificial-Intelligence/blob/main/Mamdani%20Fuzzy%20Inference%20System.ipynb
from sys import argv
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import math

import skfuzzy as fuzz
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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
anchor_ratio = ctrl.Antecedent(np.arange(10, 31, 1), 'anchor_ratio')
trans_range = ctrl.Antecedent(np.arange(12, 26, 1), 'trans_range(m)')
node_density = ctrl.Antecedent(np.arange(100, 400, 100), 'node_density')
iterations = ctrl.Antecedent(np.arange(14, 101, 1), 'iterations')

# output
ale = ctrl.Consequent(
    np.arange(0.39, 2.57, 0.001), 'ale',
    defuzzify_method='bisector' if DEFUZZIFIER == BISECTOR else 'centroid'
)

if SHAPE == TRIMF:

    anchor_ratio['low'] = fuzz.trimf(anchor_ratio.universe, [10, 14, 18])
    anchor_ratio['med'] = fuzz.trimf(anchor_ratio.universe, [16, 20, 24])
    anchor_ratio['high'] = fuzz.trimf(anchor_ratio.universe, [24, 29, 34])

    trans_range['low'] = fuzz.trimf(trans_range.universe, [12, 15, 18])
    trans_range['med'] = fuzz.trimf(trans_range.universe, [14, 17, 20])
    trans_range['high'] = fuzz.trimf(trans_range.universe, [17, 20, 23])
    trans_range['very high'] = fuzz.trimf(trans_range.universe, [21, 24, 27])

    node_density['low'] = fuzz.trimf(node_density.universe, [0, 100, 200])
    node_density['med'] = fuzz.trimf(node_density.universe, [128, 228, 328])

    iterations['low'] = fuzz.trimf(iterations.universe, [14, 28, 42])
    iterations['med'] = fuzz.trimf(iterations.universe, [39, 53, 67])
    iterations['high'] = fuzz.trimf(iterations.universe, [63, 82, 100])

    ale['low'] = fuzz.trimf(ale.universe, [0.39, 0.68, 0.90])
    ale['med'] = fuzz.trimf(ale.universe, [0.8, 1.18, 1.59])
    ale['high'] = fuzz.trimf(ale.universe, [1.50, 2.05, 2.57])

elif SHAPE == GAUSS:

    anchor_ratio['low'] = fuzz.gaussmf(anchor_ratio.universe, 14, 4)
    anchor_ratio['med'] = fuzz.gaussmf(anchor_ratio.universe, 20, 4)
    anchor_ratio['high'] = fuzz.gaussmf(anchor_ratio.universe, 29, 4)

    trans_range['low'] = fuzz.gaussmf(trans_range.universe, 15, 4)
    trans_range['med'] = fuzz.gaussmf(trans_range.universe, 17, 4)
    trans_range['high'] = fuzz.gaussmf(trans_range.universe, 20, 4)
    trans_range['very high'] = fuzz.gaussmf(trans_range.universe, 24, 4)

    node_density['low'] = fuzz.gaussmf(node_density.universe, 100, 100)
    node_density['med'] = fuzz.gaussmf(node_density.universe, 228, 100)

    iterations['low'] = fuzz.gaussmf(iterations.universe, 28, 10)
    iterations['med'] = fuzz.gaussmf(iterations.universe, 53, 10)
    iterations['high'] = fuzz.gaussmf(iterations.universe, 82, 10)

    ale['low'] = fuzz.gaussmf(ale.universe, 0.68, 1)
    ale['med'] = fuzz.gaussmf(ale.universe, 1.18, 1)
    ale['high'] = fuzz.gaussmf(ale.universe, 2.05, 1)

# Plotting the membership functions
# anchor_ratio.view()
# trans_range.view()
# node_density.view()
# iterations.view()
# ale.view()
# plt.show()

rules = [
    # 4 - Input Rules:
    ctrl.Rule(
        anchor_ratio['high'] & trans_range['low'] &
        node_density['med'] & iterations['low'],
        ale['low']),
    ctrl.Rule(
        anchor_ratio['med'] & trans_range['very high'] &
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
]

ctrl_inputs = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(ctrl_inputs)

df = pd.read_csv("./data/mcs_ds_edited_iter_shuffled.csv")

input_vars = ['anchor_ratio', 'trans_range(m)', 'node_density', 'iterations']
output_var = 'ale(m)'

mae_acc = 0
rmse_acc = 0

count = len(df)

for i in range(count):
    inps = df[input_vars].values[i]
    expected = df[output_var].values[i]

    for k, v in zip(input_vars, inps):
        sim.input[k] = v

    sim.compute()

    try:
        out = sim.output['ale']

        diff = expected - out
        mae_acc += abs(diff)
        rmse_acc += diff ** 2

        print(
            sim.input,
            'expected', expected, '\n',
            'ale: '+str(out), '\n',
            'diff: '+str(diff)
        )
    except Exception as e:
        print(f'{e}')

    print('^^^^^^^^^^^^')


mae = mae_acc / count
rmse = math.sqrt(rmse_acc / count)

print('mae: '+str(mae))
print('rmse: '+str(rmse))
print('data count: '+str(count))
print('trimf' if SHAPE == TRIMF else 'gauss', '=>',
      'bisector' if DEFUZZIFIER == BISECTOR else 'centroid')

# TRIMF
# bisector
# mae: 0.28479770856447784
# rmse: 0.39833658199481875
# centroid
# mae: 0.2758250629299591
# rmse: 0.3839476801982652

# GAUSS
# bisector
# mae: 0.43952069586444203
# rmse: 0.4969918155241743
# centroid
# mae: 0.4845841705476883
# rmse: 0.5407547847382106

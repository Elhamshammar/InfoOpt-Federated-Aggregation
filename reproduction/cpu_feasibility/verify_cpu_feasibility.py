import csv
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parent
def load(p):
    with (root/p).open(newline='') as f: return list(csv.DictReader(f))
a=load('results/mlj_p0_oracle_weights_alpha0.1.csv'); b=load('results_nojs_nonrobust/nojs_nonrobust_alpha0.1.csv')
assert len(a)==len(b)==500
assert [(x['seed'],x['round'],x['selected'],x['n_corrupt']) for x in a]==[(x['seed'],x['round'],x['selected'],x['n_corrupt']) for x in b]
a=[x for x in a if int(x['n_corrupt'])>0]; b=[x for x in b if int(x['n_corrupt'])>0]
assert len(a)==len(b)==458
for rows,key,expected in [(a,'aligned_weight_corrupt',.2697),(a,'aligned_l1_vs_shuffled',.1136),(b,'aligned_mass',.0568),(b,'l1_aligned_vs_shuffled',.5273),(a,'shuffled_weight_corrupt',.3147),(b,'expected_mass',.3166)]:
    v=mean(float(x[key]) for x in rows); assert abs(v-expected)<.00005; print(key,v)
assert all(x['aligned_fallback']=='True' for x in a)
assert all(x['kappa_satisfied']=='True' and int(x['solver_calls'])==1 for x in b)
print('PASS: 458 matched mixed cohorts; original fallback 458/458; revised target satisfied and one solver call 458/458. Revised CSV has no fallback flag.')

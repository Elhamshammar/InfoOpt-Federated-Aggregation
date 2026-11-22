import re, pathlib, numpy as np
from scipy.stats import t as tdist
root=pathlib.Path('/mnt/data/novelty_stats_audit/InfoOpt_Neurocomputing_Submission_Prep/tables')
files={'Stage1':'tableA9_stage1_perseed.tex','Stage3':'tableA12_stage3_perseed.tex','Stage4':'tableA16_stage4_perseed.tex'}
results=[]
for stage,fn in files.items():
    current_ds=None; alpha=None; rows={}
    for line in (root/fn).read_text().splitlines():
        if '&' not in line or not line.strip().endswith('\\\\'):continue
        cells=[c.strip() for c in line.split('&')]
        if cells[0] in ['CIFAR-100','EuroSAT','PathMNIST']:current_ds=cells[0]
        if not current_ds:continue
        if stage=='Stage3':
            if cells[1] in ['0.1','1.0']:alpha=cells[1]
            arm=cells[2]; numeric=cells[-5:]
        else:arm=cells[1];numeric=cells[-5:]
        try: values=np.array([float(x.replace('\\\\','').strip()) for x in numeric])
        except:continue
        rows[(current_ds,alpha if stage=='Stage3' else '',arm)]=values
    for (ds,a,arm),v in rows.items():
        if arm!='InfoOpt-Full':continue
        for ctrl in ['FedAvg','Uniform weights','Shuffled-score InfoOpt-Full',r'FedAvg, $\gamma_{\mathrm{damp}}=0.50$',r'FedAvg, $\gamma_{\mathrm{damp}}=0.75$']:
            k=(ds,a,ctrl)
            if k not in rows:continue
            d=v-rows[k]; mean=d.mean();se=d.std(ddof=1)/np.sqrt(5); p=2*tdist.sf(abs(mean/se),4) if se else (0 if mean else 1)
            ci=(mean-tdist.ppf(.975,4)*se,mean+tdist.ppf(.975,4)*se)
            results.append((stage,ds,a,ctrl,mean,p,*ci))
    print(stage,'seed rows',len(rows),'comparisons',sum(r[0]==stage for r in results))
from collections import defaultdict
families=defaultdict(list)
for r in results:families[r[0]].append(r)
for stage,items in families.items():
    ps=np.array([r[5] for r in items]); order=np.argsort(ps);adj=np.empty(len(ps));maxp=0
    for rank,i in enumerate(order):maxp=max(maxp,(len(ps)-rank)*ps[i]);adj[i]=min(1,maxp)
    print('\n',stage,'Holm family size',len(items),'reject',sum(adj<.05))
    for i,r in enumerate(items):
        if r[3]=='Shuffled-score InfoOpt-Full' or adj[i]<.05:print(r[1],r[2],r[3], 'mean %.3f raw %.4g holm %.4g CI [%.3f,%.3f]'%(r[4],r[5],adj[i],r[6],r[7]))

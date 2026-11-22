from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; D=P/'source_data'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
ds=['cifar100','eurosat','pathmnist']; titles=['CIFAR-100','EuroSAT','PathMNIST']
rows=list(csv.DictReader((D/'primary_summary.csv').open()))
lookup={(r['dataset'],r['row']):r for r in rows}
methods=['fedavg','gamma0.5','gamma0.75','InfoOpt_Full','RA_InfoOpt','fedprox','fednova','fedopt','ibfl','feddp','feddyn','scaffold']
labels=['FedAvg','FedAvg γ=0.50','FedAvg γ=0.75','InfoOpt-Full (AIB)','RA-InfoOpt (AIB)','FedProx','FedNova','FedOpt','IB-FL','FedDP','FedDyn †','SCAFFOLD †']
def save(fig,name,note):
 fig.text(.01,.015,note,ha='left',va='bottom',fontsize=8)
 fig.savefig(P/(name+'.pdf'),bbox_inches='tight');fig.savefig(P/(name+'.png'),bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(12,5.8),sharey=True)
for ax,d,t in zip(axs,ds,titles):
 for i,m in enumerate(methods):
  r=lookup[d,m]
  if int(r['n']):
   color='#b65e22' if m in ['feddyn','scaffold'] else ('#2176ae' if m=='InfoOpt_Full' else '#444444')
   ax.errorbar(float(r['mean']),i,xerr=float(r['sd']),fmt='s' if m in ['feddyn','scaffold'] else 'o',color=color,capsize=3,markersize=4)
  else: ax.text(3,i,'Withheld' if m=='RA_InfoOpt' else 'Excluded: strict E3',color='#777777',va='center',fontsize=8)
 ax.set(title=t,xlabel='Round-100 test accuracy (%)',xlim=(0,100));ax.grid(axis='x',alpha=.2)
axs[0].set_yticks(range(12),labels);axs[0].invert_yaxis()
fig.suptitle('Retrospective reanalysis: terminal accuracy (mean ± sample SD)',y=.98)
fig.subplots_adjust(left=.17,bottom=.17,top=.9,wspace=.12)
save(fig,'figure_accuracy','n=5 for available estimates. † Unresolved implementation provenance; audit observations, not method rankings.\nInfoOpt/RA use AIB; comparators do not. RA combined estimates withheld; CIFAR FedOpt excluded from primary.')
contr=json.loads((D/'paired_contrasts.json').read_text())['contrasts']
fig,ax=plt.subplots(figsize=(8,4.6));yy=[];ll=[]
for i,c in enumerate(contr):
 yy.append(i);ll.append(titles[ds.index(c['dataset'])]+': '+('InfoOpt−FedAvg' if c['estimable'] else 'RA−FedAvg'))
 if c['estimable']:
  m=c['mean_diff_pp'];lo,hi=c['ci95'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#2176ae',capsize=4)
  ax.text(3.2,i,f"n=5; Holm p={c['p_holm']:.3f}",va='center',fontsize=8)
 else:ax.text(-9.8,i,'Not estimable: combined RA strata withheld',va='center',fontsize=8,color='#777777')
ax.axvline(0,color='#888888',lw=1,ls='--');ax.set_yticks(yy,ll);ax.set_ylim(5.5,-.5);ax.set(xlabel='Mean difference in accuracy (percentage points)',xlim=(-10.5,7),title='Retrospective primary contrasts with 95% paired-t intervals');ax.grid(axis='x',alpha=.15)
fig.subplots_adjust(left=.3,bottom=.24)
save(fig,'figure_paired_contrasts','Holm family size six retained; three RA contrasts not estimable. Zero-crossing intervals do not establish equivalence.\nComparisons combine architecture and aggregation changes; pairing assumes identical same-seed partition membership.')
sens=json.loads((D/'sensitivity_latest_repeat.json').read_text())['results']
fig,axs=plt.subplots(1,3,figsize=(10,3.8),sharex=True,sharey=True)
for ax,d,t in zip(axs,ds,titles):
 rr=[x for x in sens if x['dataset']==d and x['row'] in ['fedavg','InfoOpt_Full']]
 for i,r in enumerate(rr):
  v=r['delta'];ax.plot(v,i,'o',color=['#444444','#2176ae'][i]);ax.text(v-.006,i+.15,f"{v:+.3f} pp",ha='right',fontsize=9)
 ax.axvline(0,color='#888888',ls='--',lw=1);ax.set(title=t,xlabel='Latest minus earliest (pp)',xlim=(-.22,.07),ylim=(-.4,1.6));ax.grid(axis='x',alpha=.2)
axs[0].set_yticks([0,1],['FedAvg','InfoOpt-Full (AIB)']);fig.suptitle('Within-stratum repeat sensitivity: change in mean accuracy');fig.subplots_adjust(left=.19,bottom=.27,top=.82,wspace=.15)
save(fig,'figure_repeat_sensitivity','Same-source cells retain their only run. Maximum movement among these six summaries: 0.164 pp.\nThis sensitivity does not assess FedDyn/SCAFFOLD or every Table 3 row; AIB mismatch persists.')
ra=json.loads((D/'ra_strata_summaries.json').read_text())['strata']
fig,axs=plt.subplots(1,3,figsize=(11,4.8))
for ax,d,t in zip(axs,ds,titles):
 blocks=[r for r in ra if r['dataset']==d]
 for i,r in enumerate(blocks):
  vals=[v['round100_acc_pct'] for v in r['per_seed']];ax.scatter(np.arange(len(vals))*.08+i-.04*(len(vals)-1),vals,color='#2176ae',s=23,zorder=3)
  if r['sd'] is not None:ax.errorbar(i,r['mean'],yerr=r['sd'],fmt='_',color='#333333',capsize=5,markersize=14)
  else:ax.plot(i,r['mean'],'_',color='#333333',markersize=14)
 ax.set_xticks(range(len(blocks)),[r['stratum'][:6]+'\nn='+str(r['n_seeds'])+'; seeds\n'+','.join(str(x['seed']) for x in r['per_seed']) for r in blocks],fontsize=8)
 ax.set(title=t,xlabel='Configuration stratum (hash prefix)');ax.grid(axis='y',alpha=.2)
axs[0].set_ylabel('Round-100 accuracy (%)');fig.suptitle('RA-InfoOpt: separate configuration strata');fig.subplots_adjust(bottom=.29,top=.86,wspace=.27)
save(fig,'figure_ra_strata','Dots: one selected run per seed within each stratum; bars: mean ± sample SD (no SD for n=1).\nStrata remain separate; no combined five-seed estimate and no inferential comparison across strata.')

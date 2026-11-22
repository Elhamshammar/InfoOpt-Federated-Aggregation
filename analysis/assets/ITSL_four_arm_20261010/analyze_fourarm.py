from pathlib import Path
import json,csv,hashlib,math,itertools
import numpy as np
from scipy import stats
R=Path('fourarm_results'); O=Path('outputs/fourarm_analysis_20261010');O.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,msg):
 if not ok:raise RuntimeError(msg)
cm=read(R/'COLLECTION_MANIFEST.json');check(len({x['path'] for x in cm['files']})==406,'collection keys')
for f in cm['files']:check(sha(R/f['path'])==f['sha256'] and (R/f['path']).stat().st_size==f['bytes'],f['path'])
M=read(R/'RUN_MANIFEST.json');D=['cifar100','eurosat','pathmnist'];A=['fedavg_aiboff','fedavg_aibon','infoopt_aiboff','infoopt_aibon'];S=list(range(100,105))
check({(c['dataset'],c['seed'],c['arm']) for c in M['cells']}==set(itertools.product(D,S,A)) and len(M['cells'])==60,'60 unique cells')
values={};records=[]; configs={}
commons={'lr':.01,'momentum':.9,'weight_decay':.0005,'local_epochs':2,'batch_size':64,'clip_grad_norm':5.,'rounds':100,'clients_per_round':5}
for c in M['cells']:
 d,s,a=c['dataset'],c['seed'],c['arm'];key=(d,s,a);p=R/'runs'/Path(c['run_dir']).name
 check(c['status']=='complete' and not c['failed_attempts'],str(key)+' status')
 for f,h in c['hashes'].items():
  if f!='final_model.pt':check(sha(p/f)==h,str(key)+' '+f)
 v=read(R/'verification'/Path(c['verification_report']).name)
 check(v['verified'] is True and not v['failures'] and (v['dataset'],v['seed'],v['arm'])==key and v['hashes']==c['hashes'] and v['run_dir']==c['run_dir'],str(key)+' report')
 cfg=read(p/'config_used.json');q=cfg['config'];configs[key]=cfg
 check(q['seed']==s and q['dataset']['name'].lower().replace('-','')==d and cfg['method']==('fedavg' if a.startswith('fedavg') else 'InfoOpt-Full') and q['model']['use_aib']==a.endswith('aibon'),str(key)+' identity')
 check(all(q['training'][k]==v for k,v in commons.items()),str(key)+' training')
 check(q['partition']=={'alpha':.3,'min_samples_per_client':50,'num_clients':10},str(key)+' partition cfg')
 check(q['infoopt']['early_stop']['enabled'] is False and not q.get('counterfactual_probe',{}).get('enabled',False) and cfg.get('gamma') is None,str(key)+' stopping/probe/gamma')
 check(read(p/'summary.json')['early_stop_triggered'] is False,str(key)+' summary')
 part=read(p/'partition_used.json');check(part==read(R/'shared'/f'partition_{d}_seed{s}.json')['client_indices'],str(key)+' partition')
 check({str(k):len(v) for k,v in part.items()}=={str(k):v for k,v in cfg['client_sizes'].items()},str(key)+' counts')
 with (p/'metrics.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
 check([int(x['round']) for x in rows]==list(range(1,101)),str(key)+' rounds')
 cohort=read(R/'shared'/f'cohort_schedule_{d}_seed{s}.json')['schedule']
 for i,x in enumerate(rows):
  check(math.isfinite(float(x['test_acc'])) and 0<=float(x['test_acc'])<=1 and math.isfinite(float(x['test_loss'])) and math.isfinite(float(x['train_loss'])),str(key)+' finite')
  check(x['early_stop_triggered'].lower()=='false' and x['method']==cfg['method'],str(key)+' metrics')
  check([int(z) for z in x['selected_clients'].split(';')]==cohort[i],str(key)+' cohorts')
 val=float(rows[-1]['test_acc']);check(val==v['terminal_acc']==c['terminal_acc'],str(key)+' endpoint')
 values[key]=val;records.append({'dataset':d,'seed':s,'arm':a,'round100_accuracy_fraction':val,'run_dir':c['run_dir'],'metrics_sha256':sha(p/'metrics.csv')})
# Full configs identical across arms except intended switches and output name, across seeds except seed.
for d in D:
 normalized=[]
 for s,a in itertools.product(S,A):
  q=json.loads(json.dumps(configs[d,s,a]['config']));q.pop('seed');q.pop('experiment_name');q['model'].pop('use_aib');normalized.append(q)
 check(all(q==normalized[0] for q in normalized),d+' full config compatibility')
def paired(x):
 x=np.asarray(x)*100;n=len(x);m=float(x.mean());sd=float(x.std(ddof=1));se=sd/math.sqrt(n)
 if sd==0:lo=hi=m;p=1. if m==0 else 0.
 else:
  lo,hi=stats.t.interval(.95,n-1,loc=m,scale=se);p=float(stats.ttest_1samp(x,0).pvalue)
 return {'n':n,'mean_difference_pp':m,'sd_difference_pp':sd,'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),'p_raw':p,'seed_differences_pp':x.tolist()}
def holm(rows):
 ix=sorted(range(len(rows)),key=lambda i:rows[i]['p_raw']);last=0.
 for rank,i in enumerate(ix):last=max(last,min(1.,(len(rows)-rank)*rows[i]['p_raw']));rows[i]['p_holm']=last
primary=[];interactions=[];summary=[]
for d in D:
 for setting in ['aiboff','aibon']:
  row={'dataset':d,'aib':setting,**paired([values[d,s,'infoopt_'+setting]-values[d,s,'fedavg_'+setting] for s in S])};primary.append(row)
 interactions.append({'dataset':d,**paired([(values[d,s,'infoopt_aibon']-values[d,s,'infoopt_aiboff'])-(values[d,s,'fedavg_aibon']-values[d,s,'fedavg_aiboff']) for s in S])})
 for a in A:
  x=np.array([values[d,s,a] for s in S])*100;summary.append({'dataset':d,'arm':a,'n':5,'mean_accuracy_percent':float(x.mean()),'sd_accuracy_pp':float(x.std(ddof=1))})
holm(primary);holm(interactions)
def dump(name,x):(O/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def csvout(name,x):
 with (O/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
csvout('per_seed_round100.csv',records);csvout('primary_summary.csv',summary);dump('primary_contrasts.json',primary);dump('aib_interactions.json',interactions)
dump('verification_report.json',{'archive_sha256':'0a56c0080cb8da42566be1ca4a976937d50f1f273b10eb06d4eb76d4b660f385','collection_files_hash_verified':406,'unique_runs_verified':60,'partitions_and_cohorts_verified':60,'full_common_configs_match':True,'endpoint':'round100 test_acc','limitation':'Checkpoints omitted from collection: server verifier reports and checkpoint hashes inspected, checkpoint load not independently repeated.'})
print(json.dumps({'summary':summary,'primary':primary,'interactions':interactions},indent=2))

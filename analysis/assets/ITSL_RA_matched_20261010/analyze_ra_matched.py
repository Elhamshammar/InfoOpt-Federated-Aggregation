from pathlib import Path
import csv,json,hashlib,math,itertools,shutil,zipfile
import numpy as np
from scipy import stats
import yaml
ROOT=Path(__file__).resolve().parent
raw=ROOT/'ra_results'; exp=raw/'experiment'; prep=raw/'preparation'
out=ROOT/'outputs/ITSL_RA_matched_analysis_20261010';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def write(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
m=load(raw/'COLLECTION_MANIFEST.json')
for r in m['files']:assert sha(raw/r['path'])==r['sha256'],r['path']
for line in (prep/'FROZEN_HASHES.sha256').read_text().splitlines():
 if line.strip():
  h,n=line.split(maxsplit=1);assert sha(prep/n.lstrip('*'))==h,n
manifest=load(exp/'RUN_MANIFEST.json');cells=manifest['cells']
arms=['fedavg_aibon','infoopt_aibon','ra_probeoff','ra_probeon'];ds=['cifar100','eurosat','pathmnist']
assert {(c['dataset'],c['seed'],c['arm']) for c in cells}==set(itertools.product(ds,range(100,105),arms)) and len(cells)==60
values={};perseed=[];checks=[]
for c in cells:
 d,s,a=c['dataset'],c['seed'],c['arm'];rd=exp/'runs'/Path(c['run_dir']).name
 assert c['status']=='complete' and not c['failed_attempts']
 report=load(exp/'verification'/f'{d}_{a}_seed{s}.json')
 assert report['verified'] and not report['failures'] and (report['dataset'],report['seed'],report['arm'])==(d,s,a)
 for n,h in c['hashes'].items():
  if n=='final_model.pt':assert report['hashes'][n]==h;continue
  assert sha(rd/n)==h,n
  if n in report['hashes']:assert report['hashes'][n]==h
 cfgraw=load(rd/'config_used.json');cfg=cfgraw['config'];want=yaml.safe_load((prep/'snapshot/configs'/f'raexp_{d}_{a}.yaml').read_text())
 assert cfgraw['method']=={'fedavg_aibon':'fedavg','infoopt_aibon':'InfoOpt-Full','ra_probeoff':'RA-InfoOpt','ra_probeon':'RA-InfoOpt'}[a]
 assert cfg['seed']==s and cfg['model']['use_aib'] is True and cfg['dataset']['name']=={'cifar100':'CIFAR100','eurosat':'EuroSAT','pathmnist':'PathMNIST'}[d]
 for section in ('training','infoopt','regime_reliability_controller','counterfactual_probe','evaluation','model','partition','validation'):assert cfg.get(section)==want.get(section),section
 assert cfg['infoopt']['early_stop']['enabled'] is False and cfg['counterfactual_probe']['enabled']==(a=='ra_probeon')
 rows=list(csv.DictReader((rd/'metrics.csv').open(encoding='utf-8')))
 assert [int(r['round']) for r in rows]==list(range(1,101))
 for r in rows:
  assert math.isfinite(float(r['test_loss'])) and math.isfinite(float(r['test_acc'])) and 0<=float(r['test_acc'])<=1
  assert r['early_stop_triggered']=='False'
 assert [[int(x) for x in r['selected_clients'].split(';')] for r in rows]==load(exp/'shared'/f'cohort_schedule_{d}_seed{s}.json')['schedule']
 assert load(rd/'partition_used.json')==load(exp/'shared'/f'partition_{d}_seed{s}.json')['client_indices']
 assert load(rd/'summary.json')['early_stop_triggered'] is False
 if a=='ra_probeon':assert [int(r['round']) for r in csv.DictReader((rd/'counterfactual_probe.csv').open())]==list(range(5,101,5))
 else:assert not (rd/'counterfactual_probe.csv').exists()
 acc=float(rows[-1]['test_acc']);assert acc==c['terminal_acc']==report['terminal_acc']
 values[d,s,a]=acc*100;perseed.append(dict(dataset=d,seed=s,arm=a,round=100,test_acc_fraction=acc,test_acc_percent=acc*100,run_name=rd.name))
 checks.append(dict(dataset=d,seed=s,arm=a,verified=True,hashes=c['hashes']))
def contrast(d,a,b):
 diff=np.array([values[d,s,a]-values[d,s,b] for s in range(100,105)])
 mean=float(diff.mean());sd=float(diff.std(ddof=1));se=sd/math.sqrt(5)
 p=float(2*stats.t.sf(abs(mean/se),4)) if sd else (1.0 if mean==0 else 0.0)
 ci=[mean-float(stats.t.ppf(.975,4))*se,mean+float(stats.t.ppf(.975,4))*se]
 return dict(dataset=d,contrast=f'{a} minus {b}',n=5,mean_diff_pp=mean,sd_diff_pp=sd,ci95_pp=ci,raw_p=p,paired_differences_pp=diff.tolist())
def holm(records):
 order=sorted(range(len(records)),key=lambda i:records[i]['raw_p']);v=0
 for k,i in enumerate(order):v=max(v,min(1,(len(records)-k)*records[i]['raw_p']));records[i]['holm_p']=v
 return records
primary=holm([contrast(d,'ra_probeoff',b) for d in ds for b in arms[:2]])
secondary=holm([contrast(d,'ra_probeon','ra_probeoff') for d in ds])
summary=[]
for d,a in itertools.product(ds,arms):
 x=[values[d,s,a] for s in range(100,105)];summary.append(dict(dataset=d,arm=a,n=5,mean_percent=float(np.mean(x)),sd_percent=float(np.std(x,ddof=1))))
for n,r in [('per_seed_round100.csv',perseed),('primary_summary.csv',summary)]:
 with (out/n).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
write('primary_contrasts.json',primary);write('probe_contrasts.json',secondary)
write('verification_report.json',dict(collection_files_verified=537,runs_verified=60,checkpoint_scope='Not supplied locally; checkpoint strict-load, finite-tensor and architecture checks rely on the collected server verifier reports, with hashes cross-checked.',raw_archive_sha256=sha(ROOT/'upload/ITSL_RA_matched_results_20261010_234446.tar.gz'),metadata_issue='RUN_MANIFEST experiment label remains four_arm_20261009: stale orchestration label; cell identity and all paths establish the RA campaign.',cells=checks))
shutil.copyfile(prep/'PROTOCOL.md',out/'PROTOCOL.md');shutil.copyfile(exp/'RUN_MANIFEST.json',out/'source_manifest.json');shutil.copyfile(__file__,out/'analyze_ra_matched.py')
lines=['# Matched RA analysis — 10 October 2026','','All 60 fresh runs verified from collected metadata; all arms AIB enabled. Protocol and analysis plan frozen before launch; not external preregistration. Historical runs and the previous four-arm study are not pooled.','','## Primary family (Holm across six)','','|Dataset|Contrast|Difference pp|95% CI pp|Holm p|','|---|---|---:|---|---:|']
for r in primary:lines.append(f"|{r['dataset']}|{r['contrast']}|{r['mean_diff_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}]|{r['holm_p']:.4f}|")
lines+=['','## Secondary family (Holm across three)','','|Dataset|Probe-on minus probe-off pp|95% CI pp|Holm p|','|---|---:|---|---:|']
for r in secondary:lines.append(f"|{r['dataset']}|{r['mean_diff_pp']:+.3f}|[{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}]|{r['holm_p']:.4f}|")
lines+=['','All estimates have n=5. CIs are individual paired-t 95% intervals, not multiplicity-adjusted simultaneous intervals. Holm decisions use the frozen separate families. Zero-crossing intervals express uncertainty, not equivalence.','','## Evidence and interpretive limits','','537 collection file hashes and the preparation frozen hashes checked. All 60 records have the correct arm/method, AIB, seed, frozen settings, exact unique rounds 1–100, finite losses and bounded accuracies, no early stop, exact shared partition/cohort agreement, and expected probe coverage. No failed attempts are recorded. Checkpoints remain on server: local verification relies on hashed server reports for strict model loading and tensor finiteness. Diagnostic NaNs in non-endpoint optional columns are not treated as endpoint failures.','','The manifest contains a stale experiment label four_arm_20261009; it is preserved and flagged, not used for run selection. RA paths, arms, frozen settings and all 60 cell keys establish campaign identity.','','Probe-on versus off measures instrumentation sensitivity, including unisolated probe-model initialization RNG effects. It does not establish a different aggregation mechanism or a pure probe-evaluation causal effect. Probe subsets overlap training and do not select configurations. All arms are AIB on: no AIB interaction is estimated here. Results are specific to this code, training recipe, alpha and five seeds.','','For writing AI: add this prospective matched study in a separate RA data namespace. Preserve historical tables, prior AIB factorial analysis and all frozen sources. Report all six primary and all three secondary contrasts; do not choose a better RA arm after seeing test results.']
(out/'ANALYSIS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(primary+secondary,indent=2))

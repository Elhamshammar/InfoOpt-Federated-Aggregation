from pathlib import Path
import csv,json,re,hashlib,math,statistics
B=Path('itsl_followup<REDACTED_HOME>/disk1/infoopt_dl_server');R=Path('run_metadata_recovery/runs');O=Path('outputs/feddyn_scaffold_audit_20261010');O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def flat(x,p=''):
 if isinstance(x,dict):
  return {k2:v2 for k,v in x.items() for k2,v2 in flat(v,p+'.'+k if p else k).items()}
 return {p:x}
records=[];inventory=[];diffs=[];summary=[]
for method in ['scaffold','feddyn']:
 for p in sorted(R.glob('*'+method+'*')):
  if not p.is_dir() or not (p/'config_used.json').exists():continue
  c=load(p/'config_used.json');rows=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8')));ends=[r for r in rows if r.get('round')=='100']
  inventory.append({'method':method,'run_name':p.name,'n_rows':len(rows),'terminal_accuracy':float(ends[0]['test_acc']) if len(ends)==1 and ends[0]['test_acc'] else None,'config_sha256':sha(p/'config_used.json'),'metrics_sha256':sha(p/'metrics.csv')})
 for k in range(1,16):
  log=B/'runs/_logs'/f'phase2_{method}_run_{k}.log';t=log.read_text(encoding='utf-8');paths=re.findall(r'Saved results to (\S+)',t);assert len(paths)==1,(method,k,paths)
  p=R/Path(paths[0]).name;c=load(p/'config_used.json');q=c['config'];rows=list(csv.DictReader((p/'metrics.csv').open(encoding='utf-8')))
  assert len(rows)==100 and [int(r['round']) for r in rows]==list(range(1,101))
  assert all(math.isfinite(float(r['test_acc'])) and 0<=float(r['test_acc'])<=1 and math.isfinite(float(r['test_loss'])) for r in rows)
  assert c['method']==method and q['seed'] in range(100,105) and q['model']['use_aib'] is False
  lr=re.findall(r'round=(\d+) method='+method+r' acc=([\d.]+)',t)
  assert len(lr)==100 and all(int(n)==int(row['round']) and abs(float(a)-float(row['test_acc']))<=.000051 for (n,a),row in zip(lr,rows))
  dataset=q['dataset']['name'].lower().replace('-','');s=q['seed'];end=float(rows[-1]['test_acc'])
  evidence={f:sha(p/f) for f in ['metrics.csv','config_used.json','summary.json']}
  record={'dataset':dataset,'method':method,'seed':s,'run_name':p.name,'selection_basis':'exact saved path in repair-campaign log; not accuracy','terminal_accuracy_fraction':end,'log_path':str(log),'log_sha256':sha(log),'evidence_hashes':evidence,'historical_code_hash':'NOT RECOVERED','saved_momentum':q['training'].get('momentum'),'runtime_momentum_current_snapshot':0.0}
  record['early_stop_enabled_saved']=q['infoopt']['early_stop']['enabled']
  record['early_stop_triggered_summary']=load(p/'summary.json')['early_stop_triggered']
  assert record['early_stop_triggered_summary'] is False
  records.append(record)
  # Only document earlier complete runs, never pool with later campaign.
  earlier=[]
  for old in sorted(R.glob(dataset+'*'+method+'_seed'+str(s)+'_*')):
   if old.name>=p.name or not (old/'config_used.json').exists():continue
   rr=list(csv.DictReader((old/'metrics.csv').open(encoding='utf-8')))
   if len(rr)!=100 or [int(r['round']) for r in rr]!=list(range(1,101)):continue
   a=flat(load(old/'config_used.json')['config']);b=flat(q);changes={key:{'earlier':a.get(key),'campaign':b.get(key)} for key in set(a)|set(b) if a.get(key)!=b.get(key)}
   earlier.append({'run_name':old.name,'terminal_accuracy_fraction':float(rr[-1]['test_acc']),'config_differences':changes})
  diffs.append({'dataset':dataset,'method':method,'seed':s,'campaign_run':p.name,'earlier_complete_runs':earlier})
 for dataset in ['cifar100','eurosat','pathmnist']:
  v=[r['terminal_accuracy_fraction']*100 for r in records if r['method']==method and r['dataset']==dataset];assert len(v)==5
  summary.append({'dataset':dataset,'method':method,'n':5,'mean_accuracy_percent':statistics.mean(v),'sd_accuracy_pp':statistics.stdev(v),'status':'campaign-linked descriptive audit only; no source-version certification'})
(O/'campaign_source_map.json').write_text(json.dumps(records,indent=2),encoding='utf-8');(O/'earlier_vs_campaign.json').write_text(json.dumps(diffs,indent=2),encoding='utf-8');(O/'run_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8');(O/'campaign_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Inventory',len(inventory),'campaign mapped',len(records),'earlier complete',sum(len(x['earlier_complete_runs']) for x in diffs));print(json.dumps(summary,indent=2));print('CONFIG differences',[(x['dataset'],x['method'],x['seed'],[r['config_differences'] for r in x['earlier_complete_runs']]) for x in diffs if any(r['config_differences'] for r in x['earlier_complete_runs'])])

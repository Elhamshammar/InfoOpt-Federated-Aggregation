"""CPU-only synthetic checks of recovered source; no datasets or experiment runs."""
from pathlib import Path
import ast, copy, hashlib, json, sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'equation_check_results.json'

def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing overwrite: {OUT}')
    for line in (ROOT / 'SNAPSHOT_HASHES.txt').read_text(encoding='utf-8').splitlines():
        expected, name = line.split('  ', 1)
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Snapshot mismatch: {name}')
    import torch
    sys.path.insert(0, str(ROOT / 'snapshot'))
    from methods.client import train_client, local_momentum
    from utils.feddyn import init_feddyn_state, feddyn_local_penalty, feddyn_server_update
    from utils.aggregation import average_states, weighted_sum_states
    torch.set_num_threads(1)
    checks = []
    def close(name, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
        checks.append(name)
    model = torch.nn.Linear(1, 2, bias=False).double()
    with torch.no_grad(): model.weight.copy_(torch.tensor([[0.2], [-0.1]], dtype=torch.double))
    cfg = {'training': {'lr':0.1,'momentum':0.9,'weight_decay':0.0,'batch_size':2,
                       'local_epochs':2,'num_workers':0,'clip_grad_norm':0.0,'feddyn_alpha':0.4},
           'dataset': {'num_classes':2}}
    x = torch.tensor([[1.0],[1.0]], dtype=torch.double)
    y = torch.tensor([0,0])
    data = torch.utils.data.TensorDataset(x,y)
    c = {'weight':torch.full_like(model.weight,0.3)}
    ci = {'weight':torch.full_like(model.weight,-0.2)}
    # Independent analytical two-class softmax gradients; identical samples remove shuffle effects.
    w = model.weight.detach().clone()
    anchor = w.clone()
    for _ in range(2):
        g = torch.softmax(w[:,0],dim=0).reshape(2,1) - torch.tensor([[1.0],[0.0]],dtype=torch.double)
        w = w - 0.1*(g+c['weight']-ci['weight'])
    r = train_client(0,model,data,[0,1],cfg,'cpu','scaffold',c,ci,compute_info_signals=False)
    close('SCAFFOLD local gradient correction (two steps)',r.state['weight'],w)
    expected_ci = ci['weight']-c['weight']+(anchor-w)/(2*0.1)
    close('SCAFFOLD Option-II uses actual step count',r.scaffold_ci_next['weight'],expected_ci)
    close('SCAFFOLD control delta',r.scaffold_ci_delta['weight'],expected_ci-ci['weight'])
    assert r.local_steps == 2
    close('global model preserved by local training',model.weight.detach(),anchor)
    assert local_momentum(cfg,'scaffold') == local_momentum(cfg,'feddyn') == 0
    assert local_momentum(cfg,'fedavg') == 0.9
    checks.append('effective momentum overrides')

    # Execute the exact recovered server control-update block, not a rewritten imitation.
    tree = ast.parse((ROOT/'snapshot/train.py').read_text(encoding='utf-8'))
    blocks = [n for n in ast.walk(tree) if isinstance(n,ast.If)
              and ast.unparse(n.test)=="method == 'scaffold'"
              and 'scaffold_ci_delta' in ast.unparse(n)]
    assert len(blocks)==1, 'Server block extraction must be unambiguous'
    controls = {i:{'weight':torch.zeros_like(anchor)} for i in range(4)}
    server = {'weight':torch.zeros_like(anchor)}
    r2 = SimpleNamespace(client_id=2,scaffold_ci_next={'weight':torch.full_like(anchor,2.0)},
                         scaffold_ci_delta={'weight':torch.full_like(anchor,2.0)})
    env = {'method':'scaffold','results':[r,r2],'num_clients':4,
           'scaffold_client_cs':controls,'scaffold_server_c':server,'weighted_sum_states':weighted_sum_states}
    code = compile(ast.fix_missing_locations(ast.Module(body=blocks[0].body,type_ignores=[])), '<recovered server block>', 'exec')
    exec(code,env)
    close('SCAFFOLD server participation scaling',server['weight'],(r.scaffold_ci_delta['weight']+r2.scaffold_ci_delta['weight'])/4)
    close('SCAFFOLD unselected client control preserved',controls[1]['weight'],torch.zeros_like(anchor))

    # FedDyn regularizer and derivative against explicit independent equations.
    dual = {'weight':torch.tensor([[0.7],[-0.4]],dtype=torch.double)}
    a = {'weight':anchor+0.2}
    penalty = feddyn_local_penalty(model,dual,a,0.4,'cpu')
    expected = 0.2*((model.weight-a['weight'])**2).sum()+0.4*(model.weight*dual['weight']).sum()
    close('FedDyn local penalty',penalty,expected)
    gradient = torch.autograd.grad(penalty,model.weight)[0]
    close('FedDyn penalty gradient and sign',gradient,0.4*(anchor-a['weight']+dual['weight']))
    w = anchor.clone()
    for _ in range(2):
        g = torch.softmax(w[:,0],dim=0).reshape(2,1)-torch.tensor([[1.0],[0.0]],dtype=torch.double)
        w -= 0.1*(g+0.4*(w-anchor+dual['weight']))
    fr = train_client(0,model,data,[0,1],cfg,'cpu','feddyn',feddyn_y=dual,compute_info_signals=False)
    close('FedDyn local solver two-step recurrence',fr.state['weight'],w)
    ds = init_feddyn_state(model,4,0.4)['duals']
    ds[1]['weight'].fill_(0.8)
    before = copy.deepcopy(ds)
    results = [SimpleNamespace(client_id=i,state={'weight':anchor+d}) for i,d in [(0,0.2),(2,-0.1)]]
    state = feddyn_server_update(ds,results,{'weight':anchor})
    close('FedDyn selected dual increment',ds[0]['weight'],before[0]['weight']+0.2)
    close('FedDyn unselected dual persistence',ds[1]['weight'],before[1]['weight'])
    close('FedDyn all-client bias includes unselected history',state['weight'],anchor+0.05+(0.2-0.1+0.8)/4)
    state2 = feddyn_server_update(ds,[SimpleNamespace(client_id=3,state={'weight':state['weight']+0.3})],state)
    close('FedDyn repeated-round dual persistence',ds[0]['weight'],before[0]['weight']+0.2)
    close('FedDyn second-round bias',state2['weight'],state['weight']+0.3+(0.2-0.1+0.8+0.3)/4)
    close('sample-weighted aggregation adaptation',average_states([{'weight':anchor},{'weight':anchor+1}],[1,3])['weight'],anchor+0.75)
    close('uniform aggregation comparator',average_states([{'weight':anchor},{'weight':anchor+1}])['weight'],anchor+0.5)
    report = {'status':'PASS','checks':checks,'n_checks':len(checks),'device':'cpu','torch_version':torch.__version__,
              'scope':'synthetic recovered implementation only; no historical source certification',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'snapshot_hashes':(ROOT/'SNAPSHOT_HASHES.txt').read_text(encoding='utf-8')}
    with OUT.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(f'PASS: {len(checks)} checks. Report: {OUT}')

if __name__=='__main__':
    main()

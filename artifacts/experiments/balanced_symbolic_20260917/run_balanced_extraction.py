"""Independent balanced I_dark / I_photo / terminal-Q symbolic development run."""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'bkan')]
import run_symbolic_structure_guidance_ablation as basis
import run_unified_symbolic_residual_boost as loader
import run_joint_terminal_charge_pilot as qp
from device_modeling.photodetector.common import concat_without_attrs
from device_modeling.photodetector.symbolic_gated_kan import evaluate_exported_pure_symbolic_formula as old_eval
from device_modeling.terminal_charge.reference import TerminalChargeReference

BUDGETS = [8,12,18,24,30,36,42,54,60,69,84]
ALPHAS = [1e-10,1e-8,1e-6,1e-4]
WEIGHTS = [0.,.05,.25]
QWEIGHTS = [3.,10.,30.]
SLACK = 1.05

def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))

def elimination_path(A,y,alpha,budgets,mandatory=()):
    """Exact compensated deletion cost for ridge: beta_i^2/(H^-1)_ii.

    Columns are scaled once using fitting rows only. Each deletion is followed
    by an exact coefficient refit. Test and validation never order deletions.
    """
    scales=np.sqrt(np.mean(A*A,axis=0))
    scales=np.maximum(scales,1e-14)
    X=A/scales
    G=X.T@X/len(X)+alpha*np.eye(X.shape[1])
    cross=X.T@y/len(X)
    keep=list(range(X.shape[1]))
    snapshots={}
    while len(keep)>=min(budgets):
        H=G[np.ix_(keep,keep)]
        inv=np.linalg.inv(H)
        beta=np.linalg.solve(H,cross[keep])
        if len(keep) in budgets:
            coef=np.zeros(X.shape[1]); coef[keep]=beta/scales[keep]
            snapshots[len(keep)]=coef
        if len(keep)==min(budgets): break
        costs=beta*beta/np.maximum(np.diag(inv),1e-30)
        for j,k in enumerate(keep):
            if k in mandatory: costs[j]=np.inf
        del keep[int(np.argmin(costs))]
    return snapshots

def physical_metrics(task,y,p,derivative_y=None,derivative_p=None,qfloor=None,cfloor=None):
    if task!='Q_terminal':
        rel=np.abs(np.power(10.,p-y)-1.)
        return {'rmse':rmse(y,p),'relative_p95':float(np.quantile(rel,.95))}
    qm=qp.qfit.regression_metrics(y,p,qfloor)
    cm=qp.qfit.regression_metrics(derivative_y,derivative_p,cfloor)
    return {'rmse':qm['rmse'],'relative_p95':qm['active_p95_relative_error'],
            'derivative_rmse':cm['rmse'],'derivative_p95':cm['p95_relative_error']}

def charge_sparse_eval(payload,frame):
    """Sparse, term-wise replay; independent of the fitting design matrix."""
    m=payload['charge_basis']; z=(frame[m['parameter_names']].to_numpy()-m['parameter_mean'])/m['parameter_scale']
    v=frame.bias_v.to_numpy(); q=np.zeros(len(frame)); c=q.copy()
    powers=m['parameter_powers']; npar=len(powers)
    for term in payload['terms']:
        vi,pi=divmod(term['index'],npar)
        p=np.prod(z**np.array(powers[pi]),axis=1)
        if vi<3:
            b=v**(vi+1); d=(vi+1)*v**vi
        else:
            k=m['voltage_knots_V'][vi-3]; a=np.maximum(v-k,0.)
            b=a**3-max(-k,0.)**3; d=3*a*a
        q+=term['coefficient']*p*b; c+=term['coefficient']*p*d
    return q*m['q_scale_C'],c*m['q_scale_C']

def current_sparse_eval(payload,frame):
    return basis.evaluate_formula(payload['current_formula'],frame[payload['input_order']].to_numpy())

def baseline_path(task,seed):
    if task=='I_dark':
        return ROOT/f'artifacts/results/joint_other_branches_20260917/dark/seed_{seed}/I_dark/tcad_direct/budget_36/dark_current_symbolic_gated_kan/pure_symbolic_formula.json'
    bucket='mechanism_symbolic_pareto_photo_joint54_seed42' if seed==42 else 'mechanism_symbolic_pareto_photo_joint54_remaining'
    return ROOT/f'artifacts/results/{bucket}/seed_{seed}/I_photo/joint_bkan_tcad/budget_54/photo_current_symbolic_gated_kan/pure_symbolic_formula.json'

def prepare_current(task,seed,out):
    args,frame,inputs,spec,bspec,parts=loader.base_audit._load_task(loader.loader_args(),task,seed)
    train,val,cal,test=parts
    # Match the current mechanism baseline: 65% fitting + 15% calibration.
    fit=concat_without_attrs([train,cal])
    frames={'train':fit,'validation':val,'test':test}
    ys={k:basis.single._target(f,bspec) for k,f in frames.items()}
    lib=basis.build_library(fit,inputs,task,6)
    matrices={k:np.column_stack([np.ones(len(f)),basis.candidate_matrix(f[list(inputs)].to_numpy(),lib)]) for k,f in frames.items()}
    teacher,info=basis.single._load_bkan_parameter_mean(task,train,inputs,spec,args)
    tp=teacher(fit)
    oldpath=baseline_path(task,seed); old=json.loads(oldpath.read_text(encoding='utf-8'))
    oldpred={k:old_eval(old,f[list(inputs)].to_numpy()) for k,f in frames.items()}
    labels={k:basis.single.shared.task_group_labels(f,spec).astype(str).to_numpy() for k,f in frames.items()}
    assert all(not set(labels[a])&set(labels[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
    manifest=pd.concat([pd.DataFrame({'group':labels[k],'split':k}) for k in labels]).drop_duplicates()
    manifest.to_csv(out/'split_manifest.csv',index=False)
    dense=fit.sample(n=20000,replace=True,random_state=seed)[list(inputs)].copy().reset_index(drop=True)
    rng=np.random.default_rng(seed+987)
    for name in inputs: dense[name]=rng.uniform(fit[name].min(),fit[name].max(),len(dense))
    dense[inputs[0]]=rng.uniform(fit[inputs[0]].min(),0.,len(dense))
    return dict(task=task,seed=seed,frames=frames,ys=ys,X=matrices,teacher=tp,library=lib,
        inputs=list(inputs),baseline=oldpred,baseline_payload=old,baseline_count=old['selected_term_count'],
        sources=[oldpath,Path(info['checkpoint']),args.data],dense=dense,mandatory=(0,))

def prepare_charge(seed,out):
    src=ROOT/'artifacts/results/terminal_charge_model'
    qpath=src/'terminal_charge_predictions.csv'; cpath=src/'terminal_charge_derivative_predictions.csv'
    q,c=qp.assign_splits(pd.read_csv(qpath),pd.read_csv(cpath),seed)
    frames={k:q[q.split.eq(v)].copy() for k,v in [('train','train'),('validation','val'),('test','test')]}
    cs={k:c[c.split.eq(v)].copy() for k,v in [('train','train'),('validation','val'),('test','test')]}
    model=qp.qfit.fit_model(frames['train'],cs['train'],.001,10.,parameter_degree=1)
    ys={k:f[qp.qfit.Q_COL].to_numpy() for k,f in frames.items()}
    dy={k:f[qp.qfit.C_COL].to_numpy() for k,f in cs.items()}
    X={k:model._designs(f)[0] for k,f in frames.items()}
    D={k:model._designs(f)[1]*model.q_scale_C/model.c_scale_F for k,f in cs.items()}
    checkpoint=ROOT/f'artifacts/results/joint_other_branches_20260917/charge/seed_{seed}/teacher_results/model_checkpoint.pt'
    teacher=qp.BayesKANDeviceModeler.load_model(str(checkpoint),device='cpu')
    inputs=['bias_v',*qp.qfit.PARAM_COLS]
    observed=frames['train'][inputs].to_numpy().mean(axis=0)
    if not np.allclose(observed,teacher.scaler_x.mean_,rtol=1e-6,atol=1e-14):
        raise RuntimeError('Q teacher/split mismatch')
    tp=qp.teacher_predict(teacher,frames['train'],inputs)
    anchors=q.drop_duplicates('production_source_row')
    dense=pd.DataFrame(np.repeat(anchors[inputs].to_numpy(),121,axis=0),columns=inputs)
    dense['bias_v']=np.tile(np.linspace(-3,0,121),len(anchors))
    # Also check unsampled combinations within the training rectangular envelope.
    rng=np.random.default_rng(seed+987); random=pd.DataFrame({name:rng.uniform(frames['train'][name].min(),frames['train'][name].max(),20000) for name in inputs})
    dense=pd.concat([dense,random],ignore_index=True)
    q[['production_source_row','bias_v','split']].to_csv(out/'split_manifest.csv',index=False)
    old={k:model.predict(f) for k,f in frames.items()}; oldd={k:model.predict_derivative(f) for k,f in cs.items()}
    return dict(task='Q_terminal',seed=seed,frames=frames,ys=ys,X=X,D=D,dy=dy,cframes=cs,
        teacher=tp,model=model,inputs=inputs,baseline=old,baseline_derivative=oldd,baseline_count=84,
        baseline_payload={'exported_model':model.to_dict()},sources=[qpath,cpath,checkpoint],dense=dense,
        qfloor=model.q_scale_C*.01,cfloor=model.c_scale_F*.01,mandatory=())

def metrics_for(d,split,p,dp=None):
    return physical_metrics(d['task'],d['ys'][split],p,d.get('dy',{}).get(split),dp,d.get('qfloor'),d.get('cfloor'))

def make_payload(d,coef,config):
    common={'schema':'balanced_output_sparse_v1','task':d['task'],'seed':d['seed'],'input_order':d['inputs'],
        'coefficient_count':int(np.count_nonzero(coef)),'config':config,'selection_data':'validation_only',
        'test_used_for_selection':False,'uses_kan_at_inference':False}
    if d['task']=='Q_terminal':
        common['charge_basis']={k:v for k,v in d['model'].to_dict().items() if k!='coefficients'}
        common['terms']=[{'index':int(i),'coefficient':float(coef[i])} for i in np.flatnonzero(coef)]
    else:
        ids=np.flatnonzero(coef[1:]); lib=d['library']
        common['current_formula']={**{k:lib[k] for k in ['input_order','input_scalers','family']},
            'intercept':float(coef[0]),'selected_terms':[{**lib['features'][i],'coefficient':float(coef[i+1])} for i in ids]}
        common['output_space']='log10(I/A)'
    return common

def numerical_check(d,payload,coef):
    dense=d['dense']; isq=d['task']=='Q_terminal'; axis=d['inputs'][0]
    hi=dense.copy(); lo=dense.copy(); h=1e-5
    hi[axis]+=h;lo[axis]-=h
    if isq:
        p,dp=charge_sparse_eval(payload,dense)
        native=d['model']._designs(dense)[0]@coef*d['model'].q_scale_C
        fd=(charge_sparse_eval(payload,hi)[0]-charge_sparse_eval(payload,lo)[0])/(2*h)
        zero=dense.iloc[:1000].copy();zero[axis]=0.
        zeroerr=float(np.max(np.abs(charge_sparse_eval(payload,zero)[0])))
        deriverr=float(np.max(np.abs(dp-fd)/np.maximum(np.abs(dp),1e-30)))
        feasible=bool(np.isfinite(p).all() and np.isfinite(dp).all() and np.min(dp)>0 and zeroerr<1e-26 and deriverr<1e-3)
        return {'physical_pass':feasible,'dense_points':len(dense),'dense_min_dqdv_F':float(np.min(dp)),
            'zero_reference_max_abs_C':zeroerr,'derivative_fd_max_relative':deriverr,'replay_max_abs':float(np.max(np.abs(p-native)))}
    p=current_sparse_eval(payload,dense)
    dp=(current_sparse_eval(payload,hi)-current_sparse_eval(payload,lo))/(2*h)
    native=np.column_stack([np.ones(len(dense)),basis.candidate_matrix(dense[d['inputs']].to_numpy(),d['library'])])@coef
    physical=np.power(10.,p)
    return {'physical_pass':bool(np.isfinite(p).all() and np.isfinite(dp).all() and np.isfinite(physical).all() and np.all(physical>0)),
        'dense_points':len(dense),'positive_bias_slope_fraction':float(np.mean(dp>1e-7)),
        'replay_max_abs':float(np.max(np.abs(native-p)))}

def run_one(task,seed,out):
    dest=out/task/f'seed_{seed}';dest.mkdir(parents=True)
    d=prepare_charge(seed,dest) if task=='Q_terminal' else prepare_current(task,seed,dest)
    isq=task=='Q_terminal'; X=d['X']; valbase=metrics_for(d,'validation',d['baseline']['validation'],d.get('baseline_derivative',{}).get('validation'))
    base=d['model'].q_scale_C if isq else max(np.std(d['ys']['train']),1e-12)
    offset=0. if isq else float(np.mean(d['ys']['train']))
    y=(d['ys']['train']-offset)/base; teacher=(d['teacher']-offset)/base
    records=[]; models={}; candidates=[]
    for w in WEIGHTS:
        for dw in (QWEIGHTS if isq else [0.]):
            blocks=[X['train']];targets=[y]
            if w:blocks.append(np.sqrt(w)*X['train']);targets.append(np.sqrt(w)*teacher)
            if isq:blocks.append(np.sqrt(dw)*d['D']['train']);targets.append(np.sqrt(dw)*d['dy']['train']/d['model'].c_scale_F)
            A=np.vstack(blocks); target=np.concatenate(targets)
            for alpha in ALPHAS:
                for budget,coef in elimination_path(A,target,alpha,[b for b in BUDGETS if b<=A.shape[1]],d['mandatory']).items():
                    coef=coef.copy()
                    if not isq:coef*=base;coef[0]+=offset
                    vp=X['validation']@coef*(base if isq else 1.)
                    vdp=d['D']['validation']@coef*d['model'].c_scale_F if isq else None
                    vm=metrics_for(d,'validation',vp,vdp)
                    ratio=max(vm[k]/max(valbase[k],1e-30) for k in vm)
                    config={'teacher_weight':w,'derivative_weight':dw,'ridge_alpha':alpha,'budget':budget}
                    rid=len(records)
                    row={'candidate_id':rid,**config,**{'validation_'+k:v for k,v in vm.items()},'worst_validation_ratio':ratio,
                        'eligible':bool(ratio<=SLACK and (not isq or (vm['relative_p95']<=.05 and vm['derivative_p95']<=.05)))}
                    records.append(row);models[rid]=coef
    table=pd.DataFrame(records); table.to_csv(dest/'validation_candidates.csv',index=False)
    # The smallest eligible model wins; minimax ratios resolve equal-budget ties.
    ranked=table[table.eligible & (table.budget<=d['baseline_count'])].sort_values(['budget','worst_validation_ratio','teacher_weight','ridge_alpha'])
    chosen=None; checks=[]
    for r in ranked.to_dict('records'):
        coef=models[r['candidate_id']];payload=make_payload(d,coef,{k:r[k] for k in ['teacher_weight','derivative_weight','ridge_alpha','budget']})
        check=numerical_check(d,payload,coef);checks.append({'candidate_id':r['candidate_id'],**check})
        if check['physical_pass']:
            chosen=(r,coef,payload,check);break
    selection={'protocol':'min_count_subject_to_each_validation_error_ratio_le_1.05','baseline_validation':valbase,
        'candidate_count':len(table),'test_used_for_selection':False,'physical_checks':checks,
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in d['sources']}}
    if chosen is None:
        payload={'schema':'unchanged_baseline','task':task,'seed':seed,'coefficient_count':d['baseline_count'],'baseline_payload':d['baseline_payload']}
        count=d['baseline_count'];valmetric=valbase;selected_config={};audit={};kind='fallback'
    else:
        r,coef,payload,audit=chosen;count=payload['coefficient_count'];valmetric={k:r['validation_'+k] for k in valbase};selected_config=payload['config'];kind='sparse_refit'
        selection['selected_candidate_id']=r['candidate_id']
    selection['selected_kind']=kind;payload['selection']=selection
    # Freeze selection and formula before accessing test targets for evaluation.
    dump(dest/'selected_formula.json',payload)
    frozen=json.loads((dest/'selected_formula.json').read_text(encoding='utf-8'))
    if chosen:
        if isq: pred,_=charge_sparse_eval(frozen,d['frames']['test']);_,deriv=charge_sparse_eval(frozen,d['cframes']['test'])
        else:pred=current_sparse_eval(frozen,d['frames']['test']);deriv=None
    else:pred=d['baseline']['test'];deriv=d.get('baseline_derivative',{}).get('test')
    final=metrics_for(d,'test',pred,deriv)
    original=metrics_for(d,'test',d['baseline']['test'],d.get('baseline_derivative',{}).get('test'))
    result={'task':task,'seed':seed,'selected_kind':kind,'baseline_count':d['baseline_count'],'selected_count':count,
        **selected_config,**audit,**{'baseline_test_'+k:v for k,v in original.items()},**{'selected_test_'+k:v for k,v in final.items()},
        'formula':str((dest/'selected_formula.json').relative_to(ROOT))}
    dump(dest/'test_metrics.json',result)
    predictions=d['frames']['test'][d['inputs']].copy();predictions['target']=d['ys']['test'];predictions['baseline']=d['baseline']['test'];predictions['selected']=pred
    predictions.to_csv(dest/'test_predictions.csv',index=False)
    if isq:
        pd.DataFrame({'target':d['dy']['test'],'baseline':d['baseline_derivative']['test'],'selected':deriv}).to_csv(dest/'derivative_test_predictions.csv',index=False)
    print(json.dumps(result),flush=True)
    return result

def main():
    global WEIGHTS
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--tasks',nargs='+',default=['I_dark','I_photo','Q_terminal'],choices=['I_dark','I_photo','Q_terminal']);p.add_argument('--teacher-weights',nargs='+',type=float,default=WEIGHTS);a=p.parse_args()
    WEIGHTS=a.teacher_weights
    if not WEIGHTS or min(WEIGHTS)<0 or 0. not in WEIGHTS: p.error('Teacher weights must be nonnegative and include zero')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    protocol={'script_sha256':sha(Path(__file__)), 'status':'running','seeds':a.seeds,'tasks':a.tasks,'budgets':BUDGETS,'alphas':ALPHAS,'teacher_weights':WEIGHTS,'q_derivative_weights':QWEIGHTS,
        'validation_slack':SLACK,'q_and_dqdv_p95_limit':.05,'selection':'minimum retained coefficient count, then minimum worst validation error ratio; baseline fallback',
        'current_split':'80/10/10 grouped, same as mechanism baselines','charge_split':'112/24/24 conditions; derivative 10/3/3 conditions',
        'scope':'development; historical test results already visible; no confirmatory claim or simulator validation',
        'pruning':'exact ridge loss increase after compensating all remaining coefficients; fitting rows only'}
    dump(out/'protocol.json',protocol)
    rows=[]
    with threadpool_limits(limits=1):
        for task in a.tasks:
            for seed in a.seeds:
                rows.append(run_one(task,seed,out));pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    protocol['status']='complete';dump(out/'protocol.json',protocol)

if __name__=='__main__':main()

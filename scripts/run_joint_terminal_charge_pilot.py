"""Pilot joint BKAN/TCAD terminal-charge compression without changing deployment."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bkan"))
from device_modeling.photodetector.bayesian_modeler import BayesKANDeviceModeler
from device_modeling.photodetector.common import set_seed
from device_modeling.terminal_charge import train as qfit
from device_modeling.terminal_charge.reference import TerminalChargeReference


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_predict(modeler, frame, inputs):
    def forward(raw):
        x = modeler.scaler_x.transform(raw[inputs].to_numpy(np.float32)).astype(np.float32)
        with torch.no_grad():
            pred, _ = modeler.model(torch.from_numpy(x), sample=False)
        return modeler.scaler_y.inverse_transform(pred[:, :1].numpy()).reshape(-1).astype(float) * 1e-15
    zero = frame.copy()
    zero["bias_v"] = 0.0
    return forward(frame) - forward(zero)


def assign_splits(q, c, seed):
    q = q.copy(); c = c.copy()
    conditions = sorted(q.production_source_row.unique().tolist())
    if seed != 42:
        split = qfit.make_condition_split(conditions, c.production_source_row.unique().tolist(), seed)
        q["split"] = q.production_source_row.map(split)
        c["split"] = c.production_source_row.map(split)
    target_holdout = int(round(.15 * len(conditions)))
    counts = q.drop_duplicates("production_source_row").groupby("split").size().to_dict()
    expected = {"train":len(conditions)-2*target_holdout,"val":target_holdout,"test":target_holdout}
    if counts != expected or q["split"].isna().any() or c["split"].isna().any():
        raise RuntimeError(f"Condition-level split mismatch: {counts}, expected {expected}")
    if q.groupby("production_source_row")["split"].nunique().max() != 1:
        raise RuntimeError("A Q condition was split across partitions")
    condition_split = q.drop_duplicates("production_source_row").set_index("production_source_row")["split"]
    if not c["split"].eq(c.production_source_row.map(condition_split)).all():
        raise RuntimeError("Derivative evidence and Q have inconsistent partitions")
    return q, c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/results/joint_other_branches_20260917/charge")
    p.add_argument("--seeds", nargs="+", type=int, default=[42,43,44])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--budgets", nargs="+", type=int, default=[42,60,84])
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error("epochs must be positive and seeds must be unique")
    if len(set(args.budgets)) != len(args.budgets) or any(b < 18 or b % 6 for b in args.budgets):
        p.error("budgets must be unique multiples of six, at least 18")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/results/terminal_charge_model"
    q_path = source / "terminal_charge_predictions.csv"
    c_path = source / "terminal_charge_derivative_predictions.csv"
    q_all = pd.read_csv(q_path)
    c_all = pd.read_csv(c_path)
    inputs = ["bias_v", *qfit.PARAM_COLS]
    protected = [ROOT / "paper/main_manuscript.tex", ROOT / "paper/supplementary_information.tex",
                 ROOT / "bkan/device_modeling/veriloga/ge_si_photodetector_terminal_charge.va"]
    before = {str(x):sha(x) for x in protected}
    protocol = {
        "status":"running", "seeds":args.seeds, "teacher_epochs":args.epochs,
        "parameter_degree":1, "coefficient_budgets":args.budgets, "alpha":0.001,
        "derivative_weight":10.0, "joint_teacher_weight":0.25,
        "teacher":"new BKAN parameter-mean pilot trained on Q only; f(V,p)-f(0,p) reference projection",
        "teacher_architecture":[6,8,2], "grid":8, "spline_order":3, "kl_weight":0.1,
        "input_hashes":{str(x.relative_to(ROOT)):sha(x) for x in [q_path,c_path]},
        "scope":"grouped development splits; fixed settings; not a confirmatory superiority or simulator test",
        "physical_checks":"Q(0)=0; dQ/dV>0 on dense in-envelope condition curves; independent JSON replay",
        "selection":"all fixed candidates reported; no test-driven tuning or deployment promotion",
        "reweighted_control":"1.25 times Q data loss plus unchanged derivative/ridge loss; isolates extra value-loss weighting",
        "protected_hashes_before":before,
    }
    dump(args.output / "protocol.json",protocol)
    records=[]
    for seed in args.seeds:
        q,c=assign_splits(q_all,c_all,seed)
        q["charge_fC"]=q[qfit.Q_COL]*1e15
        train=q[q.split.eq("train")].copy(); val=q[q.split.eq("val")].copy(); test=q[q.split.eq("test")].copy()
        ct=c[c.split.eq("train")].copy(); cv=c[c.split.eq("val")].copy(); ce=c[c.split.eq("test")].copy()
        group_sets=[set(f.production_source_row) for f in (train,val,test)]
        assert all(not (group_sets[i]&group_sets[j]) for i in range(3) for j in range(i))
        seed_dir=args.output/f"seed_{seed}"; seed_dir.mkdir(exist_ok=True)
        q[["production_source_row","bias_v","split"]].to_csv(seed_dir/"split_manifest.csv",index=False)
        set_seed(seed)
        teacher=BayesKANDeviceModeler("terminal_Q",str(seed_dir/"teacher_results"),device="cpu")
        teacher.load_data(train,inputs,"charge_fC",use_log_transform=False)
        teacher.build_model({"width":[6,8,2],"grid":8,"k":3,"kl_weight":0.1,
            "num_mc_samples":20,"likelihood":"gaussian","seed":seed,"prediction_seed":1729})
        teacher.train({"batch_size":256,"lr":0.001,"weight_decay":1e-5,
            "num_epochs":args.epochs,"val_freq":5,"train_mc_samples":3,
            "validation_mc_samples":20,"early_stopping_patience":None,"validation_seed":1729},df_val=val)
        teacher.model.eval(); teacher.save_model()
        teacher_train=teacher_predict(teacher,train,inputs)
        teacher_metrics={}
        for label,f in (("train",train),("val",val),("test",test)):
            pred=teacher_predict(teacher,f,inputs)
            teacher_metrics[label+"_rmse_C"]=float(np.sqrt(np.mean((pred-f[qfit.Q_COL].to_numpy())**2)))
        dump(seed_dir/"teacher_metrics.json",teacher_metrics)
        qfloor=max(np.quantile(np.abs(train[qfit.Q_COL]),.95)*.01,1e-30)
        cfloor=max(np.quantile(np.abs(ct[qfit.C_COL]),.95)*.01,1e-30)
        anchors=q.drop_duplicates("production_source_row").copy()
        dense=pd.DataFrame(np.repeat(anchors[inputs].to_numpy(),121,axis=0),columns=inputs)
        dense["bias_v"]=np.tile(np.linspace(-3,0,121),len(anchors))
        zero=anchors.copy(); zero["bias_v"]=0.
        for budget in args.budgets:
            knots=np.linspace(-3.,0.,budget//6-1)[1:-1]
            assert (3+len(knots))*6==budget
            base=qfit.fit_model(train,ct,.001,10.,parameter_degree=1,voltage_knots=knots)
            for mode,weight in (("tcad_direct",0.),("tcad_reweighted",.25),("joint_bkan_tcad",.25)):
                import copy
                model=copy.deepcopy(base)
                if weight:
                    qd,_=model._designs(train); _,cd=model._designs(ct)
                    design=np.vstack([qd,np.sqrt(weight)*qd,np.sqrt(10.)*cd*model.q_scale_C/model.c_scale_F,
                                      np.sqrt(.001)*np.eye(budget)])
                    target=np.concatenate([train[qfit.Q_COL].to_numpy()/model.q_scale_C,
                                           np.sqrt(weight)*(teacher_train if mode=="joint_bkan_tcad" else train[qfit.Q_COL].to_numpy())/model.q_scale_C,
                                           np.sqrt(10.)*ct[qfit.C_COL].to_numpy()/model.c_scale_F,np.zeros(budget)])
                    model.coefficients=np.linalg.lstsq(design,target,rcond=None)[0]
                formula=seed_dir/mode/f"budget_{budget}.json"
                dump(formula,{"exported_model":model.to_dict(),"source":mode,"seed":seed,"teacher_weight":weight})
                reference=TerminalChargeReference(formula)
                replay,replay_derivative=reference.evaluate(dense)
                native=model.predict(dense); derivative=model.predict_derivative(dense)
                replay_error=float(np.max(np.abs(replay-native)))
                if not np.all(np.isfinite(native)) or replay_error>1e-26:
                    raise RuntimeError("Charge serialization replay failed")
                derivative_replay_error=float(np.max(np.abs(replay_derivative-derivative)))
                if derivative_replay_error>1e-26:
                    raise RuntimeError("Derivative serialization replay failed")
                row={"independent_derivative_replay_max_abs_F":derivative_replay_error,"seed":seed,"source":mode,"budget":budget,"formula":str(formula.relative_to(ROOT)),
                     "teacher_test_rmse_C":teacher_metrics["test_rmse_C"],"dense_points":len(dense),
                     "dense_min_dqdv_F":float(derivative.min()),"dense_finite":bool(np.isfinite(derivative).all()),
                     "zero_reference_max_abs_C":float(np.max(np.abs(model.predict(zero)))),
                     "independent_replay_max_abs_C":replay_error}
                for label,qpart,cpart in (("validation",val,cv),("test",test,ce)):
                    qm=qfit.regression_metrics(qpart[qfit.Q_COL],model.predict(qpart),qfloor)
                    cm=qfit.regression_metrics(cpart[qfit.C_COL],model.predict_derivative(cpart),cfloor)
                    row.update({f"{label}_q_rmse_C":qm["rmse"],f"{label}_q_r2":qm["r2"],
                        f"{label}_q_active_p95":qm["active_p95_relative_error"],
                        f"{label}_dqdv_p95":cm["p95_relative_error"],f"{label}_dqdv_rmse_F":cm["rmse"]})
                records.append(row)
                pred_test=test[inputs+[qfit.Q_COL,"production_source_row"]].copy()
                pred_test["prediction_C"]=model.predict(test)
                pred_test.to_csv(formula.with_suffix('.predictions.csv'),index=False)
                pd.DataFrame(records).to_csv(args.output/"runs.csv",index=False)
                print(json.dumps(row),flush=True)
    after={str(x):sha(x) for x in protected}
    if before!=after:raise RuntimeError("Deployment or paper changed during pilot")
    protocol.update(status="complete",protected_files_unchanged=True)
    dump(args.output/"protocol.json",protocol)


if __name__=="__main__":
    main()

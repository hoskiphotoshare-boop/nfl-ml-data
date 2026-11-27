#!/usr/bin/env python3
"""Game-level component models."""
from __future__ import annotations
import argparse, json, os, sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import joblib, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from weather_features import WEATHER_FEATURES, attach_weather, forecast_weather
TARGETS = ["c_pass_att","c_rush_att","c_pass_yds","c_rush_yds","c_sacks","c_off_td","c_fg_att_short","c_fg_att_40p"]
TEST_SEASONS = (2024,2025,2026)
CLAMP_ENVELOPES = {"c_pass_att":(-0.08,0.02),"c_rush_att":(-0.02,0.10),"c_pass_yds":(-0.10,0.02),"c_rush_yds":(-0.02,0.10),"c_fg_att_40p":(-0.40,0.00),"c_sacks":(-0.20,0.20),"c_off_td":(-0.20,0.20),"c_fg_att_short":(-0.20,0.20)}
K_GRID=[round(i*0.05,2) for i in range(21)]

def _completed_game_ids():
    sched=None
    for path in ("data/schedules.parquet","data/schedules.csv"):
        if os.path.exists(path):
            sched=pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
            break
    if sched is None: return set()
    played=sched.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    return set(played["game_id"].to_list())

def build_component_frame():
    pbp=pl.read_parquet("data/pbp_latest.parquet")
    completed=_completed_game_ids()
    pbp=pbp.filter((pl.col("season")>=2022)&(pl.col("season_type")=="REG")&pl.col("game_id").is_in(list(completed)))
    vol=pbp.filter(pl.col("play_type").is_in(["run","pass"])&((pl.col("rush_attempt")==1)|(pl.col("pass_attempt")==1)))
    if "two_point_attempt" in vol.columns:
        vol=vol.filter(pl.col("two_point_attempt").fill_null(0)!=1)
    vol_agg=vol.group_by("game_id").agg(pl.col("pass_attempt").sum().alias("c_pass_att"),pl.col("rush_attempt").sum().alias("c_rush_att"),pl.col("passing_yards").fill_null(0).sum().alias("c_pass_yds"),pl.col("rushing_yards").fill_null(0).sum().alias("c_rush_yds"),pl.col("sack").sum().alias("c_sacks"))
    td_agg=vol.filter(pl.col("touchdown")==1).group_by("game_id").agg(pl.len().alias("c_off_td"))
    vol_agg=vol_agg.join(td_agg,on="game_id",how="left").with_columns(pl.col("c_off_td").fill_null(0))
    fg=pbp.filter(pl.col("field_goal_attempt")==1)
    fg=fg.with_columns(pl.when(pl.col("kick_distance")<40).then(pl.lit("short")).otherwise(pl.lit("long")).alias("_bucket"))
    fg_agg=fg.group_by("game_id").agg((pl.col("_bucket")=="short").sum().alias("c_fg_att_short"),(pl.col("_bucket")=="long").sum().alias("c_fg_att_40p"))
    frame=vol_agg.join(fg_agg,on="game_id",how="left")
    for col in ("c_fg_att_short","c_fg_att_40p"):
        if col in frame.columns: frame=frame.with_columns(pl.col(col).fill_null(0))
        else: frame=frame.with_columns(pl.lit(0).alias(col))
    if os.path.exists("features/match_features.parquet"):
        mf=pl.read_parquet("features/match_features.parquet")
        frame=frame.join(mf,on="game_id",how="left")
    frame=attach_weather(frame,source="actual")
    sched=None
    for path in ("data/schedules.parquet","data/schedules.csv"):
        if os.path.exists(path):
            sched=pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
            break
    if sched is not None:
        sp=sched.select("game_id","home_score","away_score").filter(pl.col("home_score").is_not_null()&pl.col("away_score").is_not_null()).with_columns((pl.col("home_score")+pl.col("away_score")).alias("total_points"))
        frame=frame.join(sp.select("game_id","total_points"),on="game_id",how="left")
    for t in TARGETS:
        if t not in frame.columns: frame=frame.with_columns(pl.lit(0.0).alias(t))
    if "total_points" in frame.columns: frame=frame.filter(pl.col("total_points").is_not_null())
    return frame

def build_kick_frame():
    pbp=pl.read_parquet("data/pbp_latest.parquet")
    completed=_completed_game_ids()
    fg=pbp.filter((pl.col("season")>=2022)&(pl.col("season_type")=="REG")&pl.col("game_id").is_in(list(completed))&(pl.col("field_goal_attempt")==1))
    fg=fg.with_columns((pl.col("field_goal_result")=="made").cast(pl.Int8).alias("made"))
    game_ids=fg.select("game_id").unique()
    gw=attach_weather(game_ids,source="actual")
    fg=fg.join(gw,on="game_id",how="left")
    return fg

def _base_features(frame):
    try:
        bundle=joblib.load("models/score_models.joblib")
        base=bundle.get("features",[])
    except Exception:
        base=[]
    feats=[c for c in base if c in frame.columns]
    for c in WEATHER_FEATURES:
        if c in frame.columns and c not in feats: feats.append(c)
    return feats

def _walk_forward_mae(frame,features,target):
    seasons=frame["season"].to_numpy()
    y=frame[target].to_numpy().astype(float)
    X=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in features]).to_numpy()
    per_season={}; oof=[]
    for test_season in TEST_SEASONS:
        train=seasons<test_season; test=seasons==test_season
        if train.sum()<100 or test.sum()<20: continue
        model=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=42)
        model.fit(X[train],y[train])
        pred=model.predict(X[test])
        per_season[str(test_season)]=round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(y[test],pred)),3)
        oof.extend((pred-y[test]).tolist())
    sigma=float(np.std(oof)) if oof else 1.0
    sigma=max(sigma,1e-3)
    return per_season,sigma

def _calibrate_k(frame,features,target):
    seasons=frame["season"].to_numpy()
    y=frame[target].to_numpy().astype(float)
    X=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in features]).to_numpy()
    Xneutral=X.copy()
    idx={c:i for i,c in enumerate(features)}
    for c in features:
        if c.startswith("wx_"):
            if c=="wx_temp": Xneutral[:,idx[c]]=60.0
            else: Xneutral[:,idx[c]]=0.0
    best_k=1.0; best_mae=1e18; k_results={}
    for k in K_GRID:
        maes=[]
        for test_season in TEST_SEASONS:
            train=seasons<test_season; test=seasons==test_season
            if train.sum()<100 or test.sum()<20: continue
            model=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=42)
            model.fit(X[train],y[train])
            pa=model.predict(X[test]); pn=model.predict(Xneutral[test])
            delta=pa-pn
            pred=pn+k*delta
            mae=float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(y[test],pred))
            maes.append(mae)
        pooled=float(np.mean(maes)) if maes else 1e18
        k_results[str(k)]=round(pooled,4)
        if pooled<best_mae: best_mae=pooled; best_k=k
    return best_k,k_results

def train_fg_classifier():
    kf=build_kick_frame()
    kf=kf.with_columns((pl.col("kick_distance")**2).alias("kick_distance_sq"))
    fg_features=["kick_distance","kick_distance_sq","wx_temp","wx_wind","wx_gust","wx_snow","wx_cold_32","wx_wind_15","wx_gust_30","wx_outdoor"]
    fg_features=[c for c in fg_features if c in kf.columns]
    _mono_map={'kick_distance':-1,'kick_distance_sq':-1,'wx_temp':1,'wx_wind':-1,'wx_gust':-1,
               'wx_snow':-1,'wx_cold_32':-1,'wx_wind_15':-1,'wx_gust_30':-1,'wx_outdoor':0}
    monotonic_cst=[_mono_map.get(c,0) for c in fg_features]
    for c in fg_features:
        if c.startswith("wx_"): kf=kf.with_columns(pl.col(c).fill_null(0))
    if "kick_distance_sq" not in kf.columns:
        kf=kf.with_columns((pl.col("kick_distance")**2).alias("kick_distance_sq"))
    X=kf.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in fg_features]).to_numpy()
    y=kf["made"].to_numpy().astype(int)
    seasons=kf["season"].to_numpy()
    per_season={}
    for test_season in TEST_SEASONS:
        train=seasons<test_season; test=seasons==test_season
        if train.sum()<50 or test.sum()<10: continue
        clf=HistGradientBoostingClassifier(max_iter=200,random_state=42,monotonic_cst=monotonic_cst)
        clf.fit(X[train],y[train])
        prob=clf.predict_proba(X[test])[:,1]
        try: auc=float(roc_auc_score(y[test],prob))
        except Exception: auc=float("nan")
        per_season[str(test_season)]={"auc":round(auc,4),"n":int(test.sum())}
    clf=HistGradientBoostingClassifier(max_iter=200,random_state=42,monotonic_cst=monotonic_cst)
    clf.fit(X,y)
    in_prob=clf.predict_proba(X)[:,1]
    buckets={}
    for lo,hi in ((20,30),(30,40),(40,50),(50,70)):
        for temp_label,temp_mask in (("warm",kf["wx_temp"].to_numpy()>=45),("cold",kf["wx_temp"].to_numpy()<45)):
            mask=(kf["kick_distance"].to_numpy()>=lo)&(kf["kick_distance"].to_numpy()<hi)&temp_mask
            if mask.sum()==0: continue
            buckets[f"{lo}-{hi-1}_{temp_label}"]={"n":int(mask.sum()),"actual":round(float(y[mask].mean()),4),"predicted":round(float(in_prob[mask].mean()),4)}
    print("FG calibration:")
    for k,v in buckets.items(): print(f"  {k}: n={v['n']} actual={v['actual']} predicted={v['predicted']}")
    joblib.dump({"model":clf,"features":fg_features,"metrics":per_season,"calibration":buckets},"models/fg_model.joblib")
    print("Saved models/fg_model.joblib")
    return {"model":clf,"features":fg_features,"calibration":buckets}

def train():
    frame=build_component_frame()
    features=_base_features(frame)
    if not features:
        print("No features; skipping."); return
    if "season" not in frame.columns:
        print("No season column; skipping."); return
    models={}; sigmas={}; metrics={}; k_weather={}; k_results_all={}
    for target in TARGETS:
        per_season,sigma=_walk_forward_mae(frame,features,target)
        metrics[target]=per_season
        sigmas[target]=round(sigma,4)
        X=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in features]).to_numpy()
        y=frame[target].to_numpy().astype(float)
        final=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=42).fit(X,y)
        models[target]=final
        k,kres=_calibrate_k(frame,features,target)
        k_weather[target]=k
        k_results_all[target]=kres
        print(f"{target}: sigma={sigma:.3f} per-season MAE={per_season} k_weather={k}")
    # bridge
    bridge_features=TARGETS
    tp=frame["total_points"].to_numpy().astype(float) if "total_points" in frame.columns else None
    bridge_metrics={}; best_name="linear"
    if tp is not None:
        Xb=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in bridge_features]).to_numpy()
        seasons=frame["season"].to_numpy()
        candidates={}
        for name,ctor in (("linear",LinearRegression),("ridge",Ridge),("hgb",lambda: HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=42))):
            per={}
            for test_season in TEST_SEASONS:
                train=seasons<test_season; test=seasons==test_season
                if train.sum()<100 or test.sum()<20: continue
                m=ctor(); m.fit(Xb[train],tp[train])
                pred=m.predict(Xb[test])
                mae=float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(tp[test],pred))
                per[str(test_season)]=round(mae,3)
            bridge_metrics[name]=per
            pooled=float(np.mean(list(per.values()))) if per else 1e18
            candidates[name]=pooled
        best_name=min(candidates,key=lambda k:candidates[k])
        Xb_all=Xb
        if best_name=="linear": bridge_model=LinearRegression().fit(Xb_all,tp)
        elif best_name=="ridge": bridge_model=Ridge().fit(Xb_all,tp)
        else: bridge_model=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=42).fit(Xb_all,tp)
        print(f"Scoring bridge selected: {best_name} per-season MAE={bridge_metrics.get(best_name)}")
        joblib.dump({"model":bridge_model,"features":bridge_features,"name":best_name,"metrics":bridge_metrics},"models/scoring_bridge.joblib")
        print("Saved models/scoring_bridge.joblib")
    fg_info=train_fg_classifier()
    # gate
    try:
        score_bundle=joblib.load("models/score_models.joblib")
        direct_features=score_bundle.get("total_features",score_bundle.get("features"))
    except Exception:
        direct_features=features
    direct_features=[c for c in direct_features if c in frame.columns]
    direct_maes={}; recon_maes={}; worst_weather_maes={}
    direct_biases={}; recon_biases={}; recon_pred_by_season={}; ytot_test_by_season={}
    direct_pred_by_season={}; mask_w_by_season={}
    for test_season in TEST_SEASONS:
        train=frame["season"].to_numpy()<test_season
        test=frame["season"].to_numpy()==test_season
        if train.sum()<100 or test.sum()<20: continue
        Xtr=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in features]).to_numpy()
        Xte=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in features]).to_numpy()
        ytot=frame["total_points"].to_numpy().astype(float)
        if direct_features:
            Xd_tr=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in direct_features]).to_numpy()
            Xd_te=Xd_tr
            direct_model=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=42).fit(Xd_tr[train],ytot[train])
            direct_pred=direct_model.predict(Xd_te[test])
            direct_maes[str(test_season)]=round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(ytot[test],direct_pred)),3)
            direct_biases[str(test_season)]=round(float(np.mean(direct_pred-ytot[test])),3)
        comp_pred={}
        for target in TARGETS:
            cm=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=42).fit(Xtr[train],frame[target].to_numpy().astype(float)[train])
            actual_p=cm.predict(Xte[test])
            Xte_n=Xte[test].copy()
            idx={c:i for i,c in enumerate(features)}
            for c in features:
                if c.startswith("wx_"):
                    if c=="wx_temp": Xte_n[:,idx[c]]=60.0
                    else: Xte_n[:,idx[c]]=0.0
            neutral_p=cm.predict(Xte_n)
            k=k_weather.get(target,1.0)
            delta=actual_p-neutral_p
            lo,hi=CLAMP_ENVELOPES.get(target,(-1,1))
            denom=np.where(np.abs(neutral_p)>1e-9,neutral_p,1.0)
            rel=delta/denom
            rel_clamped=np.clip(rel,lo,hi)
            delta_clamped=rel_clamped*denom
            comp_pred[target]=neutral_p+k*delta_clamped
        Xb_tr=frame.select([pl.col(c).cast(pl.Float64,strict=False).fill_null(0.0) for c in bridge_features]).to_numpy()[train]
        Xb_te=np.column_stack([comp_pred[t] for t in bridge_features])
        tp_tr=ytot[train]
        if best_name=="linear": bm=LinearRegression().fit(Xb_tr,tp_tr)
        elif best_name=="ridge": bm=Ridge().fit(Xb_tr,tp_tr)
        else: bm=HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05,max_depth=3,random_state=42).fit(Xb_tr,tp_tr)
        recon_pred=bm.predict(Xb_te)
        # neutral reconstruction
        comp_neutral={}
        for target in TARGETS:
            cm=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=3,random_state=42).fit(Xtr[train],frame[target].to_numpy().astype(float)[train])
            Xte_n=Xte[test].copy()
            idx={c:i for i,c in enumerate(features)}
            for c in features:
                if c.startswith("wx_"):
                    if c=="wx_temp": Xte_n[:,idx[c]]=60.0
                    else: Xte_n[:,idx[c]]=0.0
            comp_neutral[target]=cm.predict(Xte_n)
        Xb_neutral=np.column_stack([comp_neutral[t] for t in bridge_features])
        recon_neutral=bm.predict(Xb_neutral)
        delta_total=recon_pred-recon_neutral
        scale=np.ones_like(delta_total)
        mask=np.abs(delta_total)>5.0
        if mask.any(): scale[mask]=5.0/np.abs(delta_total[mask])
        recon_pred=recon_neutral+delta_total*scale
        recon_maes[str(test_season)]=round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(ytot[test],recon_pred)),3)
        recon_biases[str(test_season)]=round(float(np.mean(recon_pred-ytot[test])),3)
        recon_pred_by_season[str(test_season)]=recon_pred.copy()
        ytot_test_by_season[str(test_season)]=ytot[test].copy()
        direct_pred_by_season[str(test_season)]=direct_pred.copy()
        # worst-weather decile
        if "wx_wind" in frame.columns:
            wind=frame["wx_wind"].to_numpy()[test]
            cold=frame["wx_temp"].to_numpy()[test]
            severity=wind+(60-cold)*0.5
            thr=np.quantile(severity,0.9)
            mask_w=severity>=thr
            mask_w_by_season[str(test_season)]=mask_w.copy()
            if mask_w.sum()>5:
                worst_weather_maes[str(test_season)]={"n":int(mask_w.sum()),"direct_mae":round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(ytot[test][mask_w],direct_pred[mask_w])),3),"reconstructed_mae":round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(ytot[test][mask_w],recon_pred[mask_w])),3),"direct_bias":round(float(np.mean(direct_pred[mask_w]-ytot[test][mask_w])),3),"reconstructed_bias":round(float(np.mean(recon_pred[mask_w]-ytot[test][mask_w])),3)}
    direct_pooled=float(np.mean(list(direct_maes.values()))) if direct_maes else 1e9
    recon_pooled=float(np.mean(list(recon_maes.values()))) if recon_maes else 1e9
    direct_bias_pooled=float(np.mean(list(direct_biases.values()))) if direct_biases else 0.0
    recon_bias_pooled=float(np.mean(list(recon_biases.values()))) if recon_biases else 0.0

    # Out-of-fold bias correction for reconstruction.
    correction_by_season={}
    corrected_recon_maes={}
    corrected_recon_biases={}
    corrected_ww={}
    for ts in recon_pred_by_season:
        prior=[r for s in recon_pred_by_season if int(s)<int(ts) for r in (recon_pred_by_season[s]-ytot_test_by_season[s]).tolist()]
        corr=float(np.mean(prior)) if prior else 0.0
        correction_by_season[ts]=corr
        rp=recon_pred_by_season[ts]-corr
        yt=ytot_test_by_season[ts]
        corrected_recon_maes[ts]=round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(yt,rp)),3)
        corrected_recon_biases[ts]=round(float(np.mean(rp-yt)),3)
        if ts in mask_w_by_season and mask_w_by_season[ts] is not None:
            mw=mask_w_by_season[ts]
            corrected_ww[ts]={"n":int(mw.sum()),"reconstructed_mae":round(float(__import__("sklearn.metrics",fromlist=["mean_absolute_error"]).mean_absolute_error(yt[mw],rp[mw])),3),"reconstructed_bias":round(float(np.mean(rp[mw]-yt[mw])),3)}
    corrected_pooled=float(np.mean(list(corrected_recon_maes.values()))) if corrected_recon_maes else recon_pooled
    corrected_bias_pooled=float(np.mean(list(corrected_recon_biases.values()))) if corrected_recon_biases else recon_bias_pooled

    recon_biases_before = dict(recon_biases)
    recon_maes_before = dict(recon_maes)
    # condition
    apply_corr = bool(abs(recon_bias_pooled)>1.5 or abs(recon_bias_pooled) > abs(direct_bias_pooled)+1.0)
    if apply_corr:
        recon_pooled=corrected_pooled
        recon_bias_pooled=corrected_bias_pooled
        recon_maes=corrected_recon_maes
        recon_biases=corrected_recon_biases
        for ts in worst_weather_maes:
            if ts in corrected_ww:
                worst_weather_maes[ts]["reconstructed_mae"]=corrected_ww[ts]["reconstructed_mae"]
                worst_weather_maes[ts]["reconstructed_bias"]=corrected_ww[ts]["reconstructed_bias"]
    gate_passed=bool(recon_pooled<=direct_pooled*1.02)

    # worst-weather biases
    for ts in worst_weather_maes:
        # add direct_bias already there; ensure reconstructed_bias present
        if "reconstructed_bias" not in worst_weather_maes[ts]:
            worst_weather_maes[ts]["reconstructed_bias"]=None

    gate={"passed":gate_passed,
          "direct_mae_pooled":round(direct_pooled,4),
          "reconstructed_mae_pooled":round(recon_pooled,4),
          "direct_mae_per_season":direct_maes,
          "reconstructed_mae_per_season":recon_maes,
          "bias":{
             "direct_pooled":round(direct_bias_pooled,4),
             "reconstructed_pooled_before":round(float(np.mean(list(recon_biases_before.values()))) if recon_biases_before else 0.0,4),
             "reconstructed_pooled":round(recon_bias_pooled,4),
             "direct_per_season":direct_biases,
             "reconstructed_per_season_before":{k:round(float(v),4) for k,v in recon_biases_before.items()},
             "reconstructed_per_season":{k:round(float(v),4) for k,v in recon_biases.items()},
          },
          "worst_weather_decile":worst_weather_maes,
          "threshold":"reconstructed <= direct * 1.02",
          "total_delta_cap":5.0,
          "bias_correction":{
             "applied":apply_corr,
             "value":round(float(np.mean([v for v in correction_by_season.values()])) if correction_by_season else 0.0,4),
             "correction_by_season":{k:round(v,4) for k,v in correction_by_season.items()},
             "before_mae_pooled":round(float(np.mean(list(recon_maes_before.values()))) if recon_maes_before else 0.0,4),
             "after_mae_pooled":round(recon_pooled,4),
          }}
    # fix bias before value properly
    if correction_by_season:
        all_resid=[r for ts in recon_pred_by_season for r in (recon_pred_by_season[ts]-ytot_test_by_season[ts]).tolist()]
        gate["bias_correction"]["value"]=round(float(np.mean(all_resid)),4)
    # before mae = original recon pooled
    print(f"Gate: direct={direct_pooled:.3f} (bias {direct_bias_pooled:+.3f}) reconstructed={recon_pooled:.3f} (bias {recon_bias_pooled:+.3f}) passed={gate_passed} bias_correction={apply_corr}")

    bc_val = float(np.mean([v for v in correction_by_season.values()])) if (apply_corr and correction_by_season) else 0.0
    joblib.dump({"features":features,"targets":TARGETS,"models":models,"sigmas":sigmas,"metrics":metrics,"k_weather":k_weather,"k_results":k_results_all,"clamp_envelopes":CLAMP_ENVELOPES,"gate":gate,"fg_features":fg_info["features"],"bias_correction_applied":apply_corr,"bias_correction_value":bc_val},"models/component_models.joblib")
    print("Saved models/component_models.joblib")
    print("k_weather:",k_weather)

def predict_components_from_stats(home_stats,away_stats,week,season,weather=None,neutral=False):
    bundle=joblib.load("models/component_models.joblib")
    features=bundle["features"]
    models=bundle["models"]
    k_weather=bundle.get("k_weather",{})
    envelopes=bundle.get("clamp_envelopes",CLAMP_ENVELOPES)
    fg_bundle=None
    try: fg_bundle=joblib.load("models/fg_model.joblib")
    except Exception: fg_bundle=None
    import predict_upcoming as pu
    gw=forecast_weather(roof="dome") if neutral else (weather or {})
    home_s=dict(home_stats or {}); away_s=dict(away_stats or {})
    for k,v in gw.items():
        home_s[k]=v; away_s[k]=v
    x=pu.build_feature_vector(features,home_s,away_s,int(week or 0),int(season or 0),weather=gw)
    gn=forecast_weather(roof="dome")
    home_n=dict(home_stats or {}); away_n=dict(away_stats or {})
    for k,v in gn.items():
        home_n[k]=v; away_n[k]=v
    xn=pu.build_feature_vector(features,home_n,away_n,int(week or 0),int(season or 0),weather=gn)
    comps_actual={t:float(models[t].predict([x])[0]) for t in bundle["targets"]}
    comps_neutral={t:float(models[t].predict([xn])[0]) for t in bundle["targets"]}
    final={}; clamped=False
    for t in bundle["targets"]:
        k=k_weather.get(t,1.0)
        delta=comps_actual[t]-comps_neutral[t]
        denom=comps_neutral[t] if abs(comps_neutral[t])>1e-9 else 1.0
        rel=delta/denom
        lo,hi=envelopes.get(t,(-1,1))
        rel_c=min(max(rel,lo),hi)
        if abs(rel_c-rel)>0.02:
            clamped=True
        delta_c=rel_c*denom
        final[t]=comps_neutral[t]+k*delta_c
    fg_make_prob_40p=0.75; fg_make_prob_short=0.90
    if fg_bundle is not None:
        try:
            fg_features=fg_bundle["features"]
            def fg_prob_at(dist):
                frow=dict(home_s); frow.update(away_s)
                frow["kick_distance"]=dist; frow["kick_distance_sq"]=dist*dist
                for c in fg_features:
                    if c.startswith("wx_") and c not in frow:
                        frow[c]=gw.get(c,0.0 if c!="wx_temp" else 60.0)
                fx=[float(frow.get(c,0.0)) for c in fg_features]
                return float(fg_bundle["model"].predict_proba([fx])[0,1])
            fg_make_prob_40p=fg_prob_at(44.0)
            fg_make_prob_short=fg_prob_at(33.0)
        except Exception: pass
    fg_points=3.0*(final.get("c_fg_att_short",0.0)*fg_make_prob_short+final.get("c_fg_att_40p",0.0)*fg_make_prob_40p)
    out=dict(final)
    out["fg_make_prob_40p"]=fg_make_prob_40p
    out["fg_make_prob_short"]=fg_make_prob_short
    out["fg_points"]=fg_points
    out["weather_delta_clamped"]=clamped
    out["_neutral_components"]=comps_neutral
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true")
    args=parser.parse_args()
    if args.train: train()
    else: print("Use --train to train component models.")

if __name__=="__main__":
    try: main()
    except Exception as exc:
        print(f"Component model error: {exc}"); raise

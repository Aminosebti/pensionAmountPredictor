import os, re, glob, json, joblib, math
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor   # pip install xgboost

# ───────────────────────────────
# 1.  Utilitaires précédents
# ───────────────────────────────
def strip_ns(tag): return tag.split("}", 1)[1] if "}" in tag else tag
def find_first(elem, local): return next((e for e in elem.iter() if strip_ns(e.tag) == local), None)

def sexe_depuis_niss(niss: str) -> int:
    n = re.sub(r"\D", "", niss)
    return 0 if int(n[6:9]) % 2 else 1

RANK_LEVELS = OrderedDict([
    (15,[r"général\b",r"\bamiral\b",r"\bgeneraal\b",r"\badmiraal\b"]),
    (14,[r"lieutenant[- ]?général",r"vice[- ]?amiral",r"luitenant[- ]?generaal"]),
    (13,[r"général[- ]?major",r"major[- ]?général",r"divisieadmiraal",r"generaal[- ]?majoor"]),
    (12,[r"\bcolonel\b",r"kapitein[- ]?ter[- ]?zee",r"capitaine de vaisseau"]),
    (11,[r"lieutenant[- ]?colonel",r"fregatkapitein",r"capitaine de fr[eé]gate"]),
    (10,[r"\bmajor\b",r"korvetkapitein",r"capitaine de corvette"]),
    (9, [r"capitaine[- ]?commandant",r"luitenant[- ]?ter[- ]?zee.*1",r"lieutenant de vaisseau 1"]),
    (8, [r"\bcapitaine\b",r"\bkapitein\b",r"luitenant[- ]?ter[- ]?zee",r"enseigne de vaisseau 1"]),
    (7, [r"\blieutenant\b",r"sous[- ]?lieutenant",r"onderluitenant",r"enseigne de vaisseau 2"]),
    (6, [r"adjudant[- ]?major",r"oppermeester[- ]?chef",r"premier sergent[- ]?major"]),
    (5, [r"adjudant[- ]?chef",r"oppermeester\b",r"premier sergent[- ]?chef"]),
    (4, [r"\badjudant\b",r"\bmeester\b",r"sergent[- ]?chef"]),
    (3, [r"premier sergent",r"eerste sergeant",r"premier caporal",r"eerste korporaal"]),
    (2, [r"\bsergent\b",r"\bkorporaal\b",r"\bcaporal\b",r"matroos",r"\bsold(at)?\b"]),
    (1, [r"\baspirant\b",r"\bcadet\b",r"\bvolontaire\b"]),
])
def grade_level(txt:str) -> int|None:
    t = txt.lower()
    for lvl, pats in RANK_LEVELS.items():
        if any(re.search(p, t) for p in pats): return lvl
    return None

# ───────────────────────────────
# 2.  XML → dict de features
# ───────────────────────────────
def xml_to_features(path:str) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()

    person = find_first(root, "personIdentification")
    niss_el  = find_first(person, "niss")
    birth_el = find_first(person, "birthDate")
    niss      = niss_el.text if niss_el is not None else None
    birth     = birth_el.text if birth_el is not None else None
    gender    = sexe_depuis_niss(niss) if niss else None

    career = find_first(root, "publicDetailedCareer")
    periods = [e for e in career.iter() if strip_ns(e.tag)=="publicDetailedPeriod"]

    starts, ends, rank_vals, supp_cnt = [], [], [], 0
    for p in periods:
        ps, pe = find_first(p,"periodStart"), find_first(p,"periodEnd")
        if ps is not None and pe is not None:
            try:
                starts.append(datetime.fromisoformat(ps.text))
                ends.append  (datetime.fromisoformat(pe.text))
            except: pass
        # supplements
        if any(find_first(s,"salarySupplements") is not None
               for s in p.iter() if strip_ns(s.tag)=="publicSalary"):
            supp_cnt += 1
        # rank
        fd = find_first(p,"functionDescription")
        if fd is not None and fd.text:
            lvl = grade_level(fd.text)
            if lvl: rank_vals.append(lvl)

    career_years = (max(ends)-min(starts)).days/365.25 if starts and ends else None
    suppl_ratio  = supp_cnt/len(periods) if periods else None
    highest_rank = max(rank_vals) if rank_vals else None

    nominal = find_first(root,"nominalAmount")
    target  = float(nominal.text) if nominal is not None else None

    return dict(
        birthDate=birth,
        gender=gender,
        careerLengthYears=career_years,
        supplementRatio=suppl_ratio,
        highestRank=highest_rank,
        nominalAmount=target,
        file=os.path.basename(path)
    )

def folder_to_df(folder:str)->pd.DataFrame:
    rows=[]
    for p in glob.glob(os.path.join(folder,"*.xml")):
        try: rows.append(xml_to_features(p))
        except Exception as e: print("❌",os.path.basename(p),e)
    return pd.DataFrame(rows)

# ───────────────────────────────
# 3.  Entraînement / comparaison
# ───────────────────────────────
MODELS = {
    "Linear":   LinearRegression(),
    "Ridge":    Ridge(),
    "RF":       RandomForestRegressor(n_estimators=200,random_state=42),
    "GB":       GradientBoostingRegressor(),
    "XGB":      XGBRegressor(n_estimators=300,random_state=42,verbosity=0),
    "KNN":      KNeighborsRegressor(n_neighbors=5)
}

def eval_model(m, X_test, y_test):
    pred = m.predict(X_test)
    return dict(
        r2   = r2_score(y_test, pred),
        rmse = math.sqrt(mean_squared_error(y_test, pred)),
        mae  = mean_absolute_error(y_test, pred)
    )

def train_and_select(df, feature_cols, target="nominalAmount"):
    df = df.dropna(subset=feature_cols+[target])      # lignes complètes
    X, y = df[feature_cols], df[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)

    best_name,best_m,best_r2 = None,None,float("-inf")
    metrics = {}
    for name, model in MODELS.items():
        model.fit(X_tr, y_tr)
        m = eval_model(model,X_te,y_te)
        metrics[name]=m
        if m["r2"]>best_r2:
            best_name,best_m,best_r2 = name,model,m["r2"]

    return best_name,best_m,metrics,feature_cols

def save_model(model,features,dir="saved_model"):
    os.makedirs(dir,exist_ok=True)
    joblib.dump(model,os.path.join(dir,"model.pkl"))
    with open(os.path.join(dir,"features.json"),"w") as f:
        json.dump(features,f)

# ───────────────────────────────
# 4.  Pipeline complet
# ───────────────────────────────
if __name__=="__main__":
    FOLDER = "/home/aminesebti/Jiras/ai/globalCareerCollecting/run_5-6-2025/extractedIdentificationPublicCareerAndNominalAmount"          # <<< à adapter
    df = folder_to_df(FOLDER)

    feature_cols = ["careerLengthYears","supplementRatio","highestRank","gender"]
    best_name, best_model, results, used_feats = train_and_select(df, feature_cols)

    print("\n─────────  Résultats  ─────────")
    for n,m in results.items():
        print(f"{n:14s} | R²={m['r2']:.3f}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}")
    print("────────────────────────────────")

    print(f"\n✅ Meilleur modèle : {best_name} (R² {results[best_name]['r2']:.3f})")
    save_model(best_model, used_feats)
    print("💾 Modèle sauvegardé dans ./saved_model/")

"""Faithful re-run of Masters_Constrained_Customer_Segmentation_Experiment.ipynb
plus additional analyses. Usage: python3 -I run_analysis.py <xlsx> <outdir>"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from scipy.stats import kruskal

warnings.filterwarnings("ignore")
XLSX, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
R = {}  # scalar results for the report
t0 = time.time()

# ---------------- 2. Load ----------------
cache = os.path.join(OUT, "raw.pkl")
if os.path.exists(cache):
    raw = pd.read_pickle(cache)
else:
    raw = pd.read_excel(XLSX, engine="openpyxl")
    raw.to_pickle(cache)
raw.columns = raw.columns.astype(str).str.strip().str.replace(" ", "_")
R["raw_rows"] = len(raw)
R["raw_customers"] = int(raw["CustomerID"].nunique())
print("Raw shape:", raw.shape, f"{time.time()-t0:.0f}s")

# ---------------- 3. Clean + split ----------------
df = raw.copy()
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df["Country"] = df["Country"].astype("string").str.strip()
R["rows_missing_customer"] = int(df["CustomerID"].isna().sum())
df = df.dropna(subset=["InvoiceNo","InvoiceDate","Quantity","UnitPrice","CustomerID","StockCode","Country"]).copy()
R["rows_after_dropna"] = len(df)
R["rows_cancel"] = int(df["InvoiceNo"].str.upper().str.startswith("C").sum())
df = df[(~df["InvoiceNo"].str.upper().str.startswith("C")) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
R["rows_clean"] = len(df)
R["customers_clean"] = int(df["CustomerID"].nunique())
df["CustomerID"] = df["CustomerID"].astype("int64")
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
test_start = df["InvoiceDate"].max().normalize() - pd.Timedelta(days=90)
train_tx = df[df["InvoiceDate"] < test_start].copy()
future_tx = df[df["InvoiceDate"] >= test_start].copy()
R.update(train_start=str(train_tx.InvoiceDate.min()), train_end=str(train_tx.InvoiceDate.max()),
         train_rows=len(train_tx), future_start=str(future_tx.InvoiceDate.min()),
         future_end=str(future_tx.InvoiceDate.max()), future_rows=len(future_tx),
         cutoff=str(test_start))
print(R)

# ---------------- 4. Features + outcomes ----------------
def mean_gap(dates):
    days = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    return days.diff().dropna().dt.days.mean() if len(days) >= 2 else np.nan

snapshot = train_tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
customers = train_tx.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate","max"), FirstPurchase=("InvoiceDate","min"),
    Frequency=("InvoiceNo","nunique"), Monetary=("Revenue","sum"),
    TotalUnits=("Quantity","sum"), ProductDiversity=("StockCode","nunique")).reset_index()
customers["Recency"] = (snapshot - customers["LastPurchase"].dt.normalize()).dt.days
customers["CustomerTenureDays"] = (customers["LastPurchase"] - customers["FirstPurchase"]).dt.days
train_orders = train_tx.groupby(["CustomerID","InvoiceNo"]).agg(OrderRevenue=("Revenue","sum")).reset_index()
customers["AverageOrderValue"] = customers["CustomerID"].map(train_orders.groupby("CustomerID")["OrderRevenue"].mean())
customers["MeanDaysBetweenPurchases"] = customers["CustomerID"].map(train_tx.groupby("CustomerID")["InvoiceDate"].apply(mean_gap))
country = train_tx.groupby("CustomerID")["Country"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
customers["Country"] = customers["CustomerID"].map(country)
future = future_tx.groupby("CustomerID").agg(
    FutureRevenue=("Revenue","sum"), FutureOrders=("InvoiceNo","nunique"),
    FutureUnits=("Quantity","sum"), FutureProductDiversity=("StockCode","nunique")).reset_index()
customers = customers.merge(future, on="CustomerID", how="left")
future_cols = ["FutureRevenue","FutureOrders","FutureUnits","FutureProductDiversity"]
customers[future_cols] = customers[future_cols].fillna(0)
future_orders = future_tx.groupby(["CustomerID","InvoiceNo"]).agg(OrderRevenue=("Revenue","sum")).reset_index()
customers["FutureAverageOrderValue"] = customers["CustomerID"].map(future_orders.groupby("CustomerID")["OrderRevenue"].mean()).fillna(0)
R["n_customers"] = len(customers)
R["n_future_purchasers"] = int((customers["FutureOrders"] > 0).sum())
R["n_single_date"] = int(customers["MeanDaysBetweenPurchases"].isna().sum())
R["n_countries"] = int(customers["Country"].nunique())
R["share_uk"] = float((customers["Country"] == "United Kingdom").mean())
desc = customers[["Recency","Frequency","Monetary","AverageOrderValue","TotalUnits","ProductDiversity",
                  "MeanDaysBetweenPurchases","FutureRevenue","FutureOrders","FutureAverageOrderValue",
                  "FutureUnits","FutureProductDiversity"]].describe(percentiles=[.5]).T
desc["skew"] = customers[desc.index].skew()
desc.to_csv(f"{OUT}/descriptives.csv")
print("Customers:", len(customers), "future purchasers:", R["n_future_purchasers"], f"{time.time()-t0:.0f}s")

# ---------------- 5. Scale ----------------
RFM_FEATURES = ["Recency","Frequency","Monetary"]
EXTENDED_FEATURES = ["Recency","Frequency","Monetary","AverageOrderValue","TotalUnits","ProductDiversity","MeanDaysBetweenPurchases"]
def prepare_matrix(data, cols):
    x = data[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf,-np.inf], np.nan)
        x[col] = x[col].fillna(x[col].median()).clip(lower=0)
        x[col] = np.log1p(x[col])
    return RobustScaler().fit_transform(x)
X_rfm = prepare_matrix(customers, RFM_FEATURES)
X_ext = prepare_matrix(customers, EXTENDED_FEATURES)

# ---------------- 6. Constraints ----------------
def generate_constraints(data, n_must=150, n_cannot=150, seed=42, max_draws=250000):
    rng = np.random.default_rng(seed)
    countries = data["Country"].astype(str).to_numpy()
    n = len(data); must=set(); cannot=set(); draws=0
    while (len(must)<n_must or len(cannot)<n_cannot) and draws<max_draws:
        i,j = rng.integers(0,n,size=2); draws += 1
        if i==j: continue
        pair=(int(min(i,j)),int(max(i,j)))
        if countries[i] == countries[j] and len(must)<n_must: must.add(pair)
        elif countries[i] != countries[j] and len(cannot)<n_cannot: cannot.add(pair)
    return sorted(must), sorted(cannot)
must_link, cannot_link = generate_constraints(customers, seed=RANDOM_STATE)
cty = customers["Country"].astype(str).to_numpy()
R["n_must"], R["n_cannot"] = len(must_link), len(cannot_link)
R["ml_uk_share"] = float(np.mean([cty[i]=="United Kingdom" for i,j in must_link]))
R["cl_involving_uk_share"] = float(np.mean([(cty[i]=="United Kingdom") or (cty[j]=="United Kingdom") for i,j in cannot_link]))
R["constrained_points"] = len({p for pr in must_link+cannot_link for p in pr})

# ---------------- 7. Soft constrained K-means (verbatim logic) ----------------
class SoftConstrainedKMeans:
    def __init__(self, n_clusters=4, constraint_weight=1.0, max_iter=35, n_init=4, random_state=42):
        self.k=n_clusters; self.weight=constraint_weight; self.max_iter=max_iter
        self.n_init=n_init; self.random_state=random_state
    def fit_predict(self, X, must_link, cannot_link):
        X=np.asarray(X,float); best_obj=np.inf; best=None
        master=np.random.default_rng(self.random_state); n=len(X)
        ml_adj=[[] for _ in range(n)]; cl_adj=[[] for _ in range(n)]
        for i,j in must_link: ml_adj[i].append(j); ml_adj[j].append(i)
        for i,j in cannot_link: cl_adj[i].append(j); cl_adj[j].append(i)
        self.iters_=[]
        for init in range(self.n_init):
            seed=int(master.integers(0,2**31-1)); rng=np.random.default_rng(seed)
            labels=KMeans(n_clusters=self.k,n_init=1,random_state=seed).fit_predict(X)
            for iteration in range(self.max_iter):
                old=labels.copy()
                for i in rng.permutation(n):
                    centroids=np.vstack([X[labels==c].mean(axis=0) if np.any(labels==c) else X[rng.integers(n)] for c in range(self.k)])
                    costs=((centroids-X[i])**2).sum(axis=1)
                    for c in range(self.k):
                        costs[c] += self.weight * sum(labels[j]!=c for j in ml_adj[i])
                        costs[c] += self.weight * sum(labels[j]==c for j in cl_adj[i])
                    labels[i]=int(np.argmin(costs))
                for c in range(self.k):
                    if not np.any(labels==c):
                        counts=np.bincount(labels,minlength=self.k)
                        centroids=np.vstack([X[labels==j].mean(axis=0) if np.any(labels==j) else X[0] for j in range(self.k)])
                        errors=((X-centroids[labels])**2).sum(axis=1)
                        for idx in np.argsort(errors)[::-1]:
                            if counts[labels[idx]]>1:
                                labels[idx]=c; break
                if np.array_equal(old,labels): break
            self.iters_.append(iteration+1)
            centroids=np.vstack([X[labels==c].mean(axis=0) for c in range(self.k)])
            inertia=((X-centroids[labels])**2).sum()
            violations=sum(labels[i]!=labels[j] for i,j in must_link)+sum(labels[i]==labels[j] for i,j in cannot_link)
            objective=inertia+self.weight*violations
            if objective<best_obj: best_obj=objective; best=labels.copy(); self.inertia_=inertia
        self.labels_=best; self.objective_=best_obj
        return best

def violation_rates(labels, ml, cl):
    return (np.mean([labels[i]!=labels[j] for i,j in ml]) if ml else np.nan,
            np.mean([labels[i]==labels[j] for i,j in cl]) if cl else np.nan)

# ---------------- 8. Fit ----------------
K=4
km_std = KMeans(n_clusters=K,n_init=20,random_state=RANDOM_STATE).fit(X_ext)
km_rfm = KMeans(n_clusters=K,n_init=20,random_state=RANDOM_STATE).fit(X_rfm)
labels_standard, labels_rfm = km_std.labels_, km_rfm.labels_
tc=time.time()
constrained=SoftConstrainedKMeans(n_clusters=K,constraint_weight=1.0,random_state=RANDOM_STATE)
labels_constrained=constrained.fit_predict(X_ext,must_link,cannot_link)
R["ckm_runtime_s"] = time.time()-tc
R["ckm_iters"] = constrained.iters_
R["inertia_std"] = float(km_std.inertia_); R["inertia_ckm"] = float(constrained.inertia_)
assignments={"Standard K-means":labels_standard,"RFM K-means":labels_rfm,"Constrained K-means":labels_constrained}
for name,labels in assignments.items(): print(name, pd.Series(labels).value_counts().sort_index().to_dict())
mlr, clr = violation_rates(labels_constrained,must_link,cannot_link)
R["ml_viol"], R["cl_viol"] = float(mlr), float(clr)
# baseline violation rates of the unconstrained solutions on the same pairs
for nm, lab in [("std", labels_standard), ("rfm", labels_rfm)]:
    a,b = violation_rates(lab, must_link, cannot_link); R[f"ml_viol_{nm}"]=float(a); R[f"cl_viol_{nm}"]=float(b)
print("violations", mlr, clr, f"{time.time()-t0:.0f}s")

# ---------------- 9. Internal metrics ----------------
def internal_metrics(X, labels):
    counts=pd.Series(labels).value_counts()
    out={"Clusters":len(counts),"Silhouette":np.nan,"DaviesBouldin":np.nan,"CalinskiHarabasz":np.nan,
         "MinClusterSize":int(counts.min()),"MaxClusterSize":int(counts.max())}
    if len(counts)>1 and len(labels)>len(counts):
        out["Silhouette"]=silhouette_score(X,labels,sample_size=min(10000,len(X)),random_state=RANDOM_STATE)
        out["DaviesBouldin"]=davies_bouldin_score(X,labels)
        out["CalinskiHarabasz"]=calinski_harabasz_score(X,labels)
    return out
stat_rows=[]
for name,labels in assignments.items():
    x=X_rfm if name=="RFM K-means" else X_ext
    stat_rows.append({"Method":name,**internal_metrics(x,labels)})
stat_results=pd.DataFrame(stat_rows); stat_results.to_csv(f"{OUT}/statistical_validation.csv",index=False)
print(stat_results)
# cross-space: every labelling scored in both spaces (addresses comparability)
cross=[]
for name,labels in assignments.items():
    for sp, X in [("EXT",X_ext),("RFM",X_rfm)]:
        m=internal_metrics(X,labels); cross.append({"Method":name,"Space":sp,"Silhouette":m["Silhouette"],
                                                     "DaviesBouldin":m["DaviesBouldin"],"CalinskiHarabasz":m["CalinskiHarabasz"]})
pd.DataFrame(cross).to_csv(f"{OUT}/cross_space_validation.csv",index=False)
# agreement
R["ari_std_ckm"]=float(adjusted_rand_score(labels_standard,labels_constrained))
R["ari_std_rfm"]=float(adjusted_rand_score(labels_standard,labels_rfm))
R["ari_rfm_ckm"]=float(adjusted_rand_score(labels_rfm,labels_constrained))
R["n_reassigned_vs_std"]=None  # computed below after matching

# ---------------- 10. Eta squared ----------------
FUTURE_METRICS=["FutureRevenue","FutureOrders","FutureAverageOrderValue","FutureUnits","FutureProductDiversity"]
def eta_squared(values,labels):
    y=np.asarray(values,dtype=float); lab=np.asarray(labels)
    total=((y-y.mean())**2).sum()
    if total==0: return np.nan
    between=sum((lab==c).sum()*(y[lab==c].mean()-y.mean())**2 for c in np.unique(lab))
    return between/total
rows=[]
for method,labels in assignments.items():
    for metric in FUTURE_METRICS:
        y=customers[metric].to_numpy()
        H,p=kruskal(*[y[labels==c] for c in np.unique(labels)])
        rows.append({"Method":method,"FutureMetric":metric,"EtaSquared":eta_squared(y,labels),"KW_H":H,"KW_p":p})
business_separation=pd.DataFrame(rows); business_separation.to_csv(f"{OUT}/future_business_separation.csv",index=False)
print(business_separation.pivot(index="FutureMetric",columns="Method",values="EtaSquared").round(4))

# Bootstrap CIs for eta-squared and differences (customers resampled, labels fixed)
B=1000; rng=np.random.default_rng(2026); n=len(customers)
Y={m:customers[m].to_numpy(float) for m in FUTURE_METRICS}
boot={(meth,m):[] for meth in assignments for m in FUTURE_METRICS}
for b in range(B):
    idx=rng.integers(0,n,n)
    for meth,lab in assignments.items():
        lb=lab[idx]
        for m in FUTURE_METRICS: boot[(meth,m)].append(eta_squared(Y[m][idx],lb))
ci_rows=[]
for m in FUTURE_METRICS:
    s=np.array(boot[("Standard K-means",m)]); r=np.array(boot[("RFM K-means",m)]); c=np.array(boot[("Constrained K-means",m)])
    for meth,arr in [("Standard K-means",s),("RFM K-means",r),("Constrained K-means",c)]:
        ci_rows.append({"FutureMetric":m,"Quantity":meth,"lo":np.percentile(arr,2.5),"hi":np.percentile(arr,97.5)})
    for nm,d in [("Constrained-Standard",c-s),("RFM-Standard",r-s),("RFM-Constrained",r-c)]:
        ci_rows.append({"FutureMetric":m,"Quantity":nm,"mean":d.mean(),"lo":np.percentile(d,2.5),"hi":np.percentile(d,97.5),"p_le0":float(np.mean(d<=0))})
pd.DataFrame(ci_rows).to_csv(f"{OUT}/eta_bootstrap.csv",index=False)
print(f"bootstrap done {time.time()-t0:.0f}s")

# ---------------- 11. Profiles ----------------
def future_profile(data,labels,method):
    d=data.copy(); d["Cluster"]=labels
    p=d.groupby("Cluster").agg(Customers=("CustomerID","nunique"),
        Recency=("Recency","median"),Frequency=("Frequency","median"),Monetary=("Monetary","median"),
        AOV=("AverageOrderValue","median"),Diversity=("ProductDiversity","median"),
        FutureRevenueMean=("FutureRevenue","mean"),FutureRevenueMedian=("FutureRevenue","median"),
        FutureOrdersMean=("FutureOrders","mean"),FutureAOVMean=("FutureAverageOrderValue","mean"),
        FutureUnitsMean=("FutureUnits","mean"),FutureProductDiversityMean=("FutureProductDiversity","mean"),
        ReturnRate=("FutureOrders",lambda s:(s>0).mean())).reset_index()
    p["CustomerShare"]=p["Customers"]/len(d)
    totals=d.groupby("Cluster")["FutureRevenue"].sum()
    p["FutureRevenueShare"]=p["Cluster"].map(totals)/max(d["FutureRevenue"].sum(),1e-12)
    p["UKShare"]=p["Cluster"].map(d.assign(uk=d.Country.eq("United Kingdom")).groupby("Cluster")["uk"].mean())
    p["Method"]=method
    return p
profiles=pd.concat([future_profile(customers,l,m) for m,l in assignments.items()],ignore_index=True)
profiles.to_csv(f"{OUT}/future_business_segment_profiles.csv",index=False)
print(profiles.round(3).to_string())

# ---------------- 12. Weight sensitivity ----------------
sensitivity_rows=[]
for w in [0.1,0.5,1.0,2.0,5.0]:
    model=SoftConstrainedKMeans(n_clusters=K,constraint_weight=w,n_init=3,max_iter=30,random_state=RANDOM_STATE)
    lab=model.fit_predict(X_ext,must_link,cannot_link)
    a,b=violation_rates(lab,must_link,cannot_link)
    sensitivity_rows.append({"Weight":w,**internal_metrics(X_ext,lab),"MustLinkViolation":a,"CannotLinkViolation":b,
        **{f"{m}Eta2":eta_squared(customers[m],lab) for m in FUTURE_METRICS},
        "ARI_vs_standard":adjusted_rand_score(labels_standard,lab)})
    print("w",w,f"{time.time()-t0:.0f}s")
sensitivity=pd.DataFrame(sensitivity_rows); sensitivity.to_csv(f"{OUT}/constraint_weight_sensitivity.csv",index=False)
print(sensitivity.round(4).to_string())

# ---------------- 13. Robustness ----------------
robust_rows=[]
for seed in [11,22,33,44,55]:
    ml,cl=generate_constraints(customers,seed=seed)
    model=SoftConstrainedKMeans(n_clusters=K,constraint_weight=1.0,n_init=3,max_iter=30,random_state=seed)
    lab=model.fit_predict(X_ext,ml,cl)
    a,b=violation_rates(lab,ml,cl)
    robust_rows.append({"Seed":seed,"Silhouette":silhouette_score(X_ext,lab,sample_size=min(10000,len(X_ext)),random_state=RANDOM_STATE),
        **{f"{m}Eta2":eta_squared(customers[m],lab) for m in FUTURE_METRICS},
        "MustLinkViolation":a,"CannotLinkViolation":b,"ARI_vs_standard":adjusted_rand_score(labels_standard,lab)})
    print("seed",seed,f"{time.time()-t0:.0f}s")
robustness=pd.DataFrame(robust_rows); robustness.to_csv(f"{OUT}/constraint_set_robustness.csv",index=False)
robustness.drop(columns="Seed").agg(["mean","std"]).T.to_csv(f"{OUT}/constraint_set_robustness_summary.csv")
print(robustness.round(4).to_string())

# ---------------- Extra: choice of k ----------------
krows=[]
for k in range(2,9):
    for nm,X in [("EXT",X_ext),("RFM",X_rfm)]:
        km=KMeans(n_clusters=k,n_init=20,random_state=RANDOM_STATE).fit(X)
        krows.append({"k":k,"Space":nm,"Inertia":km.inertia_,"Silhouette":silhouette_score(X,km.labels_),
                      "DaviesBouldin":davies_bouldin_score(X,km.labels_),"CalinskiHarabasz":calinski_harabasz_score(X,km.labels_),
                      "FutureOrdersEta2":eta_squared(customers["FutureOrders"],km.labels_),
                      "FutureRevenueEta2":eta_squared(customers["FutureRevenue"],km.labels_)})
pd.DataFrame(krows).to_csv(f"{OUT}/k_selection.csv",index=False)
print(pd.DataFrame(krows).round(4).to_string())

# ---------------- 14. PCA ----------------
pca=PCA(n_components=2,random_state=RANDOM_STATE); coords=pca.fit_transform(X_ext)
R["pca_evr"]=pca.explained_variance_ratio_.tolist()
pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Standard":labels_standard,"RFM":labels_rfm,
              "Constrained":labels_constrained}).to_csv(f"{OUT}/pca_coords.csv",index=False)
pd.DataFrame(km_std.cluster_centers_,columns=EXTENDED_FEATURES).to_csv(f"{OUT}/centroids_standard_scaled.csv")
customers.assign(Standard=labels_standard,RFM=labels_rfm,Constrained=labels_constrained).to_csv(
    f"{OUT}/customer_features_and_future_outcomes.csv",index=False)

# ---------------- Extra: robust effect sizes and report figures ----------------
from scipy.optimize import linear_sum_assignment
rob=[]
for meth,lab in assignments.items():
    for m in FUTURE_METRICS:
        y=customers[m].to_numpy(float)
        H,_=kruskal(*[y[lab==c] for c in np.unique(lab)])
        rob.append({"Method":meth,"FutureMetric":m,"EtaSquaredLog":eta_squared(np.log1p(y),lab),
                    "EpsilonSquaredKW":H/(len(y)-1)})
pd.DataFrame(rob).to_csv(f"{OUT}/robust_effect_sizes.csv",index=False)
ct=pd.crosstab(labels_standard,labels_constrained).to_numpy()
r_,c_=linear_sum_assignment(-ct)
R["n_reassigned_vs_std"]=int(len(customers)-ct[r_,c_].sum())
p=np.bincount(labels_standard)/len(labels_standard)
R["chance_ml_viol"]=float(1-(p**2).sum()); R["chance_cl_viol"]=float((p**2).sum())
fr=customers["FutureRevenue"].sort_values(ascending=False)
top=int(round(0.01*len(fr))); R["top1pct_n"]=top; R["top1pct_future_rev_share"]=float(fr.head(top).sum()/fr.sum())
R["return_rate"]=float((customers["FutureOrders"]>0).mean())

import sklearn, scipy
R["versions"]={"python":sys.version.split()[0],"numpy":np.__version__,"pandas":pd.__version__,
               "sklearn":sklearn.__version__,"scipy":scipy.__version__}
R["total_runtime_s"]=time.time()-t0
json.dump(R,open(f"{OUT}/summary.json","w"),indent=2,default=str)
print(json.dumps(R,indent=1,default=str))

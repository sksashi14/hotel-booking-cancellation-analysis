import pandas as pd, numpy as np, json, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); R = {}

# 1. LOAD
df = pd.read_csv("hotels.csv")
R["raw_shape"] = df.shape
R["missing"] = df.isna().sum()[lambda s: s > 0].to_dict()
R["cancel_rate_raw"] = round(df.is_canceled.mean()*100, 1)
R["dups"] = int(df.duplicated().sum())

# 2. CLEAN
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["agent"] = df["agent"].fillna(0).astype(int).astype(str).replace("0", "No agent")
df["company"] = df["company"].fillna(0)
df["has_company"] = (df["company"] != 0).astype(int)
df = df.drop(columns="company")
df = df.drop_duplicates()
R["after_dups"] = df.shape[0]
guests0 = int(((df.adults + df.children + df.babies) == 0).sum())
df = df[(df.adults + df.children + df.babies) > 0]
R["zero_guests_removed"] = guests0
R["adr_neg"] = int((df.adr < 0).sum())
R["adr_q99"] = float(df.adr.quantile(.99)); R["adr_max"] = float(df.adr.max())
R["adr_gt1000"] = int((df.adr > 1000).sum())
df = df[(df.adr >= 0) & (df.adr < 1000)]
q1, q3 = df.adr.quantile([.25, .75]); iqr = q3-q1
cap = q3 + 1.5*iqr; R["adr_iqr_cap"] = float(cap); R["adr_above_cap"] = int((df.adr > cap).sum())
df["adr_capped"] = df.adr.clip(upper=cap)
R["lead_max"] = int(df.lead_time.max())
R["clean_shape"] = df.shape
R["cancel_rate"] = round(df.is_canceled.mean()*100, 1)

# 3. FEATURES
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
df["month_num"] = df.arrival_date_month.map({m:i+1 for i,m in enumerate(months)})
df["total_nights"] = df.stays_in_weekend_nights + df.stays_in_week_nights
df["total_guests"] = df.adults + df.children + df.babies
df["revenue"] = df.adr_capped * df.total_nights
df["room_changed"] = (df.reserved_room_type != df.assigned_room_type).astype(int)
df["has_kids"] = ((df.children + df.babies) > 0).astype(int)

# 4. EDA
R["by_hotel"] = df.groupby("hotel").agg(bookings=("is_canceled","size"), cancel_rate=("is_canceled","mean"), adr=("adr_capped","mean"), lead=("lead_time","mean"), nights=("total_nights","mean")).round(3).to_dict("index")
R["lead_stats"] = df.lead_time.describe().round(1).to_dict()
R["adr_stats"] = df.adr_capped.describe().round(1).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(10,4))
sns.countplot(data=df, x="hotel", hue="is_canceled", ax=ax[0], palette=["#2a9d8f","#e76f51"])
ax[0].set_title("Bookings by hotel type and cancellation"); ax[0].set_xlabel("Hotel"); ax[0].set_ylabel("Number of bookings"); ax[0].legend(title="Canceled", labels=["No","Yes"])
sns.histplot(df.lead_time, bins=50, ax=ax[1], color="#264653"); ax[1].set_title("Distribution of lead time"); ax[1].set_xlabel("Lead time (days)")
plt.tight_layout(); plt.savefig("figs/fig1_hotel_lead.png", dpi=130); plt.close()

m = df.groupby(["month_num","hotel"]).agg(adr=("adr_capped","mean"), bookings=("is_canceled","size")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(11,4))
sns.lineplot(data=m, x="month_num", y="adr", hue="hotel", marker="o", ax=ax[0]); ax[0].set_xticks(range(1,13)); ax[0].set_title("Average daily rate by arrival month"); ax[0].set_xlabel("Month (1 = January)"); ax[0].set_ylabel("ADR (euro)")
sns.lineplot(data=m, x="month_num", y="bookings", hue="hotel", marker="o", ax=ax[1]); ax[1].set_xticks(range(1,13)); ax[1].set_title("Bookings by arrival month"); ax[1].set_xlabel("Month (1 = January)"); ax[1].set_ylabel("Bookings")
plt.tight_layout(); plt.savefig("figs/fig2_seasonality.png", dpi=130); plt.close()
R["adr_by_month"] = df.groupby("month_num").adr_capped.mean().round(1).to_dict()
R["bookings_by_month"] = df.groupby("month_num").size().to_dict()
R["adr_hotel_month"] = {h: df[df.hotel==h].groupby("month_num").adr_capped.mean().round(1).to_dict() for h in df.hotel.unique()}

seg = df.groupby("market_segment").agg(n=("is_canceled","size"), cr=("is_canceled","mean")).query("n>100").sort_values("cr", ascending=False)
R["seg"] = seg.round(3).to_dict("index")
dep = df.groupby("deposit_type").agg(n=("is_canceled","size"), cr=("is_canceled","mean")).round(3)
R["deposit"] = dep.to_dict("index")
fig, ax = plt.subplots(1, 2, figsize=(11,4))
sns.barplot(x=seg.cr*100, y=seg.index, ax=ax[0], color="#e76f51"); ax[0].set_title("Cancellation rate by market segment"); ax[0].set_xlabel("Cancellation rate (%)"); ax[0].set_ylabel("")
sns.barplot(x=dep.cr*100, y=dep.index, ax=ax[1], color="#264653"); ax[1].set_title("Cancellation rate by deposit type"); ax[1].set_xlabel("Cancellation rate (%)"); ax[1].set_ylabel("")
plt.tight_layout(); plt.savefig("figs/fig3_cancel_segment.png", dpi=130); plt.close()

df["lead_bin"] = pd.cut(df.lead_time, [-1,7,30,90,180,365,800], labels=["0-7","8-30","31-90","91-180","181-365","366+"])
lb = df.groupby("lead_bin").is_canceled.mean()*100
R["lead_bin"] = lb.round(1).to_dict()
num = ["is_canceled","lead_time","total_nights","total_guests","adr_capped","booking_changes","previous_cancellations","total_of_special_requests","required_car_parking_spaces","days_in_waiting_list","is_repeated_guest","room_changed"]
corr = df[num].corr()
R["corr_cancel"] = corr["is_canceled"].drop("is_canceled").round(3).sort_values().to_dict()
fig, ax = plt.subplots(1, 2, figsize=(13,5))
sns.barplot(x=lb.values, y=lb.index.astype(str), ax=ax[0], color="#2a9d8f"); ax[0].set_title("Cancellation rate by lead time (days)"); ax[0].set_xlabel("Cancellation rate (%)"); ax[0].set_ylabel("Lead time band")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1], annot_kws={"size":6}, cbar_kws={"shrink":.7}); ax[1].set_title("Correlation matrix of numeric features")
plt.tight_layout(); plt.savefig("figs/fig4_lead_corr.png", dpi=130); plt.close()

ctry = df.groupby("country").agg(n=("is_canceled","size"), cr=("is_canceled","mean")).sort_values("n", ascending=False).head(8)
R["country"] = ctry.round(3).to_dict("index")
fig, ax = plt.subplots(figsize=(6,4)); sns.boxplot(data=df, x="hotel", y="adr", ax=ax, palette=["#8ecae6","#ffb703"]); ax.set_title("ADR before capping (outliers visible)"); ax.set_ylabel("ADR (euro)"); ax.set_xlabel("Hotel")
plt.tight_layout(); plt.savefig("figs/fig5_adr_box.png", dpi=130); plt.close()

# 5. CLUSTERING
cf = ["lead_time","total_nights","total_guests","adr_capped","total_of_special_requests","booking_changes"]
sub = df[df.is_canceled==0].copy() if False else df.copy()
X = StandardScaler().fit_transform(sub[cf])
rng = np.random.RandomState(42); idx = rng.choice(len(X), 8000, replace=False)
ks = range(2,9); inertia=[]; sil=[]
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X[idx], km.labels_[idx]))
R["inertia"] = [round(i) for i in inertia]; R["sil"] = [round(s,3) for s in sil]
fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(list(ks), inertia, marker="o", color="#264653"); ax[0].set_title("Elbow plot"); ax[0].set_xlabel("Number of clusters (k)"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, marker="o", color="#e76f51"); ax[1].set_title("Silhouette score by k"); ax[1].set_xlabel("Number of clusters (k)"); ax[1].set_ylabel("Silhouette score")
plt.tight_layout(); plt.savefig("figs/fig6_elbow.png", dpi=130); plt.close()
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X); sub["cluster"] = km.labels_
R["k"] = K; R["sil_k"] = round(silhouette_score(X[idx], km.labels_[idx]),3)
prof = sub.groupby("cluster").agg(size=("cluster","size"), lead=("lead_time","mean"), nights=("total_nights","mean"), guests=("total_guests","mean"), adr=("adr_capped","mean"), requests=("total_of_special_requests","mean"), changes=("booking_changes","mean"), cancel=("is_canceled","mean"), resort_share=("hotel", lambda s:(s=="Resort Hotel").mean())).round(2)
R["clusters"] = prof.to_dict("index")
from sklearn.decomposition import PCA
p = PCA(2, random_state=42).fit(X); Z = p.transform(X[idx]); R["pca_var"] = [round(v,3) for v in p.explained_variance_ratio_]
fig, ax = plt.subplots(figsize=(7,5))
sc = ax.scatter(Z[:,0], Z[:,1], c=km.labels_[idx], cmap="tab10", s=6, alpha=.6); ax.set_title("Clusters projected onto two principal components"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(*sc.legend_elements(), title="Cluster")
plt.tight_layout(); plt.savefig("figs/fig7_clusters.png", dpi=130); plt.close()

# 6. SUPERVISED
feat_num = ["lead_time","total_nights","total_guests","adr_capped","previous_cancellations","previous_bookings_not_canceled","booking_changes","days_in_waiting_list","required_car_parking_spaces","total_of_special_requests","is_repeated_guest","has_company","month_num","has_kids"]
feat_cat = ["hotel","meal","market_segment","distribution_channel","deposit_type","customer_type"]
y = df.is_canceled; Xd = df[feat_num+feat_cat]
Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=.2, stratify=y, random_state=42)
pre = ColumnTransformer([("num", StandardScaler(), feat_num), ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat)])
lr = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))])
rf = Pipeline([("pre", pre), ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42))])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
res = {}
for name, mdl in [("Logistic Regression", lr), ("Random Forest", rf)]:
    s = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
    mdl.fit(Xtr, ytr); pr = mdl.predict(Xte); pp = mdl.predict_proba(Xte)[:,1]
    res[name] = dict(cv_auc_mean=round(s.mean(),3), cv_auc_std=round(s.std(),3), acc=round(accuracy_score(yte,pr),3), f1=round(f1_score(yte,pr),3), auc=round(roc_auc_score(yte,pp),3), cm=confusion_matrix(yte,pr).tolist(), rep=classification_report(yte,pr,output_dict=True))
R["models"] = res
R["baseline_acc"] = round(1-yte.mean(),3)
R["train_test"] = [len(Xtr), len(Xte)]
fig, ax = plt.subplots(figsize=(6,5))
for name, mdl in [("Logistic Regression", lr), ("Random Forest", rf)]:
    RocCurveDisplay.from_estimator(mdl, Xte, yte, ax=ax, name=name)
ax.plot([0,1],[0,1],"k--",alpha=.4); ax.set_title("ROC curves on the held-out test set")
plt.tight_layout(); plt.savefig("figs/fig8_roc.png", dpi=130); plt.close()
names = rf.named_steps["pre"].get_feature_names_out(); imp = pd.Series(rf.named_steps["m"].feature_importances_, index=names).sort_values(ascending=False).head(12)
imp.index = [i.replace("num__","").replace("cat__","") for i in imp.index]
R["importance"] = imp.round(3).to_dict()
fig, ax = plt.subplots(figsize=(7,5)); sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#264653"); ax.set_title("Top 12 Random Forest feature importances"); ax.set_xlabel("Importance"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig("figs/fig9_importance.png", dpi=130); plt.close()
json.dump(R, open("results.json","w"), indent=1, default=str)
df.to_csv("hotels_clean.csv", index=False)
print(json.dumps(R, indent=1, default=str))

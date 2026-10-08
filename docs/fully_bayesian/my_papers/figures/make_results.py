"""T-IV paper figures 2-5 from the final run artifacts (restored from the session transcript on 2026-10-08).

Produces fig_sequential, fig_uncertainty, fig_selection and fig_pip (pdf+png) next to this script from
outputs/loeo_final/20261002_165129 (main, select and sensors stages).  Run from the repository root:
    python docs/fully_bayesian/my_papers/figures/make_results.py
"""
import json,glob,csv,numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'font.family':'DejaVu Sans'})
R='outputs/loeo_final/20261002_165129'; OUT='docs/fully_bayesian/my_papers/figures/'
rows=list(csv.DictReader(open(R+'/report/main_macro.csv')))
def macro(m,met):
    d={}
    for r in rows:
        if r['method']==m and r['metric']==met and r['t']!='final': d[int(r['t'])]=(float(r['mean']),float(r['se']))
    return d
meth=[('proposed','Proposed','C0','o'),('hb_full','HB-full','C1','s'),('ebmap','EB-MAP','C2','^'),('pooled','Pooled','C3','D'),('indep','Indep','C4','v')]
# Fig 1: sequential MLPD / AUROC
fig,ax=plt.subplots(1,2,figsize=(7.0,2.5))
for k,(m,lab,c,mk) in enumerate(meth):
    for a,met in zip(ax,['auroc','mlpd']):
        d=macro(m,met); ts=sorted(d); 
        if not ts: continue
        off=(k-2)*0.15
        a.errorbar([t+off for t in ts],[d[t][0] for t in ts],yerr=[d[t][1] for t in ts],label=lab,color=c,marker=mk,ms=3.5,lw=1,capsize=2)
ax[0].set_ylabel('Macro AUROC'); ax[1].set_ylabel('Macro MLPD (nat/episode)')
for a,ttl in zip(ax,['(a)','(b)']):
    a.set_xlabel('Context size $t$ (labels)'); a.set_xticks([0,5,10,20]); a.grid(alpha=.3); a.set_title(ttl,loc='left')
ax[1].legend(loc='lower right',ncol=1,frameon=False)
fig.tight_layout(); fig.savefig(OUT+'fig_sequential.pdf'); fig.savefig(OUT+'fig_sequential.png',dpi=200)
# Fig 2: uncertainty decay, macro with per-evaluator thin lines
F={}
for f in sorted(glob.glob(R+'/folds/*.json')):
    if f.endswith(('_select.json','_sensors.json')): continue
    d=json.load(open(f)); F[d['name']]=d
REG=('0','5','10','20')
fig,ax=plt.subplots(figsize=(3.4,2.5))
for n,d in F.items():
    P=d['proposed']; ts=[int(t) for t in REG if t in P]
    ax.plot(ts,[P[str(t)]['epi_mean'] for t in ts],color='C0',alpha=.3,lw=.8)
    ax.plot(ts,[P[str(t)]['ale_mean'] for t in ts],color='C3',alpha=.3,lw=.8,ls='--')
e=macro('proposed','epi_mean'); a_=macro('proposed','ale_mean'); ts=sorted(e)
ax.errorbar(ts,[e[t][0] for t in ts],yerr=[e[t][1] for t in ts],color='C0',marker='o',ms=3.5,lw=1.6,capsize=2,label=r'epistemic $\bar U^{\mathrm{epi}}(t)$')
ax.errorbar(ts,[a_[t][0] for t in ts],yerr=[a_[t][1] for t in ts],color='C3',marker='s',ms=3.5,lw=1.6,ls='--',capsize=2,label=r'aleatoric $\bar U^{\mathrm{ale}}(t)$')
ax.set_xlabel('Context size $t$ (labels)'); ax.set_ylabel('Mean uncertainty term'); ax.set_xticks([0,5,10,20]); ax.grid(alpha=.3); ax.legend(frameon=False,loc='center right')
fig.tight_layout(); fig.savefig(OUT+'fig_uncertainty.pdf'); fig.savefig(OUT+'fig_uncertainty.png',dpi=200)
# Fig 3: selection size curve (drop k=40, 3 folds) + LOSO
sel=[r for r in csv.DictReader(open(R+'/report/select_sizes.csv')) if int(r['n_folds'])==10 and r['candidate'] not in ('pip',)]
full=[r for r in sel if r['candidate']=='full'][0]; path=[r for r in sel if r['candidate'].startswith('k=')]
fig,ax=plt.subplots(1,2,figsize=(7.0,2.4),gridspec_kw={'width_ratios':[1.5,1]})
ks=[int(r['n_features']) for r in path]; mu=[float(r['mlpd_mean']) for r in path]; se=[float(r['mlpd_se']) for r in path]
ax[0].errorbar(ks,mu,yerr=se,marker='o',ms=3.5,lw=1,capsize=2,color='C0',label='projected subset of size $k$')
ax[0].axhline(float(full['mlpd_mean']),color='k',ls='--',lw=1,label='full cleaned bank ($d-1=40$)')
ax[0].axhspan(float(full['mlpd_mean'])-float(full['mlpd_se']),float(full['mlpd_mean'])+float(full['mlpd_se']),color='k',alpha=.08)
ax[0].axvline(1,color='C3',ls=':',lw=1.2,label='selected $k^\\ast=1$ (one-SE rule)')
ax[0].set_xlabel('Number of features $k$'); ax[0].set_ylabel('Held-out MLPD (refit, macro)'); ax[0].set_xscale('log'); ax[0].set_xticks([1,2,5,10,20,40]); ax[0].set_xticklabels(['1','2','5','10','20','40']); ax[0].grid(alpha=.3,which='both'); ax[0].legend(frameon=False,fontsize=6.5,loc='lower left'); ax[0].set_title('(a)',loc='left')
loso=[('VerAccel\n($a_z$)',-0.0336,0.0227),('LongAccel\n($a_x$)',-0.0034,0.0138),('LatAccel\n($a_y$)',0.0054,0.0065),('Bounce\n($\\dot z$)',0.0043,0.0062),('Pitch\n($\\dot\\theta$)',-0.0038,0.0053)]
ax[1].bar(range(5),[l[1] for l in loso],yerr=[l[2] for l in loso],color=['C3','C0','C0','C1','C1'],capsize=2,width=.6)
ax[1].axhline(0,color='k',lw=.8); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([l[0] for l in loso],fontsize=6.5); ax[1].set_ylabel(r'$\Delta$MLPD when channel removed'); ax[1].grid(alpha=.3,axis='y'); ax[1].set_title('(b)',loc='left')
fig.tight_layout(); fig.savefig(OUT+'fig_selection.pdf'); fig.savefig(OUT+'fig_selection.png',dpi=200)
# Fig 4: PIP + mu by channel (features with PIP>=0.1 shown; all 44 would be dense) -> show all, grouped
roles=list(csv.DictReader(open(R+'/report/feature_roles_macro.csv')))
order=['IMU_VerAccelVal','IMU_LongAccelVal','IMU_LatAccelVal','Bounce_rate_6D','Pitch_rate_6D']
short={'IMU_VerAccelVal':'$a_z$','IMU_LongAccelVal':'$a_x$','IMU_LatAccelVal':'$a_y$','Bounce_rate_6D':'$\\dot z$','Pitch_rate_6D':'$\\dot\\theta$'}
roles.sort(key=lambda r:(order.index(r['group']),-float(r['pip'])))
fig,ax=plt.subplots(2,1,figsize=(7.0,3.6),sharex=True,gridspec_kw={'height_ratios':[1,1]})
x=np.arange(len(roles)); cols=[f'C{order.index(r["group"])}' for r in roles]
ax[0].bar(x,[float(r['pip']) for r in roles],color=cols,width=.75); ax[0].axhline(.5,color='k',ls=':',lw=1); ax[0].set_ylabel('PIP'); ax[0].set_ylim(0,1); ax[0].grid(alpha=.3,axis='y')
ax[1].errorbar(x,[float(r['mu_mean']) for r in roles],yerr=[float(r['between_sd']) for r in roles],fmt='o',ms=2.5,color='k',ecolor=cols,elinewidth=1.5,capsize=0)
ax[1].axhline(0,color='k',lw=.8); ax[1].set_ylabel(r'$\mu_j\ \pm\sqrt{\Sigma_{jj}}$'); ax[1].grid(alpha=.3,axis='y')
ax[1].set_xticks(x); ax[1].set_xticklabels([r['feature'].split('__')[1].replace('_','-') for r in roles],rotation=90,fontsize=5.5)
# channel brackets
for g in order:
    idx=[i for i,r in enumerate(roles) if r['group']==g]
    ax[0].text((idx[0]+idx[-1])/2,0.92,short[g],ha='center',fontsize=8,color=f'C{order.index(g)}')
    if idx[-1]<len(roles)-1:
        for a in ax: a.axvline(idx[-1]+.5,color='gray',lw=.5,ls='-')
fig.tight_layout(); fig.savefig(OUT+'fig_pip.pdf'); fig.savefig(OUT+'fig_pip.png',dpi=200)
print("ok")

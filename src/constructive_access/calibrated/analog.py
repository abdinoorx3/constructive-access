"""Direct bounded analog-score inference without virtual coin noise.

A valid mean confidence contract plus the same conceptual randomization kernel
supports the physical Bernoulli-KL bound; integer successes are unnecessary.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import math
from constructive_access.frontier import energy_certificate as ec
from constructive_access.virtual.contrast import sqrt_upper
try:
    from .calibration import calibrated_support
except ImportError:
    from calibration import calibrated_support


def exact(x):
    if type(x) not in (int,F):raise TypeError('Certificate inputs must be integers or Fractions')
    return F(x)


def _stored_number(x):
    """Interpret a finite digital reading exactly, including stored binary floats."""
    if isinstance(x,F):return x
    if isinstance(x,int):return F(x)
    if hasattr(x,'as_integer_ratio'):
        if not math.isfinite(float(x)):raise ValueError('finite observations required')
        a,b=x.as_integer_ratio();return F(int(a),int(b))
    raise TypeError('Use numeric digital readings with exact integer ratios')


def quantized_pair_summary(left,right,denominator=2**24):
    """Descriptive exact score mean with certified rounding radius.

    Stored floats are treated as exact digital readings, not as latent physical
    states. Any sensor calibration/physics error remains separate. Only final
    scores are rounded, by exact integer arithmetic. No inferential claim is made.
    """
    if type(denominator) is not int or denominator<1:raise ValueError('positive integer quantizer required')
    n=total=squares=0
    for a,b in zip(left,right,strict=True):
        a,b=_stored_number(a),_stored_number(b)
        if not -1<=a<=1 or not -1<=b<=1:raise ValueError('analog observations must lie in [-1,1]')
        w=(1+a*b)/2
        q=(2*w.numerator*denominator+w.denominator)//(2*w.denominator)
        n+=1;total+=q;squares+=q*q
    if n<1:raise ValueError('nonempty pairs required')
    mean=F(total,n*denominator);second=F(squares,n*denominator**2)
    return {'records':n,'score_sum':F(total,denominator),'mean':mean,
            'mean_quantization_radius':F(1,2*denominator),
            'quantized_empirical_variance':second-mean*mean,
            'quantized_sample_variance':None if n==1 else F(n,n-1)*(second-mean*mean),
            'quantizer_denominator':denominator,
            'status':'descriptive_only; inference requires additional contracts'}


def certify_means(lags,coefficients,records,means,durations,calibration_vertices,
                  alpha=F(1,20),calibration_failure=F(0),mean_errors=None,
                  conditional_radii=None,score_bounds=None,selection='predeclared',depth=6):
    """One-sided fixed-contrast certificate for bounded score means.

    Fixed quotas, a predeclared/independent-pilot contrast and one fixed target
    mean per setting are required. Conditional mean drift may be bounded by
    conditional_radii. mean_errors bounds numerical/quantization errors in the
    supplied empirical means. Physical stationarity, detector and clock premises
    are external. Calibration region failure is combined by a union bound.
    """
    if selection not in ('predeclared','independent_pilot'):raise ValueError('unsupported data-dependent selection')
    coeff=list(map(exact,coefficients));means=list(map(exact,means));dur=list(map(exact,durations));n=list(records)
    K=len(coeff)
    if not K or not len(lags)==len(n)==len(means)==len(dur)==K or not any(coeff):raise ValueError('matching nonempty arrays required')
    if any(type(v) is not int or v<=0 for v in n) or any(not 0<=p<=1 for p in means) or any(t<=0 for t in dur):raise ValueError('invalid quotas, means or full durations')
    alpha,calfail=map(exact,(alpha,calibration_failure))
    if not 0<alpha<1 or not 0<=calfail<1 or alpha+calfail>=1:raise ValueError('invalid confidence allocation')
    err=[F(0)]*K if mean_errors is None else list(map(exact,mean_errors))
    drift=[F(0)]*K if conditional_radii is None else list(map(exact,conditional_radii))
    bounds=[(F(0),F(1))]*K if score_bounds is None else [tuple(map(exact,b)) for b in score_bounds]
    if len(err)!=K or len(drift)!=K or len(bounds)!=K or any(e<0 for e in err+drift):raise ValueError('invalid numerical or conditional error radii')
    if any(len(b)!=2 or not 0<=b[0]<b[1]<=1 for b in bounds):raise ValueError('score ranges must lie in [0,1]')
    if any(p+e<b[0] or p-e>b[1] for p,e,b in zip(means,err,bounds)):raise ValueError('empirical mean outside declared score range')
    support=calibrated_support(lags,coeff,calibration_vertices,depth)
    variance_proxy=sum((a*a*(b[1]-b[0])**2/N for a,b,N in zip(coeff,bounds,n)),F(0))
    radius=sqrt_upper(ec.log_interval(1/alpha)[1]*variance_proxy/2)
    observed=sum((a*p for a,p in zip(coeff,means)),F(0))
    numerical=sum((abs(a)*e for a,e in zip(coeff,err)),F(0))
    conditional=sum((abs(a)*d for a,d in zip(coeff,drift)),F(0))
    signed=observed-support['reversible_support_upper']-radius-numerical-conditional
    gap=max(F(0),signed)
    denominator=sum((abs(a)*sqrt_upper(t) for a,t in zip(coeff,dur)),F(0))**2
    return {'status':'positive_certificate' if gap else 'no_positive_certificate',
            'records':n,'means':means,'coefficients':coeff,'lags':list(lags),'durations':dur,
            'support':support,'empirical_contrast':observed,'sampling_radius_upper':radius,
            'quantization_penalty':numerical,'conditional_mean_penalty':conditional,
            'concentration_variance_proxy':variance_proxy,'signed_contrast_lower':signed,'contrast_lower':gap,
            'entropy_production_rate_lower':8*gap*gap/denominator,'entropy_denominator_upper':denominator,
            'science_failure_upper':alpha,'calibration_failure_upper':calfail,
            'confidence_failure_upper':alpha+calfail,'score_bounds':bounds,'selection':selection,
            'physical_contract':'stationary even-state homogeneous CTMC; fixed noninvasive detector with independent conditional endpoint noise; valid clock and calibrated moment comparison; full-duration accounting',
            'sampling_contract':'fixed quotas; bounded observations with declared conditional means; valid external calibration region'}


def certify_sums(lags,coefficients,records,score_sums,durations,calibration_vertices,**kwargs):
    sums=list(map(exact,score_sums));n=list(records)
    if len(sums)!=len(n) or any(type(N) is not int or N<=0 for N in n):raise ValueError('matching positive quotas required')
    return certify_means(lags,coefficients,n,[s/N for s,N in zip(sums,n)],durations,calibration_vertices,**kwargs)


def conditional_coin_variance(means,second_moments,coefficients,records):
    """Population variance comparison for independent records within/across settings."""
    p=list(map(exact,means));r=list(map(exact,second_moments));a=list(map(exact,coefficients));n=list(records)
    if not len(p)==len(r)==len(a)==len(n) or not p or any(type(N) is not int or N<=0 for N in n):raise ValueError('matching positive quotas required')
    if any(not 0<=m<=1 or not m*m<=s<=m for m,s in zip(p,r)):raise ValueError('invalid bounded first/second moments')
    raw=sum((v*v*(s-m*m)/N for m,s,v,N in zip(p,r,a,n)),F(0))
    coin=sum((v*v*m*(1-m)/N for m,v,N in zip(p,a,n)),F(0))
    return {'direct_contrast_variance':raw,'coin_contrast_variance':coin,'removed_conditional_coin_variance':coin-raw}


def _jsonify(x):
    if isinstance(x,F):return {'fraction':str(x),'decimal':float(x)}
    if isinstance(x,dict):return {k:_jsonify(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [_jsonify(v) for v in x]
    return x


def make_results():
    summary=quantized_pair_summary([F(1,3),F(-1,2),.125],[F(3,4),F(1,5),-.375],1000)
    example=certify_means([1,2],[1,-1],[10000,10000],[F(69,100),F(59,100)],
                          [F(1),F(2)],[(F(0),F(1,2))],alpha=F(1,40),calibration_failure=F(1,40),
                          mean_errors=[F(1,10**6),F(1,10**6)])
    variance=conditional_coin_variance([F(3,5)],[F(2,5)],[F(1)],[1000])
    return {'evidence':'exact illustrative summaries; no measured physical data','summary':summary,'certificate_example':example,'variance_example':variance}

if __name__=='__main__':
    out=Path(__file__).resolve().parents[1]/'results'/'analog.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(_jsonify(make_results()),indent=2)+'\n');print(out)

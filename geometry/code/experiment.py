"""Frozen, same-triple diffusion-geometry benchmark (numerical simulation).

The statistical theorem applies to ideal independent stationary records.  This
program samples that model numerically; it is not laboratory validation.  Exact
integer scores include a calibrated reading-rounding allowance, and exact
rational aggregate confidence arithmetic is inherited from finite_resolution.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve()
PHASE = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import finite_resolution as fr
try:
    from constructive_access.attainability.spectral_dynamics import from_gap
except ModuleNotFoundError:
    legacy = PHASE.parent / 'physical_attainability' / 'release' / 'src'
    sys.path.insert(0, str(legacy))
    from constructive_access.attainability.spectral_dynamics import from_gap

N = 200000
REPLICATES = 400
MASTER_SEED = 202610130731
D = 1 << 20
SCORE_D = 256
BASELINE_D = 1 << 24
H = F(1, 25)
S = F(24, 25)
ALPHA = F(1, 20)
BOUNDS = (F(19, 6), F(19, 6), F(10), F(10))
Q = F(4, 5)
AMIN = F(9, 25)
DELTA = F(1, D)
SCORE_ERROR = 4*(2/H+Q)*DELTA/(H*AMIN) + F(1, SCORE_D)
BASELINE_ERROR = 2*(1+S)**2*DELTA + F(1, BASELINE_D)
SCENARIOS = (('driven', 3), ('weak_driven', 1), ('equilibrium', 0))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(obj):
    if isinstance(obj, F):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, default=encode)+'\n')


def protocol():
    return fr.make_protocol(h=H, q_abs_max=Q, envelope_floor=AMIN,
                            generator_bounds=BOUNDS, global_gamma_upper=1)


def frozen_design():
    p = protocol()
    return dict(version=1, master_seed=MASTER_SEED, replicates=REPLICATES,
        independent_stationary_triples=N, gamma=1, scenarios=dict(SCENARIOS),
        lag=H, alpha_per_method=ALPHA, primary='weighted_empirical_bernstein',
        secondary=('endpoint_empirical_bernstein', 'weighted_hoeffding'),
        common_generator_bounds=BOUNDS, declared_frequency_cap=3,
        rule='z=j/2^20; retain 5|j0|<=4*2^20; q=-z0; a=1-max(|z0|-2^-20,0)^2',
        rounding_denominator=D, reading_error_upper=DELTA,
        score_denominator=SCORE_D, score_uniform_error=SCORE_ERROR,
        endpoint_parameter=S, endpoint_denominator=BASELINE_D,
        endpoint_uniform_error=BASELINE_ERROR, protocol=p,
        resources_per_dataset=dict(detector_readings=3*N,
            independent_stationary_preparations=N, modeled_evolution_time=2*H*N),
        uncounted_resources='Stationary preparation and reset duration, actual elapsed laboratory time, calibrated geometry and generator caps, exposure, clock and energy are not measured or assigned zero.',
        pilot_disclosure='Root investigated independent preliminary circle simulations at h=.035 and .05, n=200000, driven omega=3, seed82653. This informed the h=.04 quota/design choice. The frozen bank uses a different master seed and no pilot records.',
        no_adaptive_selection='No timing, weights, quota, frequency-dependent generator caps or primary statistic may be chosen from this frozen bank. Methods receive individual 95% coverage, not a simultaneous max certificate.',
        sensor_contract='The original bounded noiseless smooth f=cos(phi) readings are approximated within delta by stored j/2^20. The frozen rounded central rule is itself a deterministic state function. The envelope bounds Gamma of the original smooth f. This is numerical rounding, not an assumption of unbiased rounded sensor noise.',
        simulation_scope='PCG64 and NumPy normal/cosine routines are numerical approximations to ideal independent uniform/Brownian sampling. Exact inferential arithmetic does not prove exact sampling from real-valued probability laws. Nearest-grid rounding leaves a half-grid allowance for floating evaluation; a platform violating the declared reading error invalidates the numerical simulation contract.',
        records='All 1200 dataset sufficient statistics, deterministic per-dataset seeds and one complete raw triple exemplar per scenario are retained.',
        source_hashes={'experiment.py':sha(HERE), 'finite_resolution.py':sha(fr.__file__),
                       'spectral_dynamics.py':sha(importlib.import_module(from_gap.__module__).__file__)})


def freeze(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite a frozen contract')
    design = frozen_design()
    design['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(path, design)
    return design


def check_contract(path):
    saved = json.loads(Path(path).read_text())
    expected = json.loads(json.dumps(frozen_design(), default=encode))
    if {k:v for k,v in saved.items() if k != 'frozen_utc'} != expected:
        raise ValueError('Design or source differs from the frozen contract')
    return saved


def floor_scaled(numerator, denominator, scale):
    """Exact int64 floor(scale*numerator/denominator), avoiding its large product."""
    quotient, remainder = np.divmod(numerator, denominator)
    if np.any(denominator <= 0) or np.any(remainder > np.iinfo(np.int64).max//scale):
        raise OverflowError('Invalid denominator or scaled remainder overflow')
    return quotient*scale + (remainder*scale)//denominator


def quantized_scores(readings):
    """Exact rational score flooring from integer bounded readings, no float math."""
    j = np.asarray(readings)
    if j.dtype != np.int64 or j.ndim != 2 or j.shape[1] != 3 or np.any(np.abs(j)>D):
        raise ValueError('Expected n by 3 bounded int64 dyadic readings')
    past, center, future = j.T
    left = 25*past-24*center
    right = 25*future-24*center
    retained = 5*np.abs(center) <= 4*D
    code = np.zeros(len(j), dtype=np.int64)
    c = center[retained]
    cabs = np.abs(c)
    small = np.maximum(cabs-1, 0)
    numerator = -15*left[retained]*right[retained] - 6*cabs*D - 19*D*D
    denominator = 15*(D*D-small*small)
    code[retained] = floor_scaled(numerator, denominator, SCORE_D)
    # Cancel 2^24 against D^2=2^40 before multiplying: exact and safe.
    baseline = (-left*right)//(625*(D*D//BASELINE_D))
    return code, baseline, retained


def totals(codes):
    """Exact aggregates with a checked int64 sum-of-squares bound."""
    codes = np.asarray(codes, dtype=np.int64)
    largest = int(np.max(np.abs(codes), initial=0))
    limit=np.iinfo(np.int64).max
    if largest*largest > limit:
        return sum(map(int,codes)), sum(int(k)**2 for k in codes)
    square_codes=codes*codes
    if len(codes)*largest*largest > limit:
        chunk=max(1,limit//(largest*largest))
        blocks=np.add.reduceat(square_codes,np.arange(0,len(codes),chunk))
        square_total=sum(map(int,blocks))
    else:
        square_total=int(square_codes.sum(dtype=np.int64))
    if len(codes)*largest > limit:
        total=sum(map(int,codes))
    else:
        total=int(codes.sum(dtype=np.int64))
    return total,square_total


def baseline_certificate(n, total, squares):
    """Maurer--Pontil lower bound on the SAME-triple endpoint defect, then edition 7."""
    mean = F(total, BASELINE_D*n)
    second = F(squares, BASELINE_D**2*n)
    lower = -(1+S)**2 - BASELINE_ERROR
    upper = 1 + BASELINE_ERROR
    variance = F(n,n-1)*(second-mean*mean)
    if variance < 0 or not mean*mean <= second <= (lower+upper)*mean-lower*upper:
        raise ValueError('Inconsistent exact endpoint moments')
    log = fr.log_interval(2/ALPHA)[1]
    radius = fr.sqrt_upper(2*variance*log/n) + 7*(upper-lower)*log/(3*(n-1))
    signed = mean-radius-BASELINE_ERROR
    defect = max(F(0),signed)
    bridge = from_gap(defect/2,lag_unit=H,range_bound=2,terms=24)
    return dict(score_mean=mean, empirical_variance=variance,
                sampling_radius_upper=radius, signed_defect_lower=signed,
                defect_lower=defect,
                entropy_production_rate_lower=bridge['entropy_production_rate_lower'])


def infer(n, weighted_total, weighted_squares, endpoint_total, endpoint_squares):
    p = protocol()
    common = dict(records=n, score_total=F(weighted_total,SCORE_D),
                  score_square_total=F(weighted_squares,SCORE_D*SCORE_D),
                  protocol=p,alpha=ALPHA,quantization_radius=SCORE_ERROR)
    primary = fr.certify_totals(**common,concentration='empirical_bernstein')
    hoeffding = fr.certify_totals(**common,concentration='hoeffding')
    endpoint = baseline_certificate(n,endpoint_total,endpoint_squares)
    return primary, endpoint, hoeffding


def simulate(scenario_index, replicate, n=N):
    """Conditional independent stationary past/future increments on a circle."""
    name, omega = SCENARIOS[scenario_index]
    seed = np.random.SeedSequence([MASTER_SEED,scenario_index,replicate])
    rng = np.random.Generator(np.random.PCG64(seed))
    center = rng.uniform(0,2*np.pi,n)
    noise = rng.standard_normal((n,2))
    step = np.sqrt(2*float(H))*noise
    readings = np.column_stack((np.cos(center-omega*float(H)+step[:,0]),
                                np.cos(center),
                                np.cos(center+omega*float(H)+step[:,1])))
    rounded = np.rint(readings*D).astype(np.int64)
    return rounded


def deterministic_npz(path, arrays):
    """NPZ byte reproducibility without wall-clock ZIP timestamps."""
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for key in sorted(arrays):
            buffer=io.BytesIO(); np.lib.format.write_array(buffer,np.asarray(arrays[key]),allow_pickle=False)
            item=zipfile.ZipInfo(key+'.npy',date_time=(1980,1,1,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=0o600 << 16
            archive.writestr(item,buffer.getvalue(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)


def run(contract_path, output_path, raw_path):
    contract = check_contract(contract_path)
    datasets=[]; summaries={}; raw={}
    for idx,(name,omega) in enumerate(SCENARIOS):
        rates=[[],[],[]]; violations=[0,0,0]
        for rep in range(REPLICATES):
            j=simulate(idx,rep)
            w,b,retain=quantized_scores(j)
            wt,ws=totals(w);bt,bs=totals(b)
            certificates=infer(N,wt,ws,bt,bs)
            row=dict(scenario=name,replicate=rep,seed=[MASTER_SEED,idx,rep],
                     records=N,retained=int(retain.sum()),weighted_total=wt,
                     weighted_square_total=ws,endpoint_total=bt,endpoint_square_total=bs)
            for method,cert,values,k in zip(('weighted','endpoint','hoeffding'),certificates,rates,range(3)):
                rate=cert['entropy_production_rate_lower'];values.append(float(rate))
                violations[k]+=int(rate>omega*omega)
                row[method+'_rate_lower']=float(rate)
                row[method+'_sampling_radius_upper']=float(cert['sampling_radius_upper'])
            row['weighted_mean']=float(certificates[0]['score_mean'])
            row['weighted_variance']=float(certificates[0]['empirical_variance'])
            datasets.append(row)
            if rep == 0:
                raw[name+'_readings']=j
                raw[name+'_weighted_codes']=w
                raw[name+'_endpoint_codes']=b
            if (rep+1)%100 == 0:
                print(f'{name}: {rep+1}/{REPLICATES}',flush=True)
        summaries[name]=dict(true_entropy_rate=omega*omega,
            methods={method:dict(positive_certificates=int(np.count_nonzero(v)),
                mean_rate_lower=float(np.mean(v)),minimum_rate_lower=float(np.min(v)),
                maximum_rate_lower=float(np.max(v)),over_truth=bad)
                for method,v,bad in zip(('weighted','endpoint','hoeffding'),rates,violations)})
    deterministic_npz(raw_path,raw)
    report=dict(contract_sha256=sha(contract_path),raw_sha256=sha(raw_path),
                version=1,scenario_summary=summaries,datasets=datasets,
                source_hashes=contract['source_hashes'],
                resources_per_dataset=contract['resources_per_dataset'],
                environment=dict(python=platform.python_version(),numpy=np.__version__),
                interpretation='Finite-sample certificates under the declared physical, calibration and iid contracts; a numerical simulation, not an apparatus validation or universal optimality claim.')
    write_json(output_path,report)
    print(json.dumps(summaries,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--contract',type=Path,default=PHASE/'contracts/experiment.json')
    parser.add_argument('--output',type=Path,default=PHASE/'results/experiment.json')
    parser.add_argument('--raw',type=Path,default=PHASE/'results/experiment_raw.npz')
    args=parser.parse_args()
    if args.freeze:
        freeze(args.contract)
        print('Frozen:',args.contract,sha(args.contract))
    if args.run:
        run(args.contract,args.output,args.raw)
    if not args.freeze and not args.run:
        print(json.dumps(frozen_design(),indent=2,default=encode))

if __name__=='__main__':main()

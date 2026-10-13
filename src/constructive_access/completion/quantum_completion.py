"""Rationally certified driven-qubit violation of a classical pulse sign law.

Physical model: H=g X; Z-basis jump channels (3*kappa,kappa) and
(kappa,3*kappa); the first channel is retained by a factor r.  Optional
dephasing is d*(Z rho Z-rho). The native stationary state is I/2.

The exact phase-boundary theorem and scope are in research/quantum_completion.txt.
No thermodynamic interpretation of these driven engineered channels is assumed.
"""
from fractions import Fraction as F
from math import factorial, isqrt
from pathlib import Path
import json


def rational(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,F)):
        raise TypeError("exact int, rational string, or Fraction required")
    return F(x)


def mul2(a,b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2))
                       for j in range(2)) for i in range(2))


def integral_certificate(gamma,g,dephasing,t,order=80):
    """Enclose integral_0^t (exp(A*u))[0,0] du using exact Taylor arithmetic.

    A=[[-4*gamma,2*g],[-2*g,-2*gamma-2*dephasing]].
    Tail <= t*s^(N+1)/(N+2)!/(1-s/(N+3)), s=t*||A||_infinity.
    """
    gamma,g,dephasing,t=map(rational,(gamma,g,dephasing,t))
    if gamma<=0 or g<0 or dephasing<0 or t<0:
        raise ValueError("gamma>0 and g,dephasing,t>=0 required")
    if isinstance(order,bool) or not isinstance(order,int) or order<0:
        raise ValueError("nonnegative integer Taylor order required")
    a=((-4*gamma,2*g),(-2*g,-2*gamma-2*dephasing))
    norm=max(sum(abs(x) for x in row) for row in a)
    s=t*norm
    if s>=order+3:
        raise ValueError("Taylor order too small for the geometric tail bound")
    power=((F(1),F(0)),(F(0),F(1)))
    coefficient=t
    center=F(0)
    for k in range(order+1):
        center+=coefficient*power[0][0]
        power=mul2(power,a)
        coefficient*=t/F(k+2)
    radius=t*s**(order+1)/factorial(order+2)/(1-s/F(order+3))
    return {"gamma":gamma,"g":g,"dephasing":dephasing,"time":t,
            "order":order,"norm_infinity":norm,"center":center,
            "radius":radius,"lower":center-radius,"upper":center+radius}


def pulse_certificate(kappa,g,r,t,dephasing=0,order=80):
    kappa,g,r,t,dephasing=map(rational,(kappa,g,r,t,dephasing))
    if kappa<=0 or not 0<=r<1:
        raise ValueError("kappa>0 and 0<=r<1 required")
    integral=integral_certificate((1+r)*kappa,g,dephasing,t,order)
    factor=-(1-r)*kappa
    low,high=factor*integral["upper"],factor*integral["lower"]
    return {"kappa":kappa,"g":g,"retention":r,"time":t,
            "dephasing":dephasing,
            "native_channel_current":kappa,
            "native_stationary_population_B":F(1,2),
            "pulse_up_rate":(3*r+1)*kappa,
            "pulse_down_rate":(r+3)*kappa,
            "all_channel_rates_strictly_positive":r>0,
            "initial_response_derivative":factor,
            "response_lower":low,"response_upper":high,
            "response_sign":1 if low>0 else -1 if high<0 else 0,
            "population_B_lower":F(1,2)+low,
            "population_B_upper":F(1,2)+high,
            "integral_certificate":integral}


def atan_interval(x,terms=40):
    """Alternating-series enclosure, exact 0<=x<=1."""
    x=rational(x)
    if not 0<=x<=1 or not isinstance(terms,int) or terms<1:
        raise ValueError("atan requires 0<=x<=1 and positive term count")
    partial=sum((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms))
    other=partial+(-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(partial,other),max(partial,other)


def pi_interval():
    # Machin's identity pi=16 atan(1/5)-4 atan(1/239).
    al,au=atan_interval(F(1,5),40)
    bl,bu=atan_interval(F(1,239),12)
    return 16*al-4*bu,16*au-4*bl


def log_interval(x,terms=90):
    """Positive-series enclosure for log(x), x>=1."""
    x=rational(x)
    if x<1 or not isinstance(terms,int) or terms<1:
        raise ValueError("log enclosure requires x>=1 and positive terms")
    z=(x-1)/(x+1)
    partial=2*sum(z**(2*k+1)/F(2*k+1) for k in range(terms))
    tail=2*z**(2*terms+1)/F(2*terms+1)/(1-z*z)
    return partial,partial+tail


def sqrt_interval(x,decimal_places=24):
    x=rational(x)
    if x<0 or not isinstance(decimal_places,int) or decimal_places<1:
        raise ValueError("nonnegative root and positive decimal places required")
    scale=10**decimal_places
    floor=isqrt((x.numerator*scale*scale)//x.denominator)
    low=F(floor,scale)
    if low*low==x:
        return low,low
    return low,F(floor+1,scale)


def phase_function_interval(q):
    """Exact interval for log q - 3*(pi+atan sqrt(4q^2-1))/sqrt(...)."""
    q=rational(q)
    if q<=1:
        raise ValueError("phase bracket evaluator requires q>1")
    wl,wu=sqrt_interval(4*q*q-1)
    pl,pu=pi_interval()
    # pi+atan(w)=3*pi/2-atan(1/w), w>1.
    al,_=atan_interval(1/wu,16)
    _,au=atan_interval(1/wl,16)
    nl=F(3,2)*pl-au
    nu=F(3,2)*pu-al
    ll,lu=log_interval(q)
    return ll-3*nu/wl,lu-3*nl/wu


def outward_decimal_interval(low,high,places=18):
    scale=10**places
    lower=(low.numerator*scale)//low.denominator
    upper=-((-high.numerator*scale)//high.denominator)
    return F(lower,scale),F(upper,scale)


def phase_boundary_certificate():
    lower=F("4.57066453253")
    upper=F("4.57066453254")
    fl=phase_function_interval(lower)
    fu=phase_function_interval(upper)
    if not fl[1]<0<fu[0]:
        raise ArithmeticError("phase boundary bracket did not certify")
    return {"q_lower":lower,"q_upper":upper,
            "F_at_lower":outward_decimal_interval(*fl,places=20),
            "F_at_upper":outward_decimal_interval(*fu,places=20),
            "equation":"log(q)=3*(pi+atan(sqrt(4*q*q-1)))/sqrt(4*q*q-1)",
            "uniqueness":"F strictly increasing for q>1; F(1)<0 and F(q)->infinity",
            "certification":"exact rational Machin, alternating atan, positive log and integer sqrt enclosures"}


def timing_menu(kappa=F(1,10),g=1,r=F(1,10)):
    """Choose the largest positive response among 16 quarter-unit pulses.

    This is an exact finite-menu comparison, not a global physical energy optimum.
    The return also displays response squared / pulse duration, a clearly
    declared acquisition-time proxy rather than an apparatus power measurement.
    """
    rows=[]
    for j in range(1,17):
        c=pulse_certificate(kappa,g,r,F(j,4))
        lo,hi=c["response_lower"],c["response_upper"]
        rows.append({"time":F(j,4),"lower":lo,"upper":hi,
                     "positive_signal_squared_per_time_lower":max(F(0),lo)**2/F(j,4),
                     "positive_signal_squared_per_time_upper":max(F(0),hi)**2/F(j,4)})
    winner=max(range(len(rows)),key=lambda j:rows[j]["lower"])
    if any(rows[winner]["lower"]<=row["upper"] for j,row in enumerate(rows) if j!=winner):
        raise ArithmeticError("menu response optimum not separated by certificates")
    efficiency=max(range(len(rows)),key=lambda j:rows[j]["positive_signal_squared_per_time_lower"])
    if any(rows[efficiency]["positive_signal_squared_per_time_lower"]<=row["positive_signal_squared_per_time_upper"]
           for j,row in enumerate(rows) if j!=efficiency):
        raise ArithmeticError("menu time proxy optimum not separated by certificates")
    return {"rows":rows,"maximum_signal_time":rows[winner]["time"],
            "maximum_signal_squared_per_time":rows[efficiency]["time"],
            "scope":"16 fixed pulses; durations only; preparation/readout/control energy additional"}


def general_rates_certificate(total_up,selected_up,selected_down,g,r,t,dephasing=0,order=80):
    """Unital native qubit with arbitrary positive channel decomposition.

    total_up=total_down=K; selected channel has rates (a,b), with a,b<=K.
    Remaining aggregate rates K-a,K-b must be nonnegative. At least one current
    is required for a sign test. The mathematical interval also covers J=0.
    """
    K,a,b,g,r,t,d=map(rational,(total_up,selected_up,selected_down,g,r,t,dephasing))
    if K<=0 or not 0<=a<=K or not 0<=b<=K or not 0<=r<1:
        raise ValueError("invalid unital channel inventory or retention")
    total=2*K-(1-r)*(a+b)
    if total<=0:
        raise ValueError("post-pulse total jump rate must remain positive")
    integral=integral_certificate(total/4,g,d,t,order)
    current=(a-b)/2
    factor=-(1-r)*current
    ends=(factor*integral["lower"],factor*integral["upper"])
    return {"native_current":current,"gamma":total/4,
            "response_lower":min(ends),"response_upper":max(ends),
            "initial_derivative":factor,"integral_certificate":integral}


def jsonable(x):
    if isinstance(x,F):
        return str(x)
    if isinstance(x,dict):
        return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [jsonable(y) for y in x]
    return x


def compact_pulse(c):
    # Exact outward rounding prevents giant fractions in the public result file.
    low,high=outward_decimal_interval(c["response_lower"],c["response_upper"],20)
    return {k:v for k,v in c.items() if k not in {
        "integral_certificate","response_lower","response_upper",
        "population_B_lower","population_B_upper"}} | {
        "response_lower":low,"response_upper":high,
        "population_B_lower":F(1,2)+low,"population_B_upper":F(1,2)+high,
        "taylor_order":c["integral_certificate"]["order"],
        "tail_below_1e_minus_30":c["integral_certificate"]["radius"]<F(1,10**30)}


def main():
    small=pulse_certificate(F(1,100),1,F(1,2),2)
    first=pulse_certificate(F(1,10),1,F(1,10),F(1,2))
    late=pulse_certificate(F(1,10),1,F(1,10),F(5,2))
    selected=pulse_certificate(F(1,10),1,F(1,10),F(9,4))
    incoherent=pulse_certificate(F(1,10),0,F(1,10),F(5,2))
    dephased=[compact_pulse(pulse_certificate(F(1,10),1,F(1,10),F(5,2),dephasing=F(j,10)))
               for j in range(6)]
    menu=timing_menu()
    for row in menu["rows"]:
        row["lower"],row["upper"]=outward_decimal_interval(row["lower"],row["upper"],20)
        keys=("positive_signal_squared_per_time_lower","positive_signal_squared_per_time_upper")
        row[keys[0]],row[keys[1]]=outward_decimal_interval(row[keys[0]],row[keys[1]],20)
    gain=F(9,10)
    short_gap=F(3,100)
    late_gap=F(13,1000)
    if not -gain*first["response_upper"]>short_gap or not gain*selected["response_lower"]>late_gap:
        raise ArithmeticError("calibrated conservative pulse gaps failed")
    conservative={"readout_gain":gain,"short_time":F(1,2),"late_time":F(9,4),
                  "short_absolute_response_strict_lower":short_gap,
                  "late_response_strict_lower":late_gap,
                  "checked_with_exact_taylor":True,
                  "scope":"common affine detector gain; any offset/preparation/drift corrections require separate budget"}
    results={"status":"analytical and exact rational certificates; no empirical measurement",
             "physical_scope":"finite driven GKSL qubit with engineered Z-basis jump channels",
             "interpretation":"violates stationary classical single-edge selective-suppression sign constraint; not device-independent quantumness",
             "phase_boundary":phase_boundary_certificate(),
             "small_rate_case":compact_pulse(small),
             "short_pulse":compact_pulse(first),"late_pulse":compact_pulse(late),
             "selected_late_pulse":compact_pulse(selected),
             "conservative_readout_gaps":conservative,
             "incoherent_control":compact_pulse(incoherent),
             "dephasing_series":dephased,"timing_menu":menu}
    path=Path(__file__).resolve().parents[1]/"results"/"quantum_completion.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(results),indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(path),"short_response":float(first["response_lower"]),
                      "late_response":float(late["response_lower"]),
                      "threshold_bracket":[float(results["phase_boundary"]["q_lower"]),
                                           float(results["phase_boundary"]["q_upper"])],
                      "menu_signal_time":str(menu["maximum_signal_time"]),
                      "menu_time_proxy":str(menu["maximum_signal_squared_per_time"])}))


if __name__=="__main__":
    main()

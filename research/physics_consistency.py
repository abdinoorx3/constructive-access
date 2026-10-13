"""Independent exact transcription checks for the two physics chapters.

Run from any directory: python research/physics_consistency.py.
No archived producing module is imported. These checks support, but do not
replace, the displayed general proofs or independent physical validation.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import sympy as s


def zero(matrix):
    return all(s.expand(v) == 0 for v in matrix)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def classical():
    q = s.Matrix([[-2, 1, 0, 1], [1, -3, 2, 0],
                  [0, 1, -4, 3], [4, 0, 1, -5]])
    p = s.Matrix([[41, 18, 13, 16]]) / 88
    d = s.diag(1, 0, 1, 0)
    one = s.ones(4, 1)
    check(zero(p*q), 'four-state stationarity')
    coeff = (p*d*q*d*q**2*d*one-p*d*q**2*d*q*d*one)[0]/2
    check(coeff == -s.Rational(23, 44), 'binary aggregate cubic coefficient')

    # General symbolic bias in the exact triangle, nonconstant two-state gate.
    f, eta = s.symbols('f eta', real=True)
    qt = s.Matrix([[-2, 1+f, 1-f], [1-f, -2, 1+f], [1+f, 1-f, -2]])
    l = s.Matrix([[-1, 1, 0], [1, -1, 0], [0, 0, 0]])
    gate = s.Matrix([[-2, 2], [3, -3]])
    u = s.diag(0, eta)
    g = s.kronecker_product(gate, s.eye(3))+s.kronecker_product(s.eye(2), qt)+s.kronecker_product(u, l)
    pi = s.Matrix([[s.Rational(3,5), s.Rational(2,5)]])
    joint = s.kronecker_product(pi, s.ones(1,3)/3)
    check(zero(joint*g), 'gate product stationarity')
    detector = s.diag(1,0,0,1,0,0)
    g2, g3 = g*g, g*g*g
    coeff13 = s.expand((joint*detector*g*detector*g3*detector*s.ones(6,1)-joint*detector*g3*detector*g*detector*s.ones(6,1))[0]/6)
    # p=q=1, J=2f/3, D=6eta^2/5.
    check(s.expand(coeff13-s.Rational(2,15)*f*eta**2) == 0, 'hidden-gate fourth coefficient')
    check(s.expand((s.ones(1,3)/3)[0]*(l*qt-qt*l)[0,0]+2*f/3) == 0, 'switched current sign')
    return {'aggregate_coefficient': str(coeff), 'gate_coefficient_ts3': str(coeff13)}


def quantum():
    i = s.I
    eye = s.eye(2)
    pauli = {'I': eye, 'X': s.Matrix([[0,1],[1,0]]),
             'Y': s.Matrix([[0,-i],[i,0]]), 'Z': s.diag(1,-1)}
    def word(w):
        return s.kronecker_product(*(pauli[c] for c in w))
    def at(a, site):
        return s.kronecker_product(*(a if k == site else eye for k in range(3)))
    sm = s.Matrix([[0,1],[0,0]])
    sp = sm.T
    delta, g, gamma = s.symbols('delta g gamma', real=True)
    h = g*(at(sp,0)*at(sm,1)+at(sm,0)*at(sp,1)+at(sp,1)*at(sm,2)+at(sm,1)*at(sp,2))
    jumps = []
    for j, occupation in [(0,s.Rational(1,2)+delta),(2,s.Rational(1,2)-delta)]:
        jumps += [(gamma*(1-occupation),at(sm,j)),(gamma*occupation,at(sp,j))]
    def dissipate(rho, a):
        return a*rho*a.conjugate().T-(a.conjugate().T*a*rho+rho*a.conjugate().T*a)/2
    def liouville(rho):
        return -i*(h*rho-rho*h)+sum((weight*dissipate(rho,a) for weight,a in jumps),s.zeros(8))
    def adjoint(o):
        out=i*(h*o-o*h)
        for weight,a in jumps:
            out += weight*(a.conjugate().T*o*a-(a.conjugate().T*a*o+o*a.conjugate().T*a)/2)
        return out
    basis=[word('III'),word('IZI'),-word('IXY')+word('IYX')+word('XYI')-word('YXI'),
           word('IIZ')+word('ZII')-word('XZX')-word('YZY'),
           word('IZZ')+word('ZZI')-word('XIX')-word('YIY')]
    c=s.Matrix([[0,0,0,0,0],[0,0,-4*g,0,0],[0,g,-gamma/2,-2*g,0],
                [0,0,2*g,-gamma,0],[0,0,0,0,-gamma]])
    for j,b in enumerate(basis):
        check(zero(adjoint(b)-sum((c[k,j]*basis[k] for k in range(5)),s.zeros(8))),f'XX closure {j}')
    alpha,beta=s.symbols('alpha beta',real=True)
    a,b,cc=alpha**2+beta**2,2*alpha*beta,alpha**2-beta**2
    m=s.Matrix([[a,b,0,0,0],[b,a,0,0,0],[0,0,cc,0,0],[0,0,0,a,b],[0,0,0,b,a]])
    k=alpha*s.eye(8)+beta*word('IZI')
    for j,bj in enumerate(basis):
        check(zero(k*bj*k-sum((m[n,j]*basis[n] for n in range(5)),s.zeros(8))),f'XX Kraus closure {j}')
    rho=(word('III')+2*delta/5*(word('IIZ')-word('ZII'))
        +4*delta/5*(-word('IXY')+word('IYX')-word('XYI')+word('YXI'))
        -16*delta**2/25*(word('IZZ')+word('XIX')+word('YIY')+word('ZZI'))
        -4*delta**2/25*word('ZIZ')
        +8*delta**2/25*(-word('XYZ')+word('YXZ')+word('ZXY')-word('ZYX')))/8
    check(s.trace(rho)==1,'XX state trace')
    check(zero(liouville(rho).subs({g:1,gamma:1})),'XX stationary state')
    px=(s.eye(8)+word('IXI'))/2
    pz=(s.eye(8)+word('IZI'))/2
    out=[]
    for name,read,initial in [('XX',px,px*rho*px),('ZXZ',pz,word('IXI')*pz*rho*pz*word('IXI'))]:
        coeffs=[]
        state=initial
        for n in range(3):
            coeffs.append(s.expand(s.trace(read*state).subs({g:1,gamma:1})/s.factorial(n)))
            state=liouville(state)
        expected=[s.Rational(1,2),0,-s.Rational(1,4)+4*delta**2/25] if name=='XX' else [0,0,s.Rational(1,2)-16*delta**2/25]
        check(all(s.expand(x-y)==0 for x,y in zip(coeffs,expected)),name+' coefficients')
        out.append({'protocol':name,'coefficients':list(map(str,coeffs))})
    return {'closure_columns':10,'stationary_state_checked':True,'protocols':out}


def chemistry_and_qubit():
    intervals=[(F(399,100),F(401,100)),(F(399,100),F(401,100)),
               (F(249,100),F(251,100)),(F(349,100),F(351,100)),(F(49,100),F(51,100))]
    vals=[]
    for k0,l0,k1,l1,r in product(*intervals):
        rates=[(k0-k1)/(1-r),(l0-l1)/(1-r),(k1-r*k0)/(1-r),(l1-r*l0)/(1-r)]
        check(min(rates)>0,'chemical rate positivity')
        vals.append((k0*l1-l0*k1)/((1-r)*(k0+l0)))
    check((min(vals),max(vals))==(F(193,204),F(207,196)),'native current interval')
    a,b,lam=F(399,401),F(40,401),F(19,20)
    ct,st=a*a-b*b,2*a*b
    cm,sm=F(1),F(0)
    plans=[]
    for m in range(1,81):
        cm,sm=cm*ct-sm*st,sm*ct+cm*st
        nominal=lam**m*abs(sm)/2
        gap=nominal-2*(F(1,1000)+m*F(1,10000))
        if gap>0:
            bound=F(5)/(gap*gap)
            records=-(-bound.numerator//bound.denominator)
            plans.append((m,records,m*records,nominal))
    check(min(plans,key=lambda p:p[2])[:3]==(4,60,240),'quantum least modeled calls')
    check(max(plans,key=lambda p:p[3])[:3]==(7,44,308),'quantum contrast maximum')
    check(plans[0][:3]==(1,590,590),'quantum single call baseline')
    return {'chemical_vertices':len(vals),'current_interval':[str(min(vals)),str(max(vals))],
            'qubit_menu_depths':80,'best_calls':[4,60,240]}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    results={'status':'passed','classical':classical(),'quantum':quantum(),
             'application_numbers':chemistry_and_qubit(),'empirical_data':False}
    results['chapter_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [root/'manuscript/chapters/continuous_thermodynamics.tex',root/'manuscript/chapters/applications.tex']}
    destination=root/'results/physics_consistency.json'
    destination.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

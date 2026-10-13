"""Run installed exact completion APIs; no measured data or network required."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
from constructive_access.completion import quantum_completion as quantum
from constructive_access.completion import pulse_test
from constructive_access.completion.graph_access import Graph, Edge
from constructive_access.completion.robust_completion import floor_energy_bound, verify_floor_energy_bound


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/completion_tutorial.json'))
    args=parser.parse_args()
    phase=quantum.phase_boundary_certificate()
    example=pulse_test.example()
    graph=Graph(('s','a','t'),(Edge('target','s','t'),Edge('single','s','a'),
                              Edge('pair1','a','t'),Edge('pair2','a','t')))
    q={'single':F(4),'pair1':F(3,2),'pair2':F(3,2)}
    floors={'single':F(0),'pair1':F(1,25),'pair2':F(1,25)}
    bracket=floor_energy_bound(graph,'target',q,floors,F(1,10),F(40))
    checked=verify_floor_energy_bound(graph,'target',q,floors,F(1,10),bracket)
    output={'evidence':'Exact model calculations; no empirical data',
            'quantum_threshold':phase,'pulse_test':example,
            'energy_contract':'epsilon=delta+sqrt(q^3/E); hypothetical calibrated units',
            'energy_bracket':bracket,'energy_check':checked}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(output,default=str,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'records_per_arm':example['design']['records_per_arm'],
                      'energy_check':checked}))


if __name__=='__main__':main()

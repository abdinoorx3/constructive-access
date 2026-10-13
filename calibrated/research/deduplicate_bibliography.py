"""Idempotent citation normalization for the sixth cumulative release.

Prior bibliography files and prior chapters are never modified. Only explicit,
audited aliases are applied to new chapters and the owned phase manuscripts.
Run after root recopies phase chapters/bibliography into release/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ALIASES = {
    "ciBuchholz1994": "ac_buchholz",  # Same DOI10.2307/3215235.
    "vlHoeffding1963": "caHoeffding1963",  # Existing canonical cumulative key.
}
ITEM = re.compile(r"\\bibitem\{([^}]+)\}")
CITE = re.compile(r"(\\cite\w*(?:\[[^\]]*\])?\{)([^}]*)(\})")


def blocks(text):
    matches = list(ITEM.finditer(text))
    return [(m.group(1), text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)])
            for i,m in enumerate(matches)]


def remove_alias_entries(text):
    matches=list(ITEM.finditer(text))
    prefix=text[:matches[0].start()] if matches else text
    return prefix+"".join(block for key,block in blocks(text) if key not in ALIASES)


def replace_cites(text):
    def repl(match):
        keys=[ALIASES.get(k.strip(),k.strip()) for k in match.group(2).split(',')]
        return match.group(1)+','.join(dict.fromkeys(keys))+match.group(3)
    return CITE.sub(repl,text)


def run(phase, release):
    phase,release=Path(phase),Path(release)
    manuscript=release/'manuscript'
    old_bibs=[p for p in manuscript.glob('*bib*.tex') if p.name!='calibrated_bibitems.tex']
    old_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in old_bibs}
    old_keys={key for p in old_bibs for key,_ in blocks(p.read_text())}
    for canonical in ALIASES.values():
        if canonical not in old_keys:raise ValueError(f'Missing canonical bibliography key {canonical}')
    changed=[]
    for p in [phase/'manuscript/bibitems.tex',manuscript/'calibrated_bibitems.tex']:
        if p.exists():
            before=p.read_text();after=remove_alias_entries(before)
            if after!=before:p.write_text(after);changed.append(str(p.relative_to(phase)))
    targets=[phase/'manuscript/positioning.tex',phase/'manuscript/empirical.tex']
    targets+=sorted((manuscript/'chapters').glob('ci_*.tex'))
    for p in targets:
        if p.exists():
            before=p.read_text();after=replace_cites(before)
            if after!=before:p.write_text(after);changed.append(str(p.relative_to(phase)))
    entries={}
    duplicate_keys=[]
    doi_keys={}
    for p in sorted(manuscript.glob('*bib*.tex')):
        for key,block in blocks(p.read_text()):
            if key in entries:duplicate_keys.append(key)
            entries[key]=p.name
            dois={x.lower().rstrip('.,;') for x in re.findall(r'10\.\d{4,9}/[^}\s]+',block)}
            for doi in dois:doi_keys.setdefault(doi,set()).add(key)
    unresolved={}
    for p in sorted(manuscript.rglob('*.tex')):
        missing=sorted({key.strip() for m in CITE.finditer(p.read_text()) for key in m.group(2).split(',') if key.strip() not in entries})
        if missing:unresolved[str(p.relative_to(manuscript))]=missing
    preserved=all(hashlib.sha256(p.read_bytes()).hexdigest()==old_hashes[p.name] for p in old_bibs)
    result={'aliases':ALIASES,'changed_files':changed,'prior_bibliographies_unchanged':preserved,
            'bibliography_entries':len(entries),'duplicate_keys':duplicate_keys,
            'duplicate_dois':{k:sorted(v) for k,v in doi_keys.items() if len(v)>1},
            'unresolved_citations':unresolved,
            'note':'DOI identity audited automatically; non-DOI publications manually checked. Prior publications are never silently rewritten.'}
    if not preserved:raise AssertionError('Prior bibliography changed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parents[1]
    p.add_argument('--phase',type=Path,default=default)
    p.add_argument('--release',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();release=args.release or args.phase/'release'
    result=run(args.phase,release)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True))
    if result['duplicate_keys'] or result['duplicate_dois'] or result['unresolved_citations']:
        raise SystemExit(1)


if __name__=='__main__':main()

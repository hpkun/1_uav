"""Read-only resume preflight: refuse conflicts; report gaps without rewriting history."""
import csv
import json
from pathlib import Path


class ResumeLogConflict(ValueError):
    pass


def _step(row,label):
    try:
        value=row['sampled_steps']
        if isinstance(value,bool) or int(value)!=float(value) or int(value)<0: raise ValueError
        return int(value)
    except (KeyError,ValueError,TypeError,OverflowError) as error:
        raise ResumeLogConflict(f'{label}: invalid sampled_steps; original files unchanged') from error


def check_resume_logs(output,checkpoint,payload,read_checkpoint):
    """No repair, truncation, copy, overwrite or RNG operation is performed here."""
    output=Path(output); checkpoint=Path(checkpoint)
    anchor=payload['sampled_steps']; update=payload['updates']; warnings=[]
    for name in ('training_metrics.jsonl','evaluation_history.csv'):
        path=output/name
        if not path.exists() or path.stat().st_size==0:
            warnings.append(f'{name} missing/empty at checkpoint steps {anchor}; historical gap retained')
            continue
        try:
            if name.endswith('.jsonl'):
                rows=[json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
            else:
                with path.open(newline='',encoding='utf-8-sig') as f:
                    reader=csv.DictReader(f)
                    if reader.fieldnames!=['sampled_steps','update','checkpoint_role','success_rate','completion_rate','mean_return']:
                        raise ResumeLogConflict(f'{name}: incompatible CSV header')
                    rows=list(reader)
        except (json.JSONDecodeError,UnicodeError,csv.Error) as error:
            raise ResumeLogConflict(f'{name}: malformed or interrupted log; original files unchanged') from error
        previous=-1; previous_update=-1; seen=set()
        for i,row in enumerate(rows):
            where=f'{name} row {i+1}'; step=_step(row,where)
            role='training' if name.endswith('.jsonl') else row.get('checkpoint_role','current')
            if step<previous or (step,role) in seen:
                raise ResumeLogConflict(f'{where}: duplicate/out-of-order record at {step}; original files unchanged')
            try:
                raw_update=row.get('update',0)
                recorded_update=int(raw_update)
                if isinstance(raw_update,bool) or recorded_update!=float(raw_update) or recorded_update<0:
                    raise ValueError
            except (ValueError,TypeError,OverflowError) as error: raise ResumeLogConflict(f'{where}: invalid update') from error
            if step>anchor or recorded_update>update:
                raise ResumeLogConflict(f'{where}: log is newer ({step} steps/update {recorded_update}) than resume checkpoint ({anchor}/update {update}); use a coherent checkpoint/output pair; original files unchanged')
            if recorded_update<previous_update:
                raise ResumeLogConflict(f'{where}: update counter moved backwards; original files unchanged')
            previous=step; previous_update=recorded_update; seen.add((step,role))
        if not rows or previous<anchor:
            warnings.append(f'{name} ends at {previous if rows else "N/A"}, checkpoint at {anchor}; missing history will not be fabricated')
    config=output/'run_config.json'
    if config.exists():
        try: saved=json.loads(config.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError,UnicodeError) as error: raise ResumeLogConflict('invalid existing run_config.json') from error
        for field,key in (('environment','env_config'),('algorithm','train_config')):
            canonical=json.loads(json.dumps(payload[key]))
            if saved.get(field)!=canonical: raise ResumeLogConflict(f'existing run_config.json {field} conflicts with resume checkpoint')
    summary=output/'run_summary.json'
    if summary.exists():
        try: saved=json.loads(summary.read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError,UnicodeError) as error: raise ResumeLogConflict('invalid existing run_summary.json') from error
        if _step(saved,'run_summary.json')>anchor or saved.get('updates',0)>update:
            raise ResumeLogConflict('run_summary.json is newer than resume checkpoint; original files unchanged')
    # Inspect both the destination and the best that would be copied from source.
    candidates={output/'latest.pt',output/'best.pt',checkpoint.parent/'best.pt'}
    for path in sorted(candidates):
        if path.exists() and path.resolve()!=checkpoint.resolve():
            other=read_checkpoint(path)
            if other['sampled_steps']>anchor or other['updates']>update:
                raise ResumeLogConflict(f'{path} is newer than resume checkpoint; refusing to overwrite or mislabel it')
            if path.name=='best.pt' and other.get('best_score')!=payload.get('best_score'):
                raise ResumeLogConflict(f'{path} best-score provenance differs from resume checkpoint; original files unchanged')
    if payload.get('best_score') is not None and not (output/'best.pt').exists() and not (checkpoint.parent/'best.pt').exists():
        warnings.append('best.pt is missing; its recorded best score cannot be presented as an available model')
    return warnings

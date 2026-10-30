"""Validate completed PAPER_V2 artifacts and export publication tables and figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, precision_recall_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reports.paper_inputs import select_run

PROJECTS = ("COD_CANON_V2", "MP_ALL")
MODELS = ("rf", "lgbm")
DOMAINS = ("N", "O", "M")
METRICS = ("auc", "bacc", "mcc", "auprc", "acc", "precision", "recall", "f1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scores(frame):
    y, p = frame["y"].to_numpy(), frame["y_prob"].to_numpy()
    require(set(np.unique(y)) == {0, 1}, "Expected both binary target classes")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probabilities")
    pred = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p), bacc=balanced_accuracy_score(y, pred),
        mcc=matthews_corrcoef(y, pred), auprc=average_precision_score(y, p),
        acc=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))


class Export:
    def __init__(self, base, output):
        self.base, self.output = base.resolve(), output.resolve()
        self.inputs = {}
        self.tables = {}
        self.copies = []

    def track(self, path):
        path = Path(path).resolve()
        require(path.is_file(), f"Missing input: {path}")
        if path not in self.inputs:
            self.inputs[path] = sha256(path)
        return path

    def read(self, path, **kwargs):
        return pd.read_csv(self.track(path), float_precision="round_trip", **kwargs)

    def read_json(self, path):
        return json.loads(self.track(path).read_text(encoding="utf-8"))

    def table(self, name, rows):
        self.tables[name] = pd.DataFrame(rows)

    def copy(self, source, destination):
        self.copies.append((self.track(source), destination))

    def write(self):
        self.output.mkdir(parents=True, exist_ok=False)
        for name, frame in self.tables.items():
            path = self.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        for source, name in self.copies:
            path = self.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
        pd.DataFrame([{"path": p.relative_to(self.base).as_posix(), "sha256": digest,
                       "bytes": p.stat().st_size} for p, digest in self.inputs.items()]).to_csv(
                           self.output / "input_manifest.csv", index=False)
        outputs = sorted(p for p in self.output.rglob("*") if p.is_file())
        pd.DataFrame([{"path": p.relative_to(self.output).as_posix(), "sha256": sha256(p),
                       "bytes": p.stat().st_size} for p in outputs]).to_csv(
                           self.output / "output_manifest.csv", index=False)


def collect_tables(ex, paper_run, runs_file, cross_version):
    selected = json.loads(Path(runs_file).read_text(encoding="utf-8")) if runs_file else []
    require(len({(r['project'], r['model'], r['domain']) for r in selected}) == len(selected),
            "Duplicate explicit run selections")
    overrides = {mode: {(r['project'], r['model'], r['domain']): r[mode] for r in selected}
                 for mode in ('final', 'cv')}
    counts, final, cv, cv_folds, parameters, run_rows, extra_elements, thresholds = ([] for _ in range(8))
    fold_hashes, selected_final, meta_hashes = {}, {}, {}
    for project in PROJECTS:
        root = ex.base / project
        cfg = SimpleNamespace(PROJECT_NAME=project, RESULTS_DIR=root / 'results')
        reg = ex.read_json(root / 'tuning' / f'best_params_{paper_run}.json')
        entries = [e for e in reg['entries'] if e['key'].get('step') == 'magpie'
                   and e['key'].get('mode') == 'cv']
        require(len(entries) == 6 and len({(e['key']['model'], e['key']['domain']) for e in entries}) == 6,
                f"Incomplete parameter registry: {project}")
        for entry in entries:
            parameters.append(dict(project=project, **entry['key'], **entry['params']))
        for domain in DOMAINS:
            path = root / 'datasets' / f'dataset_splits_folds_{domain}.csv'
            columns = ex.read(path, nrows=0).columns
            frame = ex.read(path, usecols=['id', 'centrosymmetry', 'group_id', 'is_test', 'fold'],
                            dtype={'id': str, 'group_id': str})
            test, train = frame[frame.is_test.eq(1)], frame[frame.is_test.eq(0)]
            require(len(test) + len(train) == len(frame) and not frame.id.duplicated().any()
                    and not frame[['id', 'group_id', 'centrosymmetry']].isna().any().any()
                    and frame.centrosymmetry.isin(['C', 'N']).all()
                    and not (set(test.group_id) & set(train.group_id))
                    and train.groupby('group_id').fold.nunique().le(1).all()
                    and set(train.fold) == set(range(5)) and test.fold.eq(-1).all(),
                    f"Invalid split: {path}")
            n_features = sum(c.startswith('MagpieData') for c in columns)
            require(n_features == 132, f"Expected 132 Magpie features: {path}")
            short = 'COD' if project == 'COD_CANON_V2' else 'MP'
            fold_hashes[f'{short}_{domain}'] = ex.inputs[path.resolve()]
            counts.append(dict(project=project, domain=domain, n=len(frame), n_train=len(train),
                n_test=len(test), groups=frame.group_id.nunique(), test_groups=test.group_id.nunique(),
                C=int(frame.centrosymmetry.eq('C').sum()), N=int(frame.centrosymmetry.eq('N').sum()),
                test_C=int(test.centrosymmetry.eq('C').sum()), test_N=int(test.centrosymmetry.eq('N').sum()),
                train_groups=train.group_id.nunique(), features=n_features, cv_folds=train.fold.nunique()))
            for model in MODELS:
                resolved = {mode: select_run(cfg, model, domain, mode=mode, paper_run=paper_run,
                            overrides=overrides[mode]) for mode in ('final', 'cv')}
                run_rows.append(dict(project=project, domain=domain, model=model, **resolved))
                for run in resolved.values():
                    meta = ex.track(root / 'results' / f'meta_{run}.json')
                    meta_hashes[run] = ex.inputs[meta]
                run = resolved['final']
                selected_final[(model, f'{short}_{domain}')] = run
                hold = ex.read(root / 'results' / f'holdout_{run}.csv', dtype={'id': str})
                joined = test[['id', 'centrosymmetry']].merge(hold, on='id', validate='one_to_one')
                require(len(joined) == len(test) == len(hold)
                        and joined.y.eq(joined.centrosymmetry.map({'C': 0, 'N': 1})).all(),
                        f"Holdout does not match test labels/IDs: {run}")
                values = scores(joined)
                saved = ex.read(root / 'results' / f'metrics_{run}.csv').iloc[0]
                for metric in METRICS:
                    if metric in saved:
                        require(np.isclose(values[metric], saved[metric], atol=1e-12, rtol=0),
                                f"Saved {metric} differs: {run}")
                final.append(dict(project=project, domain={'N': 'I'}.get(domain, domain), model=model,
                                  n=len(joined), n_groups=test.group_id.nunique(), run_id=run, **values))
                if model == 'rf':
                    for element, z in [('Np', 93), ('Pu', 94)]:
                        sub = hold[hold.elements_cif.fillna('').str.split(',').apply(lambda x: element in x)]
                        if len(sub) >= 30 and sub.y.nunique() == 2:
                            extra_elements.append(dict(project=project, domain=domain, element=element, Z=z,
                                n=len(sub), C=int(sub.y.eq(0).sum()), N=int(sub.y.eq(1).sum()),
                                auc=roc_auc_score(sub.y, sub.y_prob)))
                    if domain == 'N':
                        precision, recall, threshold = precision_recall_curve(hold.y, hold.y_prob)
                        index = int(np.argmin(np.abs(recall[:-1] - 0.8)))
                        p, r = precision[index], recall[index]
                        thresholds.append(dict(project=project, domain='I', model=model,
                            threshold=threshold[index], precision=p, recall=r, f1=2*p*r/(p+r+1e-12)))
                cv_run = resolved['cv']
                oof = ex.read(root / 'results' / f'oof_{cv_run}.csv', dtype={'id': str})
                aligned = train[['id', 'centrosymmetry', 'fold']].merge(oof, on='id', validate='one_to_one',
                                                                     suffixes=('_expected', ''))
                require(len(aligned) == len(train) == len(oof)
                        and aligned.fold.eq(aligned.fold_expected).all()
                        and aligned.y.eq(aligned.centrosymmetry.map({'C': 0, 'N': 1})).all(),
                        f"OOF predictions do not match train folds: {cv_run}")
                saved_cv = ex.read(root / 'results' / f'metrics_{cv_run}.csv')
                saved_cv = saved_cv[pd.to_numeric(saved_cv.fold, errors='coerce').isin(range(5))].copy()
                require(len(saved_cv) == 5 and saved_cv.fold.astype(int).nunique() == 5,
                        f"Expected five fold-level rows: {cv_run}")
                fold_values = []
                for fold, sub in aligned.groupby('fold'):
                    value = scores(sub)
                    old = saved_cv[saved_cv.fold.astype(int).eq(int(fold))].iloc[0]
                    for metric in METRICS:
                        if metric in old:
                            require(np.isclose(value[metric], old[metric], atol=1e-12, rtol=0),
                                    f"CV {metric} differs: {cv_run}, fold={fold}")
                    fold_values.append(value)
                    cv_folds.append(dict(project=project, domain=domain, model=model, run_id=cv_run,
                                         fold=fold, n=len(sub), **value))
                fold_frame = pd.DataFrame(fold_values)
                row = dict(project=project, domain=domain, model=model, n_folds=5, run_id=cv_run)
                for metric in METRICS:
                    row[metric+'_mean'], row[metric+'_sd'] = fold_frame[metric].mean(), fold_frame[metric].std(ddof=1)
                cv.append(row)
    boot = ex.read(ex.base / PROJECTS[0] / 'results/paper/tables/bootstrap_summary.csv')
    require(len(boot) == 24 and not boot.duplicated(['project','model','domain','metric']).any()
            and boot.unit.eq('group_id').all() and boot.n_boot.eq(2000).all()
            and boot.n_requested.eq(2000).all() and boot.seed.eq(42).all(), 'Unexpected bootstrap protocol')
    for row in final:
        subset = boot[(boot.project == row['project']) & (boot.model == row['model']) & (boot.domain == row['domain'])]
        expected = row['project'] == PROJECTS[0] or row['domain'] == 'I'
        require(len(subset) == (3 if expected else 0), 'Unexpected bootstrap coverage')
        row['bootstrap_status'] = 'complete' if expected else 'not_run'
        if not expected:
            continue
        require(set(subset.metric) == {'auc', 'mcc', 'bacc'}, 'Incomplete bootstrap metrics')
        row.update(bootstrap_unit='group_id', n_requested=2000, seed=42)
        for _, item in subset.iterrows():
            require(np.isclose(item.test_value, row[item.metric], rtol=0, atol=1e-12)
                    and item.n_groups == row['n_groups'] and item.n_records == row['n']
                    and item.ci_low <= item.ci_high, 'Bootstrap input/point estimate mismatch')
            if item.metric != 'auc':
                require(item.threshold == 0.5, 'Unexpected bootstrap threshold')
            for suffix in ['ci_low', 'ci_high']:
                row[item.metric+'_'+suffix] = item[suffix]
            row[item.metric+'_n_valid'] = item.n_boot
    cross_dir = ex.base / PROJECTS[0] / 'results/cross' / (paper_run+'_CROSS') / cross_version
    state = ex.read_json(cross_dir / 'state.json')
    require(state['status'] == 'complete' and state['version'] == cross_version
            and len(state['tasks']) == 72, 'Incomplete cross-evaluation state')
    cross = ex.read(cross_dir / 'cross_metrics.csv')
    require(len(cross) == 144 and not cross.duplicated(['model','source','target','protocol']).any(),
            'Incomplete/duplicate cross-evaluation rows')
    for _, item in cross.iterrows():
        task = state['tasks'][f'{item.model}__{item.source}__to__{item.target}']
        require(item.run_id == selected_final[(item.model, item.source)] == task['run_id']
                and task['status'] == 'complete'
                and task['source_folds_sha256'] == fold_hashes[item.source]
                and task['target_folds_sha256'] == fold_hashes[item.target]
                and task['manifest_sha256'] == meta_hashes[item.run_id], 'Cross source provenance mismatch')
        recorded = [m for m in task['metrics'] if m['protocol'] == item.protocol]
        require(len(recorded) == 1, 'Cross protocol missing from state')
        for key in ['n', 'auc', 'bacc', 'mcc']:
            require(np.isclose(item[key], recorded[0][key], rtol=0, atol=1e-12, equal_nan=True),
                    'Cross metrics differ from completed state')
    for protocol, letter in [('all_test', 'a'), ('composition_disjoint_test', 'b')]:
        subset = cross[cross.protocol.eq(protocol)]
        require(len(subset) == 72, 'Incomplete cross protocol')
        ex.table(f'paper/tables/Table04{letter}_cross_{protocol}.csv', subset)
    ex.table('paper/tables/Table01_dataset_counts.csv', counts)
    ex.table('paper/tables/Table02_cv_metrics.csv', cv)
    ex.table('paper/tables/Table03_test_metrics_CI.csv', final)
    ex.table('supplement/tables/best_hyperparameters.csv', parameters)
    ex.table('supplement/tables/bootstrap_details.csv', boot)
    ex.table('supplement/tables/cv_fold_metrics.csv', cv_folds)
    ex.table('supplement/tables/exploratory_RF_thresholds.csv', thresholds)
    ex.table('supplement/tables/additional_Np_Pu_metrics.csv', extra_elements)
    ex.table('supplement/tables/run_identifiers.csv', run_rows)
    ex.copy(cross_dir / 'cross_overlap_summary.csv', 'supplement/tables/cross_overlap_summary.csv')


def collect_figures(ex, paper_run):
    cod, mp = (ex.base / project / 'results' for project in PROJECTS)
    aliases = {model: f'{paper_run}__final__{model}__magpie' for model in MODELS}
    for dataset, root, letter in [('MP', mp, 'a'), ('COD', cod, 'b')]:
        stems = [
            (mp / 'paper', f'paper_roc_panel__dataset_{dataset}', f'Fig01{letter}_ROC_{dataset}'),
            (mp / 'paper', f'stats_vs_n_elements_overlay__run_{aliases["lgbm"]}+{aliases["rf"]}__mode_final__ds={dataset}', f'Fig03{letter}_n_elements_{dataset}'),
            (root / 'paper', 'auc_by_periodic_block__overlay', f'Fig05{letter}_periodic_blocks_RF_{dataset}'),
        ]
        for folder, stem, destination in stems:
            ex.copy(folder / 'figures' / (stem+'.png'), f'paper/figures/{destination}.png')
            ex.copy(folder / 'tables' / (stem+'.csv'), f'paper/tables/{destination}.csv')
        project = 'COD_CANON_V2' if dataset == 'COD' else 'MP_ALL'
        ex.copy(root / 'paper/figures' / f'threshold_metrics__proj_{project}__model_rf__domain_I.png',
                f'paper/figures/Fig02{letter}_threshold_RF_{dataset}_I.png')
        ex.copy(root / 'paper/figures' / f'feature_importance__rf__run_{aliases["rf"]}_notitle.png',
                f'paper/figures/Fig06{letter}_importance_RF_{dataset}.png')
        number = '03' if dataset == 'MP' else '04'
        ex.copy(root / 'paper/figures' / f'feature_importance__lgbm__run_{aliases["lgbm"]}_notitle.png',
                f'supplement/figures/FigS{number}_importance_LGBM_{dataset}.png')
        for model in MODELS:
            for domain in DOMAINS:
                paths = list((root / aliases[model] / 'tables').glob(f'feature_importance__magpie__domain_{domain}__*__all.csv'))
                require(len(paths) == 1, f'Expected one full importance table: {project}/{model}/{domain}')
                ex.copy(paths[0], f'supplement/tables/importance_{model}_{dataset}_{domain}.csv')
    for root, domain, destination in [
        (mp, 'I', 'paper/Fig04a_auc_vs_Z_RF_MP_I'), (cod, 'M', 'paper/Fig04b_auc_vs_Z_RF_COD_M'),
        (cod, 'I', 'supplement/FigS01_auc_vs_Z_RF_COD_I'), (cod, 'O', 'supplement/FigS02_auc_vs_Z_RF_COD_O')]:
        section, name = destination.split('/')
        stem = f'metric_auc_vs_Z__model_rf__domain_{domain}'
        ex.copy(root / aliases['rf'] / 'figures' / (stem+'__no_labels.png'), f'{section}/figures/{name}.png')
        ex.copy(root / aliases['rf'] / 'tables' / (stem+'.csv'), f'{section}/tables/{name}.csv')
    ex.copy(cod / 'paper/figures/metric_vs_Z_legend.png', 'paper/figures/Fig04_legend.png')
    ex.copy(cod / 'paper/figures/metric_vs_Z_legend.png', 'supplement/figures/per_element_legend.png')
    ex.copy(cod / 'paper/figures/threshold_metrics_legend.png', 'paper/figures/Fig02_legend.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projects-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--paper-run', default='PAPER_V2')
    parser.add_argument('--cross-version', default='v0001')
    parser.add_argument('--runs', type=Path, help='JSON list with project/model/domain/final/cv selections')
    parser.add_argument('--tables-only', action='store_true', help='Export numerical tables without existing plot assets')
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new output directory')
    require(args.cross_version.startswith('v') and args.cross_version[1:].isdigit(), 'Invalid cross version')
    ex = Export(args.projects_base, args.output)
    collect_tables(ex, args.paper_run, args.runs, args.cross_version)
    if not args.tables_only:
        collect_figures(ex, args.paper_run)
    ex.write()
    print(f'Exported {len(ex.tables)} tables and {len(ex.copies)} assets to {ex.output}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare, submit, and inspect the replacement peptide/PDB confidence run.

Prepare first, then run the printed preflight command. --submit creates one new
job and stops the named old job only after the new job is accepted. Artifacts and
checkpoints of the old job are retained. API credentials are never copied.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.request
import uuid

import yaml

REFERENCE = Path('/mnt/project_abd/shiyu/peptide_cluster_32gpu_msa1024_resume_wandbfix_20261009')
SAVED_CONFIG = REFERENCE / 'runs/peptide_v2_32gpu_msa1024_resumed_wandbfix_20261010_065636/config.yaml'
PYTHON = '/mnt/project_abd/klp/miniconda3/envs/protenix/bin/python'
PEPTIDE = 'weightedPDB_before2109_wopb_nometalc_0925'
PDB = 'weightedPDB_before210930_v20260101'
WANDB_APP_BASE = 'https://app.bandw.top'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def api(args, endpoint, payload):
    token = os.environ.get('LANQI_API_KEY')
    if not token:
        token = re.search(r'sk-[A-Za-z0-9_-]+', args.api_key_file.read_text()).group()
    request = urllib.request.Request(
        'https://cloud.lanqi.tech/api/platform/open/v1/' + endpoint,
        data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': args.api_proxy}))
    with opener.open(request, timeout=45) as response:
        result = json.load(response)
    if result.get('code') != 0:
        raise RuntimeError(f"{endpoint}: code={result.get('code')} message={result.get('msg')}")
    return result


def detail(args, job_id):
    result = api(args, 'job/detail', {'job_id_list': [job_id]})
    entries = result['data'].get('job_detail_info_list') or result['data'].get('job_info_list')
    if not entries:
        raise RuntimeError(f'No detail found for {job_id}')
    value = entries[0]
    value.pop('env_vars', None)
    return value


def prepare(args):
    root = args.root
    if root.exists():
        raise FileExistsError(f'Refusing to overwrite prepared artifacts: {root}')
    old = detail(args, args.replace_job)
    assert old['worker_num'] == 4 and old['spec_info']['gpu_num'] == 8
    root.mkdir(parents=True)
    write_json(root / 'old_job_before.json', old)
    shutil.copytree(REFERENCE / 'code', root / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    runtime = Path(__file__).parent / 'runtime/train_peptide_pdb_cluster.py'
    shutil.copy2(runtime, root / 'code/scripts/runtime/train_peptide_pdb_cluster.py')
    c = yaml.safe_load(SAVED_CONFIG.read_text())
    c.update(project=c['project'], run_name='peptide_pdb_v2_confidence_32gpu', base_dir=str(root / 'runs'),
             wandb_id=uuid.uuid4().hex[:8], use_wandb=True, load_params_only=True, load_strict=True,
             load_checkpoint_path='/mnt/project_abd/public_databases/protenix.20260101/checkpoint/protenix-v2.pt',
             load_ema_checkpoint_path='', skip_load_optimizer=True, skip_load_scheduler=True,
             skip_load_step=True, load_step_for_scheduler=False, lr=1e-4, warmup_steps=3000,
             max_steps=100000, lr_scheduler='cosine_annealing', min_lr_ratio=0.1,
             iters_to_accumulate=1, diffusion_batch_size=1, diffusion_chunk_size=1,
             train_crop_size=1024, train_confidence_only=False,
             finetune_params_with_substring=[''], distributed_strategy='ddp', optimizer_sharding='none')
    c['adam']['lr'] = 1e-4
    c['af3_lr_scheduler'].update(lr=1e-4, warmup_steps=3000)
    c['model']['confidence_head']['stop_gradient'] = True
    c['loss']['resolution']['min'] = -1.0
    c['loss']['weight'].update(alpha_confidence=1e-4, alpha_pae=1.0, alpha_except_pae=1.0)
    c['data']['train_sets'] = [PEPTIDE, PDB]
    c['data']['train_sampler']['train_sample_weights'] = [0.5, 0.5]
    c['data']['test_sets'] = []
    c['data']['query_only_msa_probability'] = 0.0
    c['data']['msa']['enable_prot_msa'] = True
    c['data']['msa']['max_size']['train'] = 1024
    c['data']['msa']['sample_cutoff']['train'] = 1024
    for name in c['data']['train_sets']:
        c['data'][name]['cropping_configs']['crop_size'] = 1024
    (root / 'training_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    launch = (REFERENCE / 'launch.sh').read_text()
    launch = launch.replace('export WANDB_RESUME=must', 'export WANDB_RESUME=never')
    launch = launch.replace('export PROTENIX_RESUME_CHECKPOINT_STEP=1499',
                            'unset PROTENIX_RESUME_CHECKPOINT_STEP\nexport PROTENIX_MSA_DEBUG_LOG_LIMIT=4')
    launch = launch.replace('scripts/runtime/train_peptide_cluster.py', 'scripts/runtime/train_peptide_pdb_cluster.py')
    (root / 'launch.sh').write_text(launch)
    (root / 'launch.sh').chmod(0o755)
    job = json.loads((REFERENCE / 'job_config.json').read_text())
    job.update(job_name='peptide-pdb-confidence32-' + root.name.rsplit('_', 2)[-2] + '-' + root.name.rsplit('_', 1)[-1],
               job_description='Official Protenix-v2 weights; peptide/PDB 50:50; confidence incl PAE; resolution min -1; '
                               '32x1 structures, diffusion1, crop1024, MSA1024; LR1e-4 warmup3000 cosine100k.',
               entry_point='bash ' + str(root / 'launch.sh'), mount=old['mount_info_list'])
    job['mount'] = [{k: v for k, v in m.items() if k in ('path', 'volume_id', 'rw_setting', 'sub_dir')}
                    for m in job['mount']]
    write_json(root / 'job_config.json', job)
    manifest = {'old_job': args.replace_job, 'reference_config': str(SAVED_CONFIG),
                'reference_code': str(REFERENCE / 'code'), 'wandb_project': c['project'],
                'wandb_id': c['wandb_id'], 'wandb_url': WANDB_APP_BASE +
                '/shiyu19941007-individual-researcher/' + c['project'] + '/runs/' + c['wandb_id'],
                'sha256': {}}
    for path in [root / 'training_config.yaml', root / 'launch.sh', root / 'code/runner/train.py',
                 root / 'code/runner/wandb_logging.py', root / 'code/scripts/runtime/train_peptide_pdb_cluster.py']:
        manifest['sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(root / 'manifest.json', manifest)
    shutil.copy2(__file__, root / Path(__file__).name)
    print(f'Prepared {root}')
    print(f'Preflight: cd {root}/code && PROTENIX_ROOT_DIR=/mnt/project_abd/public_databases/protenix.20260101 '
          f'PYTHONPATH={root}/code {PYTHON} scripts/runtime/train_peptide_pdb_cluster.py '
          f'--config {root}/training_config.yaml --check-data')


def submit(args):
    root = args.root
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    assert json.loads((root / 'config_preflight.json').read_text())['valid']
    assert (root / 'data_preflight.json').exists()
    receipt = root / 'submission_response.json'
    if receipt.exists():
        job_id = json.loads(receipt.read_text())['data']['job_id']
    else:
        attempt = root / 'submission_attempt.json'
        if attempt.exists():
            raise RuntimeError('A submission was attempted without a receipt. Inspect job/list before retrying.')
        write_json(attempt, {'time': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        response = api(args, 'job/create', json.loads((root / 'job_config.json').read_text()))
        write_json(receipt, response)
        job_id = response['data']['job_id']
        (root / 'job_id.txt').write_text(job_id + '\n')
    new = detail(args, job_id)
    write_json(root / 'new_job_detail.json', new)
    assert new['status'] in (10000, 10001, 10002, 10003), new
    old_id = manifest['old_job']
    old = detail(args, old_id)
    if old['status'] in (10000, 10001, 10002, 10003, 10012):
        response = api(args, 'job/stop', {'job_id': old_id})
        write_json(root / 'old_job_stop_response.json', response)
    print(json.dumps({'new_job': job_id, 'new_status': new['status'], 'replaced_job': old_id,
                      'wandb_url': manifest['wandb_url']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--replace-job', default='jo-ddorierp2mfx3c45')
    parser.add_argument('--api-key-file', type=Path, default=Path('/root/shiyu/API_Key.txt'))
    parser.add_argument('--api-proxy', default='http://127.0.0.1:8119')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args)
    elif args.submit:
        submit(args)
    else:
        for job_id in [(args.root / 'job_id.txt').read_text().strip(),
                       json.loads((args.root / 'manifest.json').read_text())['old_job']]:
            value = detail(args, job_id)
            write_json(args.root / f'status_{job_id}.json', value)
            print(json.dumps({k: value.get(k) for k in ('job_id', 'status', 'err_msg', 'active_alert_count')}))


if __name__ == '__main__':
    main()

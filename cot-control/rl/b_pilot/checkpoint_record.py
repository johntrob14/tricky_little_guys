"""Record checkpoint sizes and optionally publish the final sampler; never deletes."""
import argparse, asyncio, json
from pathlib import Path
from dotenv import load_dotenv
import tinker

async def main(a):
    out=Path(a.out);load_dotenv(Path.home()/'.config/sauce/tinker.env')
    ck=json.load(open(out/'checkpoints.json'));final=ck[-1]['sampler'];run=tinker.types.ParsedCheckpointTinkerPath.from_tinker_path(final).training_run_id
    client=tinker.ServiceClient().create_rest_client()
    runs=sorted({tinker.types.ParsedCheckpointTinkerPath.from_tinker_path(c['sampler']).training_run_id for c in ck})
    checkpoints={r:(await client.list_checkpoints_async(r)).model_dump(mode='json') for r in runs}
    record=dict(final_training_run_id=run,checkpoints_by_run=checkpoints,published=[])
    if a.publish:
        await client.publish_checkpoint_from_tinker_path_async(final)
        record['checkpoints_by_run']={r:(await client.list_checkpoints_async(r)).model_dump(mode='json') for r in runs}
        final_meta=next(c for c in record['checkpoints_by_run'][run]['checkpoints'] if c['tinker_path']==final)
        assert final_meta['public'],final_meta
        summary=json.load(open(out/'summary.json'))
        record['published'].append(dict(name='nocotrl-b-pilot-final',path=final,expires_at=final_meta['expires_at'],headline=dict(held_out_phrasing_output_strict={l:summary['dev'][l]['constrained']['strict_robust'] for l in ['base','pilot']},held_out_phrasing_n=53,tiktok_pass={l:summary['tiktok'][l]['passed'] for l in ['base','pilot']},tiktok_n=4,clean_held_out_cot_strict={l:summary['astra_groups']['clean_heldout_style'][l]['strict'] for l in ['base','pilot']},clean_held_out_cot_n=36),description='GPT-OSS-120B output-only RL with empty analysis; mixed 480-task pilot, including short executable tool episodes. Output control improved; no convincing held-out CoT transfer.'))
    json.dump(record,open(out/'checkpoint_inventory.json','w'),indent=2);print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--publish',action='store_true');asyncio.run(main(p.parse_args()))

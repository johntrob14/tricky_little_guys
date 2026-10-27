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
        record['published'].append(dict(name='nocotrl-b-pilot-final',path=final,description='GPT-OSS-120B output-only RL with empty analysis; mixed 480-task pilot, including short executable tool episodes. See paired results; not a generalization claim.'))
    json.dump(record,open(out/'checkpoint_inventory.json','w'),indent=2);print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--publish',action='store_true');asyncio.run(main(p.parse_args()))

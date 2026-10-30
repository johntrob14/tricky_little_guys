"""Prepare a reproducible, blinded style-rating experiment; no inference."""
import collections
import hashlib
import json
import pathlib
import random
import re

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE.parent
SEED = 20261029
MODELS = ['base', 'mt1b-s150', 'mt2e-s170']
FIELDS = ['reasoning', 'response']

def units(text):
    # Preserve paragraph/table structure; split prose at sentence boundaries.
    result = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if paragraph.startswith('```') or '|' in paragraph or '\n' in paragraph:
            chunks = [paragraph]
        else:
            chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z“"\[])', paragraph)
        for chunk in chunks:
            result.append({'index': len(result), 'text': chunk,
                           'words': len(chunk.split())})
    assert sum(x['words'] for x in result) == len(text.split())
    return result

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    rows = {}
    hashes = {}
    configs = {}
    for model in MODELS:
        path = SOURCE / model / 'baseline.jsonl'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = [json.loads(line) for line in path.open()]
        assert len(data) == 175
        rows[model] = {r['scenario_id']: r for r in data if r['rep'] == 0}
        assert len(rows[model]) == 35
        configs[model] = json.load(open(SOURCE / model / 'config.json'))
    scenarios = sorted(rows[MODELS[0]])
    assert all(set(rows[m]) == set(scenarios) for m in MODELS)
    for scenario in scenarios:
        assert len({rows[m][scenario]['user'] for m in MODELS}) == 1
        assert all(rows[m][scenario]['finish'] == 'stop' for m in MODELS)

    selected = [(s, m, f) for s in scenarios for m in MODELS for f in FIELDS]
    rng.shuffle(selected)
    manifest = []
    shards = [[] for _ in range(3)]
    full_records = []
    crop_lengths = {}
    for s in scenarios:
        crop_lengths[s] = min(60, *(len(rows[m][s][f].split())
                                    for m in MODELS for f in FIELDS))
    # Same relative window location across all six groups within a scenario.
    positions = {s: rng.random() for s in scenarios}
    for i, (scenario, model, field) in enumerate(selected):
        row = rows[model][scenario]
        record_id = f'x{i:04d}'
        primary_shard = i % 3
        excerpt_shard = (primary_shard + 1 + rng.randrange(2)) % 3
        record = {'id': record_id, 'view': 'full', 'question': row['user'],
                  'units': units(row[field])}
        shards[primary_shard].append(record)
        full_records.append(record)
        words = row[field].split()
        n = crop_lengths[scenario]
        start = round(positions[scenario] * (len(words) - n))
        excerpt = {'id': f'e{i:04d}', 'view': 'excerpt',
                   'question': row['user'],
                   'units': units(' '.join(words[start:start+n]))}
        shards[excerpt_shard].append(excerpt)
        manifest.append({'id': record_id, 'excerpt_id': excerpt['id'],
                         'scenario_id': scenario, 'model': model,
                         'channel': field, 'rep': 0,
                         'primary_shard': primary_shard,
                         'excerpt_shard': excerpt_shard,
                         'excerpt_words': n, 'excerpt_start_word': start,
                         'full_words': len(words)})
    # Five repeats in each of the six groups, with new opaque identifiers.
    repeated = []
    for model in MODELS:
        for field in FIELDS:
            eligible = [r for r in manifest
                        if r['model'] == model and r['channel'] == field]
            repeated.extend(rng.sample(eligible, 5))
    by_id = {r['id']: r for r in full_records}
    for i, m in enumerate(repeated):
        repeat_id = f'z{i:04d}'
        repeat_shard = 3 - m['primary_shard'] - m['excerpt_shard']
        shards[repeat_shard].append(dict(by_id[m['id']], id=repeat_id))
        m.update(repeat_id=repeat_id, repeat_shard=repeat_shard)
    for i, shard in enumerate(shards):
        rng.shuffle(shard)
        (HERE / f'items_{i}.jsonl').write_text(''.join(
            json.dumps(r, ensure_ascii=False) + '\n' for r in shard))
    write_json(HERE / 'manifest.json', manifest)
    config = {'seed': SEED, 'primary_texts': len(manifest),
              'full_text_repeats': len(repeated), 'excerpt_texts': len(manifest),
              'scenarios': len(scenarios), 'source_sha256': hashes,
              'source_configs': configs,
              'shard_counts': [len(x) for x in shards],
              'excerpt_word_range': [min(crop_lengths.values()), max(crop_lengths.values())],
              'primary_words': sum(x['full_words'] for x in manifest),
              'judge': 'Fresh Codex scorer subagents, inherited model; exact model ID unavailable',
              'blinding': 'No model, channel, prior labels or hypothesis in scorer inputs. Each scorer sees at most one full version per original text.',
              'external_model_api_spend_usd': 0,
              'target_model_sampling_calls': 0}
    write_json(HERE / 'config.json', config)
    print(json.dumps({k: config[k] for k in ['primary_texts', 'full_text_repeats',
          'excerpt_texts', 'scenarios', 'shard_counts', 'excerpt_word_range',
          'primary_words']}, indent=2))

if __name__ == '__main__':
    main()

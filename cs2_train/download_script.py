from pathlib import Path
import json

from urllib.parse import urlparse
import boto3

# Download from the S3 buckets and store into data/ repo.

# demos/, videos/, actions/, annotations/
DEMOS_BUCKET_NAME = 'csgo-demos-s3-1' #demos/
RENDERS_BUCKET_NAME = 'csgo-renders-s3-1' #videos/
ACTIONS_BUCKET_NAME = 'csgo-actions-s3-1' #actions/
DEBUG_BUCKET_NAME = 'csgo-debug-s3-1' #debug/

# Util functions
session = boto3.Session()

def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)

    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    return bucket, key

def download_clip(clip, tmp_dir):
    s3 = session.client('s3')
    tmp_dir = Path(tmp_dir)

    clip_path = tmp_dir / 'videos' / f'match_{clip["match_id"]}' / f'player_{clip["player_id"]}' / f'clip_{int(clip["clip_id"]):03d}.mp4'
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    if not clip_path.exists():
        print(f'Clip {clip["clip_id"]} for player {clip["player_id"]} does not exist, downloading.')

        #Split the uri
        s3_uri = clip['video_s3']
        bucket, key = parse_s3_uri(s3_uri)

        s3.download_file(Bucket=bucket, Key=key, Filename=str(clip_path))

    print(f'Clip {clip["clip_id"]} for player {clip["player_id"]} downloaded.')
    return clip_path

def download_actions(clip, tmp_dir):
    s3 = session.client('s3')

    tmp_dir = Path(tmp_dir)
    actions_path = tmp_dir / 'actions' / f'match_{clip["match_id"]}' / f'player_{clip["player_id"]}' / f'clip_{int(clip["clip_id"]):03d}.parquet'
    actions_path.parent.mkdir(parents=True, exist_ok=True)

    if not actions_path.exists():
        print(f'Actions {clip["clip_id"]} for player {clip["player_id"]} does not exist, downloading.')

        #Split the uri
        s3_uri = clip['actions_s3']
        bucket, key = parse_s3_uri(s3_uri)

        s3.download_file(Bucket=bucket, Key=key, Filename=str(actions_path))

    print(f'Actions {clip["clip_id"]} for player {clip["player_id"]} downloaded.')
    return actions_path

def download_debug(clip, tmp_dir):
    s3 = session.client('s3')
    tmp_dir = Path(tmp_dir)

    clip_path = tmp_dir / 'debug' / f'match_{clip["match_id"]}' / f'player_{clip["player_id"]}' / f'clip_{int(clip["clip_id"]):03d}.mp4'
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    if not clip_path.exists():
        print(f'Clip {clip["clip_id"]} for player {clip["player_id"]} does not exist, downloading.')

        #Split the uri
        s3_uri = clip['debug_s3']
        bucket, key = parse_s3_uri(s3_uri)

        s3.download_file(Bucket=bucket, Key=key, Filename=str(clip_path))

    print(f'Debug clip {clip["clip_id"]} for player {clip["player_id"]} downloaded.')
    return clip_path


# 1. Create the dataset folder
project_root = Path.cwd()
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# 2. Download the manifest
s3 = session.client('s3')

manifest_path = data_dir / 'manifest.json'

s3.download_file(
    Bucket=ACTIONS_BUCKET_NAME,
    Key='manifest.json',
    Filename=str(manifest_path),
)

with open(manifest_path, 'r') as f:
    manifest = json.load(f)

print(manifest)

# 3. Download clips and actions

debug = True

for clip in manifest:
    print(clip)
    clip_path = download_clip(clip, data_dir)
    actions_path = download_actions(clip, data_dir)
    
    if debug:
        debug_path = download_debug(clip, data_dir)

# Verify the dataset
total_ticks = 0

for clip in manifest:
    ticks = int(clip['end_tick']) - int(clip['start_tick'])
    total_ticks += ticks

print(total_ticks)
duration_s = total_ticks/64
print(f"Duration: {duration_s}seconds = {duration_s/60}mins = {duration_s/60/60}hours")

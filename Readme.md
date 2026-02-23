# swing_annotator

Video-to-image classification dataset tool with anchor-based labeling.

## Install

```powershell
setup.bat
```

or

```powershell
pip install opencv-python
```

## Run GUI

```powershell
run.bat
```

or

```powershell
python gui_app.py
```

## GUI workflow

1. Open a video.
2. Move to a frame.
3. Type label and click `Add/update at current frame`.
4. Repeat to define anchors (same label is applied until next anchor).
5. Export dataset.

## Save labels for multiple videos (append mode)

- `Label DB` is a single JSON file storing labels per video path.
- Click `Append save this video` to upsert current video's anchors.
- Click `Load this video` to restore anchors for the current video.

This lets you continue labeling across many videos without overwriting others.

## Roboflow format output

Set `format=roboflow` in GUI (default), then export.

Output structure:

- `output/train/<label>/*.jpg`
- `output/valid/<label>/*.jpg`
- `output/test/<label>/*.jpg`
- `output/labels.txt`
- `output/manifest.csv`

`split` uses deterministic ratio (default `80,10,10`).

## CLI

```powershell
python dataset_builder.py `
  --video input.mp4 `
  --anchors anchors.json `
  --output dataset_rf `
  --format roboflow `
  --split 80,10,10 `
  --step 2
```

`--format native` also exists and exports `images/<label>/*.jpg`.

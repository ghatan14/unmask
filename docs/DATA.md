# Data preparation

UNMASK works on face crops stored as images, one folder per class. A folder whose name contains
`real` is real, every other folder is fake (`--real_hint` changes the substring). Frames of one
video share the file-name stem before `_frame_` (`--video_sep`); the video id used for video-level
AUC is `<class folder>/<stem>`.

```
data/
├── FaceForensics++/c23/
│   ├── train/{real_youtube,fake_four}/
│   ├── val/{real_youtube,fake_four}/
│   └── test/{real_youtube,fake_four}/
├── Celeb-DF-v2/test/{real,fake}/
├── WildDeepfake/test/{real,fake}/
└── HydraFake/
    ├── train/{real,fake}/...
    ├── val/{real,fake}/
    ├── test/{id,cm,cf,cd}/<subset>/{0_real,1_fake}/
    └── jsons/test/{id,cm,cf,cd}/*.json        # official test lists (Git LFS)
```

## FaceForensics++ (c23) and Celeb-DF-v2

We use the face crops of the FSFM / FS-VFM pre-processing (32 frames sampled per video, one face
crop per frame). Follow the fine-tuning data instructions of the
[FSFM repository](https://github.com/wolo-wolo/FSFM-CVPR25) and place (or link) the output as
above. The FF++ file names must keep the video ids, e.g.

```
train/real_youtube/youtube_001_frame_216.png
train/fake_four/DeepFakes_001_870_frame_0.png       # <manipulation>_<target>_<source>_frame_<n>
```

The first numeric token (the target video, `001` above) pairs every forgery with the real video it
was made from in the same batch, and the manipulation name (matched case-insensitively) lets
training hold out NeuralTextures (`--holdout_manip NeuralTextures`).

**Protocol note.** Our Celeb-DF-v2 test folder holds 448 videos (340 fake, 108 Celeb-real; the
YouTube-real videos of the official list are absent from the pre-processed set), 14,268 frames.
All Celeb-DF-v2 numbers of the paper, ours and re-measured baselines alike, use these 448 videos;
numbers quoted from other papers use their own protocol (usually the official 518 videos).

## WildDeepfake

```bash
python tools/prepare_wdf.py --out_dir data/WildDeepfake --frames_per_seq 32
```

downloads only the test archives of the Hugging Face dataset `xingjunm/WildDeepfake` and writes
one frame per file, grouped by face sequence (`<archive>-<sequence>_frame_<n>.png`). The paper
uses 806 test sequences with 32 frames each (25,792 frames).

## HydraFake

HydraFake is released with Veritas (Tan et al., ICLR 2026; see the
[Veritas repository](https://github.com/EricTan7/Veritas) for the download, ModelScope dataset
`EricTanh/HydraFake`). The official test lists in `jsons/` are Git-LFS files; a plain clone leaves
small pointer files, so download them inside the dataset folder:

```bash
git lfs install
git lfs pull --include="jsons/test/**,jsons/val/**"
```

Then check the dataset and set aside unreadable images (the release contains at least one empty
file in `train/`):

```bash
python tools/eval_hydrafake.py --check --root data/HydraFake      # 23/23 lists, images per subset
python tools/check_images.py --root data/HydraFake --split train val --move
```

The training split used in the paper has 48,160 images (24,160 real, 24,000 fake) and the
validation split 4,000. Without the official lists the evaluator falls back to scanning the
folders; one subset (Facevid2vid) then uses a seeded sample of FF++ reals and is marked in every
table, so final numbers should always use the official lists.

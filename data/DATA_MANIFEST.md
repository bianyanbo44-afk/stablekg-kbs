# StableKG data manifest

All public data were downloaded on 2026-09-07 (Asia/Shanghai) from the
versioned TFLEX dataset mirrors. The mirror metadata reported `Apache-2.0` on
the access date. The archive hashes below are SHA-256 hashes of the exact
files stored in `data/raw/`.

| Dataset | Source archive | Mirror revision | Archive size (bytes) | SHA-256 | License metadata |
|---|---|---:|---:|---|---|
| ICEWS14 | [linxy/ICEWS14 `zips/all.zip`](https://huggingface.co/datasets/linxy/ICEWS14/resolve/main/zips/all.zip) | `462141693b1c9c2ece49384fab41955eae89471a` | 20,350,171 | `0BF107F5413221FEF3703F5025814015D8E329D2FE31CA8446F06561A0B17C77` | Apache-2.0 |
| ICEWS05-15 | [linxy/ICEWS05_15 `zips/all.zip`](https://huggingface.co/datasets/linxy/ICEWS05_15/resolve/main/zips/all.zip) | `17a1a7af550d7d61fa2b0a718dc742fcdc31fa11` | 234,179,108 | `ADAE2A8D958BE85D04C1494F70CAF0F5D8F6C06CAF29BE518A7A3541FF4A3286` | Apache-2.0 |
| GDELT | [linxy/GDELT `zips/all.zip`](https://huggingface.co/datasets/linxy/GDELT/resolve/main/zips/all.zip) | `8ee57955b595b4edddf39b5bb93ea3309bc3233f` | 410,911,855 | `34837172579C8577AA2CA78F29EC0DFC2DF0D798FD408188C829A0943C61242C` | Apache-2.0 |

The archive contains `train.jsonl`, `valid.jsonl` and `test.jsonl` at its root.
The query-format source is [TFLEX](https://github.com/LinXueyuanStdio/TFLEX).
Its source paper is linked in that repository. Dataset provenance is attributed
to that distribution, separately from the manuscript's 2025--2026 related work.
The download script resolves the recorded immutable revision, not the moving main branch.
The files are TFLEX query-format records. The experiments retain only records
whose `query_name` is `Pe`, and treat each record's answer list as a set of
labelled objects. The benchmark counts, entity counts and timestamp counts are
recorded in the corresponding `*_meta.json` files.

## Data processing and leakage controls

The public benchmark uses the training events as the evidence index and applies
the causal condition `event_time <= query_time` before scoring a query. The
validation split is randomly divided into calibration and operating halves
using a fixed seed. Test labels are read only after the calibrator and
operating thresholds have been frozen. The chronological audit instead joins
all splits by timestamp, predicts every query at time `t` before adding any
event from `t`, and uses a 70/15/15 timestamp split.

## Re-download and verification

```powershell
python scripts/download_data.py --dataset all --root data/raw
```

The script verifies the archive hash before extraction and refuses to overwrite
an archive whose hash does not match this manifest. No credentials are required.

# StableKG

**Exact stability certificates for neural temporal knowledge graph completion**

StableKG combines a frozen neural predictor with an editable temporal memory.
It computes exact guarantees for shared-window evidence deletion, uses those
certificates alongside calibrated confidence, and refreshes affected answers
locally after an event edit.

This version accompanies the manuscript prepared for **Neurocomputing**.

- [Main manuscript](manuscript/neurocomputing/main.pdf)
- [Supplementary Information](manuscript/neurocomputing/supplementary.pdf)
- [Reproduction instructions](README_NEUROCOMPUTING.md)
- [Editable figures and source data](figures/neurocomputing)
- [Submission materials](submission/neurocomputing)
- [Release assets](https://github.com/bianyanbo44-afk/stablekg-kbs/releases/tag/neurocomputing-v1)

The experiments use TeRDy and a compact temporal factorization model on
ICEWS14 and ICEWS05-15, with three fitted seeds per combination. GDELT provides
a separate maintenance benchmark. Exact, conservative and sampled deletion
checks are compared on the same predictions.

To rebuild the figures and paper from archived query outputs:

```powershell
python -m pip install -r requirements-neurocomputing.txt
./run_neurocomputing.ps1 -CompilePaper
```

The source archive is downloaded from the release and verified by SHA-256.
The full training route is documented in the reproduction instructions.

Author: Yanbo Bian, Weihai International College, Beijing Jiaotong University.
Contact: 24722081@bjtu.edu.cn.

Original code is MIT licensed. Public datasets and external implementations
retain their original terms; source URLs and hashes are in
[data/DATA_MANIFEST.md](data/DATA_MANIFEST.md).

The original KBS study is preserved at commit `18267a3` and described in the
[historical README](revision/README_KBS_HISTORY.md). Its outputs are separate
from the present renormalized-memory experiments.

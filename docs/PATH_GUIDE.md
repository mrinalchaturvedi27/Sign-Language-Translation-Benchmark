# Path Configuration - Visual Guide 📊

## Understanding Path Setup

### What Your Mentor Sends You 📨

```python
# Example from your mentor (Sanjeet)
train_csv = '/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/train_split_unicode_filtered_matched.csv'
val_csv = '/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/val_split_unicode_filtered_matched.csv'
test_csv = '/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/translationexperiment/tokenization/test_split_unicode_filtered_matched.csv'

POSE_DIR_ISIGN = "/DATACSEShare/sanjeet/dattatreya/jan8A40/rtmfeaturesisign/performance/"
```

### ❌ Don't Do This

**Bad:** Copy paths directly without changing
```yaml
# configs/my_experiment.yaml
data:
  train_path: "/DATACSEShare/sanjeet/..."  # ← Won't work for you!
  pose_dir: "/DATACSEShare/sanjeet/..."    # ← This is someone else's path!
```

**Result:** `FileNotFoundError: No such file or directory`

---

### ✅ Do This Instead

#### Step 1: Start with Template

```yaml
# configs/TEMPLATE.yaml (provided)
data:
  train_path: "/PATH/TO/YOUR/train.csv"  # ← Placeholder
  val_path: "/PATH/TO/YOUR/val.csv"      # ← Placeholder
  test_path: "/PATH/TO/YOUR/test.csv"    # ← Placeholder
  pose_dir: "/PATH/TO/YOUR/POSE/FILES/"  # ← Placeholder
```

#### Step 2: Replace with YOUR Paths

```yaml
# configs/my_experiment.yaml (your file)
data:
  train_path: "/DATA7/yourname/project/train.csv"  # ← Your path!
  val_path: "/DATA7/yourname/project/val.csv"
  test_path: "/DATA7/yourname/project/test.csv"
  pose_dir: "/DATA7/yourname/project/poses/"
```

**Result:** ✓ Works perfectly!

---

## Visual Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                     MENTOR'S PATHS                              │
│  (Reference - shows where data is on their system)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  /DATACSEShare/sanjeet/dattatreya/.../train.csv                │
│  /DATACSEShare/sanjeet/dattatreya/.../val.csv                  │
│  /DATACSEShare/sanjeet/datatatreya/.../performance/            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │  Copy the STRUCTURE, not the path
                             │  (Same filenames, your location)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      YOUR PATHS                                 │
│  (Actual paths you use in your config)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  /DATA7/yourname/myproject/train.csv                           │
│  /DATA7/yourname/myproject/val.csv                             │
│  /DATA7/yourname/myproject/poses/                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Different Team Members, Different Paths ✨

### Team Member: Sanjeet

```yaml
data:
  train_path: "/DATACSEShare/sanjeet/dattatreya/.../train.csv"
  pose_dir: "/DATACSEShare/sanjeet/datatatreya/.../performance/"
```

### Team Member: Ashish

```yaml
data:
  train_path: "/DATA405/ashishu23/SURGE/iSign-videos_v1.1/tokenization/train.csv"
  pose_dir: "/DATA405/ashishu23/SURGE/iSign-videos_v1.1/poses/"
```

### Team Member: You

```yaml
data:
  train_path: "/DATA7/yourname/data/train.csv"  # ← Fill in your path
  pose_dir: "/DATA7/yourname/data/poses/"
```

### All use same code, different configs! 🎉

---

## Common Path Patterns

### Pattern 1: Shared Server Directory

```
/DATAXShare/username/project/...
/DATA405/username/project/...
/DATA7/username/project/...
```

**Your path will look like:**
```yaml
train_path: "/DATA<number>/<your_username>/
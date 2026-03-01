import os, shutil

base = r'd:\github.com\ninja-marduk\ml_precipitation_prediction\docs\papers\1'
items = os.listdir(base)
revdir = [i for i in items if i.startswith('Revi')][0]
scidir = os.path.join(base, revdir, 'papers', 'Scientific articles')
tmpdir = r'd:\github.com\ninja-marduk\ml_precipitation_prediction\tmp_papers'
os.makedirs(tmpdir, exist_ok=True)

prefix = "\\\\?\\"

for f in os.listdir(scidir):
    if not f.endswith('.pdf'):
        continue
    src = prefix + os.path.join(scidir, f)

    if 'Artificial intelligent' in f:
        dst = os.path.join(tmpdir, 'zerouali2023.pdf')
    elif 'standardized precipitation' in f:
        dst = os.path.join(tmpdir, 'coskun2023.pdf')
    elif 'Gaussian mutation' in f:
        dst = os.path.join(tmpdir, 'ehteram2024.pdf')
    elif 'lion swarm' in f.lower():
        dst = os.path.join(tmpdir, 'priestly2023.pdf')
    else:
        continue

    try:
        shutil.copy2(src, dst)
        print(f'OK: {os.path.basename(dst)}')
    except Exception as e:
        print(f'FAIL {os.path.basename(dst)}: {e}')

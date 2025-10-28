import zipfile
import os
from pathlib import Path

# ZIP 파일 생성
zip_path = '/mnt/e/Final_project/deepfake_submission_new.zip'
source_dir = '/mnt/e/Final_project/deepfake_submit.zip'

print(f"Creating submission ZIP: {zip_path}")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # task.ipynb 추가 (루트에)
    task_nb = os.path.join(source_dir, 'task.ipynb')
    if os.path.exists(task_nb):
        zipf.write(task_nb, 'task.ipynb')
        print(f"✓ Added: task.ipynb")
    else:
        print(f"✗ Not found: {task_nb}")

    # model 폴더 추가
    model_dir = os.path.join(source_dir, 'model', 'deepfake_vit_final')
    if os.path.exists(model_dir):
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file == 'task.ipynb':  # 모델 폴더 안의 task.ipynb는 제외
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.join('model', 'deepfake_vit_final',
                                      os.path.relpath(file_path, model_dir))
                zipf.write(file_path, arcname)
                print(f"✓ Added: {arcname}")
    else:
        print(f"✗ Model directory not found: {model_dir}")

print(f"\n✓ ZIP created successfully: {zip_path}")

# ZIP 내용 확인
print("\n=== ZIP Contents ===")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    for info in zipf.filelist:
        print(f"  {info.filename} ({info.file_size} bytes)")

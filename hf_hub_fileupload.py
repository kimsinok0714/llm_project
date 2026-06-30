from huggingface_hub import HfApi

api = HfApi()

# 업로드할 저장소 정보 설정
repo_id = "Kimsinok0714/raw_model"
local_folder = "/home/ubuntu/llm_project/exaone4-law-refined"

# 1. Hugging Face Hub에 저장소 생성 (이미 존재하면 생략 가능)
# private=True로 설정하면 본인만 볼 수 있는 비공개 저장소가 됩니다.
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

# 2. 폴더 내 모든 파일 업로드
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model"
)
print(f"업로드가 완료되었습니다! ➔ https://huggingface.co/{repo_id}")
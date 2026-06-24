import os
import glob
import shutil
import subprocess
import datetime
from typing import List

def push_experiment_results(
    token: str,
    repo_url: str,
    branch: str,
    files: List[str],
    commit_message: str = "feat(experiment): auto-save results"
):
    """
    Google Colab 上の実験生成物ファイルを検出し、
    日付別の outputs フォルダに格納した上で、GitHubへ自動プッシュします。
    
    Args:
        token (str): GitHubのPersonal Access Token (PAT)
        repo_url (str): リポジトリのURL (例: "github.com/ShotaMiwa/year1-research.git")
        branch (str): プッシュ先のブランチ名 (例: "refactor/memory-improvements")
        files (List[str]): 保存・プッシュ対象のファイルリスト (ワイルドカード対応, 例: ["*.csv", "sentiment_samples_*.md"])
        commit_message (str): コミットメッセージの接頭辞
    """
    if not token:
        print("[git_pusher] エラー: GitHubのアクセストークン(GH_PAT)が空です。処理をスキップします。")
        return

    # 1. 保存対象ファイルを glob を使って収集
    matched_files = []
    for pattern in files:
        matched = glob.glob(pattern)
        matched_files.extend(matched)
    
    # 重複の削除
    matched_files = list(set(matched_files))
    
    if not matched_files:
        print(f"[git_pusher] 警告: 指定されたパターンにマッチするファイルが見つかりません。対象: {files}")
        return

    print(f"[git_pusher] 検出された保存対象ファイル: {matched_files}")

    # 2. タイムスタンプ付きの出力先フォルダを定義
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"run_{timestamp}"
    
    # git_pusher.pyの場所からリポジトリのルートパスを自動検出
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # outputs/ ディレクトリ配下に作成
    output_path = os.path.join(repo_root, "outputs", output_dir_name)
    os.makedirs(output_path, exist_ok=True)

    # 3. ファイルを保存先フォルダにコピー
    copied_files = []
    for file in matched_files:
        if os.path.exists(file):
            dest_file = os.path.join(output_path, os.path.basename(file))
            shutil.copy2(file, dest_file)
            copied_files.append(dest_file)
            print(f"[git_pusher] ファイルをコピーしました: {file} -> {dest_file}")

    if not copied_files:
        print("[git_pusher] 保存先フォルダへのファイルのコピーに失敗しました。")
        return

    # 4. Git コマンドを実行してプッシュ
    try:
        # トークンを埋め込んだURLを構築 (プロトコルのプレフィックスを確認)
        clean_url = repo_url.replace("https://", "").replace("http://", "")
        authed_url = f"https://{token}@{clean_url}"

        # Gitのユーザー設定（Colab環境用）
        subprocess.run(["git", "config", "user.name", "ColabBot"], cwd=repo_root, check=True)
        subprocess.run(["git", "config", "user.email", "colab-bot@example.com"], cwd=repo_root, check=True)

        # リモートURLの書き換え
        subprocess.run(["git", "remote", "set-url", "origin", authed_url], cwd=repo_root, check=True)

        # 最新変更を pull --rebase して競合を回避
        print(f"[git_pusher] 最新のリモートブランチ '{branch}' からリベースプルを実行中...")
        subprocess.run(["git", "pull", "--rebase", "origin", branch], cwd=repo_root, check=True)

        # outputs フォルダを追加
        print(f"[git_pusher] 変更をインデックスに追加中...")
        subprocess.run(["git", "add", "outputs/"], cwd=repo_root, check=True)

        # コミット
        full_commit_msg = f"{commit_message} {timestamp}"
        print(f"[git_pusher] コミットを実行中: '{full_commit_msg}'")
        subprocess.run(["git", "commit", "-m", full_commit_msg], cwd=repo_root, check=True)

        # プッシュ
        print(f"[git_pusher] リモートブランチ '{branch}' へプッシュ中...")
        subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], cwd=repo_root, check=True)
        
        print(f"\n[git_pusher] ★ 実験結果のGitHub自動保存が完了しました！ ({output_path})")

    except subprocess.CalledProcessError as e:
        print(f"\n[git_pusher] エラー: Gitコマンドの実行に失敗しました: {e}")
        # 例外を投げずにエラー表示のみに留め、実験自体の停止を防ぐ
    except Exception as e:
        print(f"\n[git_pusher] 予期せぬエラーが発生しました: {e}")

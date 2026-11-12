# GitHub upload — easiest method on Windows

## Option A: GitHub Desktop (recommended)

1. Install **GitHub Desktop** from the official GitHub website.
2. Sign in with your GitHub account.
3. Choose **File -> Add local repository**.
4. Select the extracted `github_project` folder.
5. If GitHub Desktop says it is not a repository, choose **create a repository** from that folder.
6. Repository name suggestion:
   `dual-microphone-respiratory-sound-ml`
7. Keep the project **Public** only if you are comfortable publishing the report, presentation, notebooks, checkpoints and prediction CSV. Otherwise choose **Private**.
8. Click **Create repository**.
9. Review the changed files. All project files are small enough for normal GitHub storage; no file in this package is near GitHub's 100 MB single-file hard limit.
10. Enter a commit message such as:
    `Add respiratory sound ML experiments and project documentation`
11. Click **Commit to main**.
12. Click **Publish repository**.
13. Refresh the GitHub repository page and verify the folders:
    `data`, `models`, `notebooks`, `results`, `docs`, `archive`.

## Option B: Git command line

Open Git Bash or PowerShell inside the extracted folder:

```bash
git init
git add .
git commit -m "Add respiratory sound ML experiments and project documentation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dual-microphone-respiratory-sound-ml.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and create the empty GitHub repository first.

## What should be public

The package contains the project report, PPT, notebooks, model checkpoints, split manifest and official-test prediction CSV. The raw ICBHI WAV dataset is intentionally **not** included. Do not upload the raw audio dataset unless its distribution terms explicitly allow you to do so.

## Important project wording

The repository README intentionally states that the current evidence supports laptop-based ML inference and that embedded ESP32 inference and real-hardware ANC measurements remain future validation steps. Keep that wording in your public repository rather than claiming clinical diagnosis or completed on-device deployment.

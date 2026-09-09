# Manual GitHub connection

Run these commands in PowerShell after creating an empty repository on GitHub. Replace `<USER>` and `<REPO>` with the account and repository name. The commands do not overwrite an existing remote.

```powershell
cd F:\CCFA\new_kbs_idea_20260907
git init
git add .
git commit -m "Initial StableKG KBS submission package"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

For later updates:

```powershell
git add .
git commit -m "Update manuscript and experiments"
git push
```

Check the connection without changing files:

```powershell
git remote -v
git status
```

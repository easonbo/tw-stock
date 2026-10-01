# 台股上市櫃 技術選股網頁

每個交易日 15:30（台灣時間）由 GitHub Actions 自動抓最新股價、計算指標，並更新網頁。
不用安裝任何軟體，也不用再跑 R。

## 一次性設定（約 10 分鐘）

1. 登入 GitHub → 右上角 **+** → **New repository**
   - Repository name：`tw-stock`（可自訂）
   - 選 **Public**（免費帳號的 GitHub Pages 需要公開 repo）→ **Create repository**
2. 在新 repo 頁面點 **uploading an existing file**，把以下檔案拖進去 → **Commit changes**
   - `update.py`、`requirements.txt`、`README.md`、`.gitignore`、整個 `site` 資料夾
3. 建立自動更新設定檔（Mac 會隱藏 `.github` 資料夾，所以用網頁建立）：
   - **Add file → Create new file**
   - 檔名欄位輸入：`.github/workflows/update.yml`
   - 內容：貼上壓縮檔裡 `.github/workflows/update.yml` 的全部內容 → **Commit changes**
4. **Settings → Pages → Build and deployment → Source** 選 **GitHub Actions**
5. **Actions** 分頁 → 左側「每日更新台股資料」→ **Run workflow**
   - 第一次要完整下載約 1,800 檔，約 10–25 分鐘；之後每天只補最近資料，幾分鐘內完成
6. 完成後網址：`https://<你的帳號>.github.io/tw-stock/`

## 日常使用

- 自動：週一～五 15:30 後自動更新（GitHub 排程常延遲 10–30 分鐘）
- 手動：Actions → 每日更新台股資料 → Run workflow（手機 GitHub App 也能按）
- 網頁上方會顯示「資料日期」與「更新於」時間

## 注意

- 公開 repo 超過 60 天沒有任何 commit，GitHub 會暫停排程並寄信通知，到 Actions 頁面按 Enable 即可。
- 價量來自 Yahoo Finance；若某天 Actions 失敗（紅色 ✗），點進去看錯誤訊息。網站會保留前一次的資料，不會變空白。
- 策略選股是規則篩選的候選名單，不是投資建議；當沖前請確認可現股當沖、非處置／注意股。

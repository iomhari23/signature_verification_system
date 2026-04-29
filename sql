What gets saved to MySQL
Every time your partner calls /verify, this row is inserted automatically:
ColumnExampleida3f9bc12-... (unique ID)checked_at2026-04-29 10:32:15ref_filenamejohn_genuine.pngqry_filenameunknown_sign.jpgverdictGENUINEdistance0.2341threshold0.50similarity_pct76.59confidenceHighprocessing_ms134

Two endpoints your partner needs
Verify (saves to DB automatically):
bashcurl -X POST http://YOUR_IP:8000/verify \
  -F "reference=@sign1.png" \
  -F "query=@sign2.png"
View all past checks:
bashcurl http://YOUR_IP:8000/logs

# Filter only forged ones:
curl http://YOUR_IP:8000/logs?verdict=FORGED

Quick setup
bashpip install -r requirements.txt
cp .env.example .env        # fill in your MySQL password
uvicorn api:app --host 0.0.0.0 --port 8000

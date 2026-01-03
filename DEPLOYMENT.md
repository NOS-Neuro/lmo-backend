# Deployment Guide for VizAI Backend

## ⚠️ Important: Timeout Configuration

VizAI scans can take 60-180 seconds to complete due to:
- Main business scan: 4 questions × ~15-20 seconds = 60-80 seconds
- Competitor scans (parallel): 3 competitors × ~15-20 seconds = 45-60 seconds (parallel)
- **Total: 105-140 seconds per request**

### Render.com Configuration

**CRITICAL**: Set the following environment variables in Render:

```bash
# API Timeout (increased from 45s to 90s)
PERPLEXITY_TIMEOUT=90

# Uvicorn timeout (must be higher than total scan time)
UVICORN_TIMEOUT_KEEP_ALIVE=180
```

### Render Service Settings

In your Render dashboard:

1. **Go to**: Your service → Settings → Environment
2. **Add** these environment variables:
   - `PERPLEXITY_TIMEOUT=90`
   - `WEB_CONCURRENCY=2` (limit concurrent requests)

3. **Go to**: Settings → Build & Deploy
4. **Start Command**: Update to include timeout:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 180
   ```

### Alternative: Create Procfile

Create a `Procfile` in the repository root:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 180 --timeout-graceful-shutdown 30
```

### Testing Timeout Configuration

After deploying, test with:

```bash
# Should complete without timeout
curl -X POST https://your-app.onrender.com/run_scan \
  -H "Content-Type: application/json" \
  -d '{
    "businessName": "Test Company",
    "website": "https://example.com",
    "contactEmail": "test@example.com",
    "competitors": [
      {"name": "Competitor 1", "website": "https://competitor1.com"},
      {"name": "Competitor 2", "website": "https://competitor2.com"}
    ]
  }'
```

Expected response time: 90-140 seconds (should not timeout)

---

## Performance Settings

### Parallel Competitor Scanning

Control parallelism via environment variable:

```bash
# Default: 3 workers (recommended)
MAX_COMPETITOR_SCAN_WORKERS=3

# For faster scans (may hit rate limits):
MAX_COMPETITOR_SCAN_WORKERS=5

# For conservative API usage:
MAX_COMPETITOR_SCAN_WORKERS=2
```

### Database Connection Pooling

```bash
DATABASE_URL=postgresql://user:pass@host:5432/db
DB_POOL_MIN=1
DB_POOL_MAX=10
```

---

## Troubleshooting

### "Request timed out" errors

**Symptom**: Scans fail with timeout after 60-120 seconds

**Solution**:
1. Check Render environment variables include `PERPLEXITY_TIMEOUT=90`
2. Verify Start Command includes `--timeout-keep-alive 180`
3. Check Render logs for actual timeout value

### "429 Too Many Requests" from Perplexity

**Symptom**: Rate limiting errors from Perplexity API

**Solution**:
1. Reduce `MAX_COMPETITOR_SCAN_WORKERS` to 2
2. Add delays between competitor scans
3. Check Perplexity API rate limits

### Slow response times

**Symptom**: Scans take longer than expected

**Check**:
1. Perplexity API latency (normal: 10-20s per question)
2. Number of competitors (each adds ~60s total)
3. Network latency to Perplexity

---

## Monitoring

### Request Tracing

All logs include request IDs for debugging:

```bash
# Find all logs for a specific request
grep "req:a1b2c3d4" logs.txt

# Find all logs for a specific scan
grep "scan:e5f6g7h8" logs.txt
```

### Health Check

```bash
curl https://your-app.onrender.com/health
```

Expected response includes:
- Database status
- API provider status
- System identity

---

## Production Checklist

Before deploying to production:

- [ ] Set `PERPLEXITY_TIMEOUT=90` in environment
- [ ] Configure uvicorn with `--timeout-keep-alive 180`
- [ ] Set `MAX_COMPETITOR_SCAN_WORKERS=3` (or lower for conservative usage)
- [ ] Configure `DATABASE_URL` for persistent storage
- [ ] Set `FRONTEND_ORIGIN` to actual frontend URL (not wildcard)
- [ ] Set `LOG_LEVEL=INFO` for production logging
- [ ] Test with sample scan including competitors
- [ ] Monitor first few production scans for timeout issues

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PERPLEXITY_API_KEY` | (required) | Perplexity API key |
| `PERPLEXITY_TIMEOUT` | 90 | API timeout in seconds (10-300) |
| `PERPLEXITY_MODEL` | sonar-pro | Perplexity model name |
| `MAX_COMPETITOR_SCAN_WORKERS` | 3 | Parallel workers for competitor scans (1-10) |
| `DATABASE_URL` | None | PostgreSQL connection string |
| `FRONTEND_ORIGIN` | http://localhost:3000 | CORS allowed origin |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## Support

For deployment issues:
1. Check Render logs for specific error messages
2. Verify all environment variables are set
3. Test with `/health` endpoint first
4. Enable DEBUG logging temporarily for detailed diagnostics

**Deployment verified**: 2026-01-01
**Last updated**: 2026-01-01

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VOICE_STUDIO_DATA_DIR=/data VOICE_STUDIO_HOST=0.0.0.0 PORT=7860
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 studio \
    && mkdir /data && chown studio:studio /data
COPY requirements.lock.txt /app/
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY --chown=studio:studio app.py /app/
COPY --chown=studio:studio studio/ /app/studio/
COPY --chown=studio:studio scripts/download_models.py /app/scripts/download_models.py
USER studio
EXPOSE 7860
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/',timeout=4)"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "app.py", "--no-browser"]

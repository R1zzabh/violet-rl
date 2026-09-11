# docker/frontend/Dockerfile — VIOLET

FROM node:20-slim AS base

WORKDIR /app

# ---- deps ----
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --only=production 2>/dev/null || npm ci

# ---- build ----
COPY frontend .
RUN npm run build

# ---- serve ----
ENV NODE_ENV=production
EXPOSE 3000

# non-root for safety
RUN addgroup --system violet \
    && adduser --system --ingroup violet violet \
    && chown -R violet:violet /app
USER violet

CMD ["npm", "run", "start"]

# Pi Agent sidecar — Node รุ่นที่ pi 1.2 ต้องการ (>= 22.19)
FROM node:22.23.3-slim

ENV NODE_ENV=production
WORKDIR /app

# ติดตั้งตาม lockfile เท่านั้น + --ignore-scripts ตามที่ pi.dev แนะนำ (กัน postinstall script ของ dependency)
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund

COPY server.mjs ./

USER node
EXPOSE 7070
CMD ["node", "server.mjs"]

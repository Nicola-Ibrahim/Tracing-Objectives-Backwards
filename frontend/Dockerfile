# ============================================================
#  Stage 0 — base: Shared OS setup and environment
# ============================================================
FROM node:20-alpine AS base

WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

# ============================================================
#  Stage 1 — deps: install production dependencies
# ============================================================
FROM base AS deps
RUN apk add --no-cache libc6-compat

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm install --legacy-peer-deps && \
    npm cache clean --force

# ============================================================
#  Stage 2 — dev: hot-reload development server
# ============================================================
FROM deps AS dev
EXPOSE 3000
CMD ["npm", "run", "dev"]

# ============================================================
#  Stage 3 — builder: compile production bundle
# ============================================================
FROM deps AS builder
COPY . .
RUN npm run build

# ============================================================
#  Stage 4 — runner: minimal production image
# ============================================================
FROM base AS runner

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid 1001 nextjs

# Automatically leverage output traces to reduce image size
# https://nextjs.org/docs/advanced-features/output-file-tracing
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

# server.js is created by next build from the standalone output
CMD ["node", "server.js"]

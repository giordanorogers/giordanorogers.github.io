# giorogers.com Astro Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the academicpages Jekyll site with a bespoke Astro v5 site (dark Lattice aesthetic, Essays + About only) deployed to GitHub Pages at giorogers.com.

**Architecture:** Static Astro site in this repo. Essays are a content collection; styling is one hand-written CSS file of Lattice design tokens; the only client JS is a theme toggle. GitHub Actions builds and deploys to Pages; the custom domain is claimed via the Pages API (DNS already points at GitHub).

**Tech Stack:** Astro ^5, @astrojs/rss, @fontsource/sora, GitHub Actions (withastro/action@v3 + actions/deploy-pages@v4).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-02-website-redesign-design.md` — tokens and copy are verbatim from there.
- Dark default: bg `#000`, text `#eeeeea`, strong `#fff`, muted `rgba(238,238,234,.58)`, border `rgba(238,238,234,.22)`, glow `rgba(113,218,255,.42)`.
- Light theme: bg `#fff`, text `#1a1a18`, strong `#111`, muted `rgba(17,17,17,.58)`, border `rgba(17,17,17,.2)`.
- Type: Georgia serif for titles/prose; Sora (400/600) for UI; `ui-monospace` 11px uppercase `letter-spacing:.08em` for metadata. Masthead "Gio Rogers" is regular weight.
- About copy verbatim: "I work on identifying, understanding, and mitigating vulnerabilities in AI assistants and agents to make them safer." and "All views on this site are my own and do not reflect those of any other organization or person."
- Links ONLY on /about: LinkedIn, Google Scholar, Email. No GitHub link anywhere. Footer = single RSS link.
- Site title/brand: "Gio Rogers". Site URL: `https://giorogers.com`.
- GitHub repo is `giordanorogers/giordanorogers.github.io` (local remote name is stale). Live Pages build_type is currently `legacy`; DNS A records for giorogers.com already point to GitHub Pages; `www` CNAME → giordanorogers.github.io.
- All work on branch `astro-rebuild`, merged to `master` at the end (Pages deploys from master).

---

### Task 1: Branch + remote hygiene

**Files:** none (git only)

- [ ] `cd ~/Documents/Documents/Code/giorogers.github.io`
- [ ] `git remote set-url origin https://github.com/giordanorogers/giordanorogers.github.io.git`
- [ ] `git checkout -b astro-rebuild`
- [ ] `git rm -r --cached . -q 2>/dev/null || true` — NOT needed; skip. (No step.)

### Task 2: Remove the Jekyll site and experiments

**Files:** Delete: `_config.yml _data _drafts _includes _layouts _pages _portfolio _publications _sass _site _talks _teaching assets markdown_generator study test openclaw-security Gemfile Gemfile.lock Dockerfile docker-compose.yaml CONTRIBUTING.md LICENSE README.md talkmap.ipynb talkmap_out.ipynb files .github/workflows/scrape_talks.yml` plus any stray root files not needed. KEEP for now: `_posts/`, `images/` (migration sources, removed in Task 5), `docs/`, `.git`.

- [ ] **Step 1:** `git rm -rq _config.yml _data _drafts _includes _layouts _pages _portfolio _publications _sass _talks _teaching assets markdown_generator study test openclaw-security Gemfile Gemfile.lock Dockerfile docker-compose.yaml CONTRIBUTING.md LICENSE README.md files` (add `talkmap*.ipynb`, `.github/`, `_site` if tracked; `ls -la` after to catch strays)
- [ ] **Step 2:** Write fresh `.gitignore`:

```
node_modules/
dist/
.astro/
.DS_Store
```

- [ ] **Step 3:** `git add -A && git commit -m "Remove Jekyll site, placeholder content, and experiment directories"`

### Task 3: Scaffold Astro

**Files:** Create: `package.json`, `astro.config.mjs`, `tsconfig.json`, `src/pages/index.astro` (placeholder), `src/styles/global.css` (empty for now)

**Interfaces — Produces:** npm scripts `dev`/`build`/`preview`; `astro.config.mjs` exporting `site: 'https://giorogers.com'`, redirects, shiki dual themes.

- [ ] **Step 1:** Write `package.json`:

```json
{
  "name": "giorogers.com",
  "type": "module",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  }
}
```

- [ ] **Step 2:** `npm install astro@latest @astrojs/rss@latest @fontsource/sora@latest`
- [ ] **Step 3:** Write `astro.config.mjs`:

```js
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://giorogers.com',
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'vesper' },
      defaultColor: false,
    },
  },
  redirects: {
    '/posts/2025/07/activation_patching_residual/': '/essays/activation-patching-the-residual-stream/',
    '/posts/2025/08/what_is_creativity/': '/essays/on-creativity/',
    '/posts/2999/11/research_taste/': '/essays/research-taste/',
    '/posts/2025/12/ai_mediated_mediocrity/': '/essays/ai-mediated-mediocrity/',
  },
});
```

- [ ] **Step 4:** Write `tsconfig.json`:

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 5:** Placeholder `src/pages/index.astro` (`<h1>ok</h1>`), then `npm run build` — Expected: build succeeds, `dist/index.html` exists, redirect pages generated under `dist/posts/...`.
- [ ] **Step 6:** Commit: `git add -A && git commit -m "Scaffold Astro 5"`

### Task 4: Design system, layout, and chrome

**Files:**
- Create: `src/styles/global.css`, `src/layouts/Base.astro`, `src/components/Constellation.astro`, `public/favicon.svg`
- Modify: `src/pages/index.astro` (real homepage in Task 5 consumes Base)

**Interfaces — Produces:** `Base.astro` props: `{ title: string; description: string; home?: boolean }`. Named slot default = page content. Header renders name+constellation when `home`, else mono "← Essays" link to `/`. Always: About link (only when `home`), theme toggle button `#theme-toggle`. Footer: RSS link only.

- [ ] **Step 1:** Write `src/styles/global.css` (complete file):

```css
:root {
  --bg: #000;
  --text: #eeeeea;
  --text-strong: #fff;
  --muted: rgba(238, 238, 234, 0.58);
  --border: rgba(238, 238, 234, 0.22);
  --border-strong: rgba(238, 238, 234, 0.54);
  --border-dotted: rgba(238, 238, 234, 0.16);
  --glow: rgba(113, 218, 255, 0.42);
  --halo: rgba(113, 218, 255, 0.18);
  --core: rgba(204, 239, 255, 0.9);
  --code-bg: #0d0d0c;
  --serif: Georgia, 'Times New Roman', serif;
  --sans: 'Sora', ui-sans-serif, system-ui, -apple-system, sans-serif;
  --mono: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
}
[data-theme='light'] {
  --bg: #fff;
  --text: #1a1a18;
  --text-strong: #111;
  --muted: rgba(17, 17, 17, 0.58);
  --border: rgba(17, 17, 17, 0.2);
  --border-strong: rgba(17, 17, 17, 0.44);
  --border-dotted: rgba(17, 17, 17, 0.16);
  --glow: rgba(0, 140, 200, 0.3);
  --halo: rgba(0, 140, 200, 0.12);
  --core: rgba(0, 110, 170, 0.75);
  --code-bg: #f6f5f0;
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0 auto;
  padding: 56px 24px 40px;
  max-width: 700px;
  background: var(--bg);
  color: var(--text);
  font-family: var(--serif);
  font-size: 1.02rem;
  line-height: 1.68;
  transition: background 180ms ease, color 180ms ease;
}
a { color: inherit; text-decoration: none; }
.site-header { display: flex; align-items: center; justify-content: space-between; }
.site-name { font-family: var(--serif); font-size: 1.65rem; font-weight: 400; color: var(--text-strong); letter-spacing: 0.01em; }
.header-nav { display: flex; align-items: center; gap: 18px; }
.mono-label, .mono-link {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted);
}
a.mono-link:hover { color: var(--text); text-shadow: 0 0 14px var(--glow); }
#theme-toggle {
  background: transparent; border: 1px solid var(--border); border-radius: 999px;
  width: 34px; height: 34px; padding: 0; color: var(--muted); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
}
#theme-toggle:hover { border-color: var(--border-strong); color: var(--text); }
#theme-toggle .moon { display: none; }
[data-theme='light'] #theme-toggle .sun { display: none; }
[data-theme='light'] #theme-toggle .moon { display: block; }
.essay-list { list-style: none; margin: 0; padding: 0; }
.essay-list li { border-bottom: 1px dotted var(--border-dotted); }
.essay-list li:last-child { border-bottom: none; }
.essay-link {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 16px; padding: 16px 0;
}
.essay-link .title { font-size: 1.19rem; transition: text-shadow 180ms ease; }
.essay-link:hover .title { text-shadow: 0 0 16px var(--glow); }
.essay-link .date { font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; color: var(--muted); white-space: nowrap; }
.section-label { margin: 64px 0 6px; font-weight: 600; }
.site-footer {
  display: flex; justify-content: flex-end; border-top: 1px solid var(--border);
  margin-top: 72px; padding-top: 18px;
}
.prose h1 {
  font-size: clamp(2rem, 6vw, 2.9rem); line-height: 1.04; font-weight: 700;
  color: var(--text-strong); margin: 0.4em 0 0.7em;
}
.prose h2, .prose h3 { color: var(--text-strong); line-height: 1.15; margin: 1.8em 0 0.6em; }
.prose h2 { font-size: 1.45rem; }
.prose h3 { font-size: 1.2rem; }
.prose img { max-width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }
.prose blockquote { margin: 1.4em 0; padding-left: 1.1em; border-left: 1px solid var(--border-strong); color: var(--muted); }
.prose code { font-family: var(--mono); font-size: 0.86em; }
.prose :not(pre) > code { background: var(--code-bg); border: 1px solid var(--border-dotted); border-radius: 4px; padding: 0.08em 0.35em; }
.astro-code {
  border: 1px solid var(--border); border-radius: 6px; padding: 16px 18px;
  font-size: 0.8rem; line-height: 1.6; overflow-x: auto;
  background: var(--code-bg) !important;
}
.astro-code, .astro-code span { color: var(--shiki-dark); }
[data-theme='light'] .astro-code, [data-theme='light'] .astro-code span { color: var(--shiki-light); }
.constellation .node { animation: breathe 4.5s ease-in-out infinite alternate; }
@keyframes breathe { from { opacity: 0.35; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .constellation .node { animation: none; } }
```

- [ ] **Step 2:** Write `src/components/Constellation.astro` (the mockup SVG, `class="constellation"`, `aria-hidden="true"`, sized ~96×40, positioned absolutely relative to the masthead name).
- [ ] **Step 3:** Write `src/layouts/Base.astro`: imports `@fontsource/sora/400.css`, `@fontsource/sora/600.css`, `../styles/global.css`; `<head>` with charset/viewport/title/description/canonical, `<link rel="alternate" type="application/rss+xml" ...>`, favicon.svg, inline theme script (`is:inline`, sets `document.documentElement.dataset.theme` from localStorage before paint, default `dark`); header per Interfaces; footer with `<a class="mono-link" href="/rss.xml">RSS</a>`; toggle `<script>` flipping `dataset.theme` + localStorage; sun/moon inline SVGs inside the button.
- [ ] **Step 4:** Write `public/favicon.svg` — 3-node constellation, `<style>@media (prefers-color-scheme: light){...}</style>` for adaptive stroke/fill.
- [ ] **Step 5:** `npm run build` — Expected: PASS. Commit: `git add -A && git commit -m "Add Lattice design system, base layout, favicon"`

### Task 5: Essays collection, migration, homepage, essay pages

**Files:**
- Create: `src/content.config.ts`, `src/lib/format.ts`, `src/content/essays/{activation-patching-the-residual-stream,on-creativity,research-taste,ai-mediated-mediocrity}.md`, `src/pages/essays/[slug].astro`, `public/images/transformer_architecture.png`, `public/images/indirect_effects_heatmap.png`
- Modify: `src/pages/index.astro`
- Delete (after migration): `_posts/`, `images/`

**Interfaces — Produces:** collection `essays`, schema `{ title: string; date: Date; description: string }`; `formatDate(d: Date): string` returning e.g. `DEC 14 2025`; `readingTime(body: string): number` (minutes, words/220, min 1). Essay URLs `/essays/<file-slug>/`.

- [ ] **Step 1:** Write `src/content.config.ts`:

```ts
import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const essays = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/essays' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    description: z.string(),
  }),
});

export const collections = { essays };
```

- [ ] **Step 2:** Write `src/lib/format.ts`:

```ts
export function formatDate(d: Date): string {
  return d
    .toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' })
    .replace(',', '')
    .toUpperCase();
}

export function readingTime(body: string): number {
  return Math.max(1, Math.round(body.trim().split(/\s+/).length / 220));
}
```

- [ ] **Step 3:** Migrate the four posts: copy body verbatim from `_posts/*`, new frontmatter only `title`, `date` (research-taste → `2025-11-26`), and a one-line `description` written from the actual content. `cp images/transformer_architecture.png images/indirect_effects_heatmap.png public/images/` (body image paths `/images/...` keep working).
- [ ] **Step 4:** Write `src/pages/index.astro` — Base with `home`, `.section-label` "Essays", `.essay-list` of all essays sorted date desc, each `.essay-link` (title serif + `formatDate` date).
- [ ] **Step 5:** Write `src/pages/essays/[slug].astro`:

```astro
---
import { getCollection, render } from 'astro:content';
import Base from '../../layouts/Base.astro';
import { formatDate, readingTime } from '../../lib/format';

export async function getStaticPaths() {
  const essays = await getCollection('essays');
  return essays.map((e) => ({ params: { slug: e.id }, props: { essay: e } }));
}
const { essay } = Astro.props;
const { Content } = await render(essay);
---
<Base title={`${essay.data.title} — Gio Rogers`} description={essay.data.description}>
  <article class="prose">
    <div class="mono-label" style="margin-top:48px">
      {formatDate(essay.data.date)} &middot; {readingTime(essay.body ?? '')} min read
    </div>
    <h1>{essay.data.title}</h1>
    <Content />
  </article>
</Base>
```

- [ ] **Step 6:** `git rm -rq _posts images` then `npm run build` — Expected: 4 pages under `dist/essays/`, homepage lists 4 essays. Verify: `ls dist/essays/`.
- [ ] **Step 7:** Commit: `git add -A && git commit -m "Migrate essays to content collection; homepage and essay pages"`

### Task 6: About, RSS, 404

**Files:** Create: `src/pages/about.astro`, `src/pages/rss.xml.js`, `src/pages/404.astro`

- [ ] **Step 1:** Find Giordano's Google Scholar profile URL via web search ("Giordano Rogers Google Scholar Northeastern"). If no confident match, link `https://scholar.google.com/scholar?q=%22Giordano+Rogers%22` and flag in the final report.
- [ ] **Step 2:** Write `src/pages/about.astro` — Base, `<h1>About</h1>`, the two verbatim sentences (second one smaller/muted), then dotted-top-border row of mono links: LinkedIn `https://www.linkedin.com/in/giorogers`, Google Scholar (Step 1 URL), Email `mailto:giordanopaulrogers@gmail.com`.
- [ ] **Step 3:** Write `src/pages/rss.xml.js`:

```js
import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const essays = (await getCollection('essays')).sort((a, b) => b.data.date - a.data.date);
  return rss({
    title: 'Gio Rogers',
    description: 'Essays on AI safety, interpretability, and research.',
    site: context.site,
    items: essays.map((e) => ({
      title: e.data.title,
      pubDate: e.data.date,
      description: e.data.description,
      link: `/essays/${e.id}/`,
    })),
  });
}
```

- [ ] **Step 4:** Write `src/pages/404.astro` — Base, mono-label "404", serif line "This page doesn't exist.", mono-link back to `/`.
- [ ] **Step 5:** `npm run build`; verify `dist/about/index.html`, `dist/rss.xml` (well-formed: `python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('dist/rss.xml')"`), `dist/404.html`.
- [ ] **Step 6:** Commit: `git add -A && git commit -m "Add about page, RSS feed, 404"`

### Task 7: Local visual verification

**Files:** Create: `.claude/launch.json` (dev server config)

- [ ] **Step 1:** `.claude/launch.json` with `{"name":"site","runtimeExecutable":"npm","runtimeArgs":["run","dev"],"port":4321}`; start preview server.
- [ ] **Step 2:** Snapshot + screenshot homepage (dark), toggle to light, screenshot; essay page with code blocks in both themes; about; 404; mobile (375px) homepage + essay. Check console for errors. Fix anything broken; re-verify.
- [ ] **Step 3:** Commit any fixes: `git add -A && git commit -m "Visual polish from preview verification"`

### Task 8: Deploy — workflow, domain, go live

**Files:** Create: `public/CNAME` (content: `giorogers.com`), `.github/workflows/deploy.yml`, `README.md` (3 lines: what the site is, `npm run dev`, deploys via Actions on push to master)

- [ ] **Step 1:** Write `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages
on:
  push:
    branches: [master]
  workflow_dispatch:
permissions:
  contents: read
  pages: write
  id-token: write
concurrency:
  group: pages
  cancel-in-progress: false
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: withastro/action@v3
  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2:** Commit; merge: `git checkout master && git merge astro-rebuild --no-edit`
- [ ] **Step 3:** Switch Pages to workflow builds: `gh api repos/giordanorogers/giordanorogers.github.io/pages -X PUT -f build_type=workflow`
- [ ] **Step 4:** `git push origin master`; `gh run watch` until deploy succeeds.
- [ ] **Step 5:** Claim domain: `gh api repos/giordanorogers/giordanorogers.github.io/pages -X PUT -f cname=giorogers.com`; wait for DNS check + HTTPS cert (poll `gh api .../pages` for `status`/`https_enforced`; cert can take a few minutes).
- [ ] **Step 6:** Post-deploy checks (expect all to pass):
  - `curl -sI https://giorogers.com/` → 200
  - `curl -sI https://giordanorogers.github.io/` → 301 to giorogers.com
  - `curl -sL https://giorogers.com/posts/2025/12/ai_mediated_mediocrity/` → lands on the essay (meta-refresh page or final content)
  - `curl -sI https://giorogers.com/rss.xml` → 200
  - `curl -sI https://giorogers.com/study/` → 404
- [ ] **Step 7:** Final commit of any tweaks; report results.

## Self-Review Notes

- Spec coverage: architecture (T3), structure/URLs+redirects (T3/T5/T6), visual tokens (T4), migration+deletions (T2/T5), copy (T6), error handling (T6 404, T3 redirects), testing (T5–T8). Covered.
- Old `www.giorogers.com` CNAME already points at giordanorogers.github.io — works once apex domain is claimed.
- Email choice (personal gmail over expiring Northeastern address) and Scholar URL are flagged for Giordano in the final report.

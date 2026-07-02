# giorogers.com Redesign — Design Spec

**Date:** 2026-07-02
**Status:** Approved pending user review
**Repo:** `giorogers.github.io` (this repo)

## Goal

Replace the stock academicpages Jekyll site with a bespoke, writer-first personal site that carries Giordano's Lattice design language. The site is his core digital footprint: it should present him as a thinker/researcher/writer and make publishing essays frictionless. Success looks like: a stranger lands on giorogers.com, reads an essay, and comes away with an accurate sense of Giordano's taste and thinking — with zero template residue.

## Decisions (settled with Giordano)

| Decision | Choice |
|---|---|
| Stack | Astro v5, custom CSS, no UI framework |
| Aesthetic | Dark-first Lattice language, light mode behind a toggle |
| Homepage | Writer-first: short identity paragraph, then essays |
| Affiliation | Ideas-first; OpenAI mentioned only on /about |
| Domain | `giorogers.com` (already owned) |
| Sections | Essays + About only, plus a quiet RSS feed |
| Old experiments | `study/`, `test/`, `openclaw-security/` removed from the site |

## Architecture

- **Astro v5** static site in this repo, replacing all Jekyll machinery. Content collection `essays` holds Markdown/MDX with schema-validated frontmatter (`title`, `date`, `description`, optional `draft`).
- **Styling:** one hand-written global stylesheet plus per-component styles, driven by CSS custom properties (the Lattice tokens). No Tailwind, no component library.
- **Client JS:** theme toggle only (a few lines, inlined). Everything else ships as static HTML/CSS.
- **Code highlighting:** Astro's built-in Shiki with dual themes (dark default, light variant) consistent with the palette.
- **Math:** none of the current essays need it; add remark-math/KaTeX only when a future essay does (build-time, still zero JS).
- **Deploy:** GitHub Actions builds Astro and publishes to GitHub Pages. `public/CNAME` contains `giorogers.com`. Giordano configures the custom domain in repo settings and points DNS (apex A/ALIAS records + `www` CNAME per GitHub docs) — manual step, his account. GitHub then 301-redirects all `giorogers.github.io/*` links to the same paths on `giorogers.com`.

## Site structure & URLs

| Path | Content |
|---|---|
| `/` | Name, one-paragraph identity statement (ideas-first, no employer), full essay list reverse-chron with uppercase mono date metadata. Footer: GitHub, LinkedIn, email, RSS. |
| `/essays/<slug>/` | Essay page: large Georgia serif title, mono metadata line (date · reading time), ~65ch reading measure. |
| `/about/` | Fuller bio; the only place OpenAI appears. Contact links. |
| `/rss.xml` | Full-content RSS feed via `@astrojs/rss`; linked only in the footer. |
| `/404.html` | Styled 404 in the site's voice. |

**Redirect map** (old Jekyll permalinks → new slugs), implemented with Astro's `redirects` config (generates static meta-refresh pages):

| Old | New |
|---|---|
| `/posts/2025/07/activation_patching_residual/` | `/essays/activation-patching-the-residual-stream/` |
| `/posts/2025/08/what_is_creativity/` | `/essays/on-creativity/` |
| `/posts/2999/11/research_taste/` | `/essays/research-taste/` |
| `/posts/2025/12/ai_mediated_mediocrity/` | `/essays/ai-mediated-mediocrity/` |

## Visual design

Tokens ported from Lattice (`insight-loom-mvp`), as CSS custom properties on `:root` / `[data-theme="light"]`:

- **Dark (default):** background `#000`; body text `#eeeeea`; strong text `#fff`; muted `rgba(238,238,234,0.58)`; hairline borders `rgba(238,238,234,0.22)`, strong `rgba(238,238,234,0.54)`.
- **Light:** background `#fff`; text `#111`; muted `rgba(17,17,17,0.58)`; borders `rgba(17,17,17,0.2)`.
- **Accent:** Lattice cyan glow (`rgba(113,218,255,0.42)` family) used only for link hover glow and the masthead motif. No other color.
- **Type:** Georgia serif for titles/headings (essay titles `clamp(2rem, 4vw, 3.65rem)`, line-height ≈1.02, weight 700–800) and essay body (~1.05rem, line-height 1.6). Self-hosted **Sora** (woff2, subset, weights 400/600) for nav, UI labels, homepage identity paragraph. `ui-monospace` system stack for metadata: 0.7rem, uppercase, letter-spacing 0.08em.
- **Motif:** one restrained constellation touch — a faint breathing node-glow in the masthead and a soft glow on essay-link hover, 180ms ease transitions, fully disabled under `prefers-reduced-motion`.
- **Theme toggle:** button in the header; inline `<head>` script reads `localStorage` and sets `data-theme` before paint (no flash). Defaults to dark.
- **Density:** airy — content max-width ~700px for prose, generous section gaps, hairline dividers (dotted between essay list items, per Lattice).

## Content migration

**Migrate** the four real essays into the `essays` collection with cleaned frontmatter:

1. *Activation Patching the Residual Stream* (2025-07-19) — verify code blocks and any referenced images/PDF render; keep `files/activation_patching_tutorial_residual.pdf` only if the essay links to it.
2. *On creativity* (2025-08-18)
3. *Research Taste* — fix date to 2025-11-26 (frontmatter currently says 2999).
4. *AI Mediated Mediocrity* (2025-12-14)

Tags are dropped from all essays during migration — nothing in the new design renders them.

**Delete** (git history preserves everything): all Jekyll machinery (`_config.yml`, `_sass/`, `_includes/`, `_layouts/`, `_data/`, `Gemfile*`, `Dockerfile`, `docker-compose.yaml`, `assets/`, `_site/`); all placeholder collections (`_publications/`, `_talks/`, `_teaching/`, `_portfolio/`); template posts (`blog-post-template`, `reading_list.md` stub, `first_five.d`, `_drafts/`); generator tooling (`markdown_generator/`, `talkmap*.ipynb`, `scrape_talks` workflow); experiments (`study/`, `test/`, `openclaw-security/`); unreferenced `images/` and `files/` assets.

**Rewrite** (drafted during implementation, user approves copy before deploy): homepage identity paragraph and the /about page.

**Out of scope:** newsletter/email capture, analytics, Research/Projects page, Notes stream, interactive essay components, Lattice public page. Any of these can be added later without structural change.

## Error handling

- Styled `404.html` (GitHub Pages serves it automatically for unknown paths).
- Redirect pages carry `<link rel="canonical">` to the new URL.
- Build fails loudly on invalid essay frontmatter (content collection schema).
- Fonts load with `font-display: swap`; system-font fallbacks keep the site readable if Sora fails.

## Testing & verification

- `astro build` in CI on every push; deploy only on green build from `master`.
- Local verification before deploy: dev-server screenshots of homepage + one essay in dark, light, and mobile (375px) widths; check console for errors.
- Internal link check across built output (including the four redirect paths).
- Post-deploy: confirm `giorogers.com` serves, `giorogers.github.io` redirects, old post URLs land on the right essays, `/rss.xml` validates.

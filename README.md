# giorogers.com

Personal site of Gio Rogers — essays, notes, and tutorials on AI safety, interpretability, research, and art. Built with Astro and Markdown; deployed to GitHub Pages.

## Run locally

Use Node.js 22, matching the deployment workflow, and npm.

```sh
npm ci
npm run dev
```

Open the local URL printed by Astro. Development mode includes a clearly marked preview section for unpublished work on the homepage, and unpublished article pages show a preview notice. RSS and the sitemap always use the public publication rules, even during development.

To inspect the exact production output:

```sh
npm test
npm run build
npm run check
npm run preview
```

`build` writes the static site to `dist/`; `preview` serves that build. Drafts and future-dated posts are absent from this production preview. Restart the dev server after changing site configuration; normal Markdown and style edits update automatically.

`check` inspects the existing production build for broken local links and fragments, canonical and social metadata, the custom domain, and publication visibility across article routes, homepage, recommendations, RSS, and sitemap. Run it after `build`. For an additional end-to-end publication check, run `npm run check:publication`; it creates four temporary featured draft/future fixtures and a published Unicode-title control, builds and checks them, then removes them and rebuilds the normal site even if a check fails. This command replaces `dist/`, so finish any production-preview inspection first.

## Write the next post

From the repository root:

```sh
npm run new -- "My post title"
npm run new -- "A short observation" --note
npm run new -- "A practical walkthrough" --tutorial
```

The default format is `essay`. The command creates `src/content/essays/my-post-title.md` with today's local calendar date and `draft: true`. Titles must be quoted as a single argument. Choose at most one format flag. Filenames use lowercase words separated by hyphens, normalize accents, and retain Unicode characters supported by Astro's post IDs. Unsupported characters are removed; a title that leaves no usable letters or numbers is rejected. Invalid titles, unsupported flags, and duplicate post IDs (including explicit frontmatter slugs) fail without overwriting an existing file.

Edit the generated Markdown file, fill in its description, and replace the starter text with your post. The layout supplies the title as the page heading, so start the body with an introduction or a `##` subheading. Preview with `npm run dev`. When ready, set `draft: false`, confirm the publication date, then run the tests and production build before opening a pull request.

## Post metadata

```yaml
---
title: "My post title"
date: 2026-10-24
description: "A short, concrete summary for the post list and RSS."
draft: true
featured: false
tags: ["interpretability", "research"]
format: essay
archived: false
# updated: 2026-10-25
---
```

| Field | Meaning |
| --- | --- |
| `title` | Display title. Quote titles containing YAML punctuation such as `:` or `#`. The generator quotes titles safely. |
| `date` | Original publication date in `YYYY-MM-DD` form. Keep this date when revising an existing post. |
| `description` | Brief summary. The generator leaves it empty so you can write it before publishing. |
| `draft` | `true` keeps the post out of production routes and homepage lists, plus RSS and sitemap in every mode. The generator sets it to `true`; omitted values in existing posts default to `false`. |
| `featured` | `true` includes a public, nonarchived post in the homepage's selected writing. Defaults to `false`. |
| `tags` | A list of free-form topic labels, such as `interpretability` or `art`. Defaults to `[]`. |
| `format` | `essay`, `note`, or `tutorial`. Defaults to `essay`. |
| `archived` | `true` removes a published post from the homepage, RSS, and sitemap while keeping its direct URL available with an archive notice and `noindex` for search engines. Defaults to `false`. |
| `updated` | Optional date of a substantive revision, shown separately from the publication date. Uncomment and change it when appropriate. |

Dates without a time are interpreted as midnight UTC. A post is publishable only when `draft` is false and its date is no later than the time of the build. A future date does **not** schedule a deployment: after that date arrives, another build and deployment are required. There is no scheduler in this repository.

Draft status controls the generated website, not access to the Markdown. This is a public repository: a draft pushed to GitHub can still be read in the source and commit history. Keep private writing outside the public repository until you are ready to share it.

## Review and publish

Work on a branch and use a pull request for review:

```sh
git switch -c writing/my-post-title
npm run new -- "My post title"
# Write, preview, and set draft: false when ready.
npm test
npm run build
npm run check
git add src/content/essays/my-post-title.md
git commit -m "Add My post title"
git push -u origin writing/my-post-title
```

Open a pull request into `master` and review the content and production preview. Merge when ready to publish. Pull requests into `master` run the production build, unit tests, and generated-site checks. The workflow in `.github/workflows/deploy.yml` builds, checks, and deploys automatically on pushes to `master`; it can also be run manually from GitHub Actions. Pull request runs do not deploy. After merging, confirm that the deployment succeeds and inspect the live article.

Preserve `public/CNAME` with `giorogers.com`, the `site` URL in `astro.config.mjs`, and the existing redirect mappings. These keep the custom domain, canonical links, feed URLs, and older article links working. Keep existing Markdown filenames stable when editing titles so shared post URLs continue to resolve.

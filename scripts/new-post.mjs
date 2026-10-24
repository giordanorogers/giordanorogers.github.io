import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseFrontmatter } from '@astrojs/internal-helpers/frontmatter';
import { slug as githubSlug } from 'github-slugger';

const defaultContentDir = fileURLToPath(new URL('../src/content/essays/', import.meta.url));
const formats = new Set(['essay', 'note', 'tutorial']);
const usage = 'Usage: npm run new -- "My post title" [--note | --tutorial]';

function normalizeTitle(title) {
  if (typeof title !== 'string' || !title.trim()) {
    throw new Error('A nonempty post title is required.');
  }
  if (/[\u0000-\u001f\u007f-\u009f\u2028\u2029]/u.test(title)) {
    throw new Error('The title must be one line without control characters.');
  }
  return title.trim();
}

function filenameWords(value) {
  return value.normalize('NFKD').replace(/\p{M}/gu, '').toLowerCase();
}

function safeSlug(value) {
  const words = filenameWords(value)
    .replace(/[’'“”"]/gu, '')
    .replace(/[^\p{L}\p{N}]+/gu, '-');
  // Match Astro's own slugger: some Unicode letters/numbers are not valid IDs.
  return githubSlug(words)
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export function slugTitle(title) {
  const slug = safeSlug(normalizeTitle(title));
  if (!slug) throw new Error('The title must contain at least one letter or number.');
  if (Buffer.byteLength(slug, 'utf8') > 200) {
    throw new Error('The title is too long for a safe filename; use a shorter title.');
  }
  return slug;
}

export function parseArgs(args) {
  let format = 'essay';
  let formatChosen = false;
  const titles = [];
  for (const argument of args) {
    if (argument.startsWith('-')) {
      if (!['--note', '--tutorial'].includes(argument)) {
        throw new Error(`Unsupported option: ${argument}`);
      }
      if (formatChosen) throw new Error('Choose only one format option: --note or --tutorial.');
      format = argument.slice(2);
      formatChosen = true;
    } else {
      titles.push(argument);
    }
  }
  if (titles.length !== 1) {
    throw new Error('Supply exactly one title, enclosed in quotes.');
  }
  const title = normalizeTitle(titles[0]);
  slugTitle(title);
  return { title, format };
}

async function findExistingPost(directory, slug, prefix = '') {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relativeName = prefix ? `${prefix}/${entry.name}` : entry.name;
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      const collision = await findExistingPost(fullPath, slug, relativeName);
      if (collision) return collision;
    } else if (/\.md$/i.test(entry.name)) {
      const stem = relativeName.replace(/\.md$/i, '');
      // Be conservative about equivalent names. Astro lowercases post IDs and
      // removes punctuation; our filename convention also joins word separators.
      const aliases = [
        stem.split('/').map(safeSlug).join('/').replace(/\/index$/, ''),
        stem.split('/').map((segment) => githubSlug(segment)).join('/').replace(/\/index$/, ''),
      ];
      if (aliases.includes(slug)) return relativeName;
      // Astro's glob loader gives an explicit frontmatter slug precedence over
      // its generated filename ID. Use its installed parser for the same value.
      const { frontmatter } = parseFrontmatter(await readFile(fullPath, 'utf8'));
      if (frontmatter.slug === slug) return relativeName;
    }
  }
  return undefined;
}

export async function createPost({ title, format = 'essay' }, { contentDir = defaultContentDir, now = new Date() } = {}) {
  title = normalizeTitle(title);
  const slug = slugTitle(title);
  if (!formats.has(format)) throw new Error('Unsupported format; choose essay, note, or tutorial.');
  if (!(now instanceof Date) || Number.isNaN(now.valueOf())) throw new Error('A valid date is required.');
  const date = [now.getFullYear(), String(now.getMonth() + 1).padStart(2, '0'), String(now.getDate()).padStart(2, '0')].join('-');
  const directory = path.resolve(contentDir);
  const destination = path.join(directory, `${slug}.md`);
  const markdown = `---
title: ${JSON.stringify(title)}
date: ${date}
description: ""
draft: true
featured: false
tags: []
format: ${format}
archived: false
# updated: ${date}
---

Write your post here.
`;

  await mkdir(directory, { recursive: true });
  const existing = await findExistingPost(directory, slug);
  if (existing) throw new Error(`A post with an equivalent ID already exists: ${existing}`);
  try {
    // Exclusive creation also prevents races and refuses existing symlinks.
    await writeFile(destination, markdown, { encoding: 'utf8', flag: 'wx' });
  } catch (error) {
    if (error.code === 'EEXIST') throw new Error(`A post already exists: ${slug}.md`);
    throw error;
  }
  return destination;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const destination = await createPost(parseArgs(process.argv.slice(2)));
    console.log(`Created draft: ${path.relative(process.cwd(), destination)}`);
    console.log('Run npm run dev to preview. Set draft: false when ready to publish.');
  } catch (error) {
    console.error(`Cannot create post: ${error.message}\n${usage}`);
    process.exitCode = 1;
  }
}

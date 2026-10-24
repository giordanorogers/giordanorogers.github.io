import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { parseFrontmatter } from '@astrojs/internal-helpers/frontmatter';
import { slug as githubSlug } from 'github-slugger';
import { createPost, parseArgs, slugTitle } from '../scripts/new-post.mjs';

const today = new Date(2026, 9, 24, 13);

async function temporaryDirectory(t) {
  const directory = await mkdtemp(path.join(tmpdir(), 'giorogers-post-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test('parses one quoted title and an optional format flag in either order', () => {
  assert.deepEqual(parseArgs(['My post title']), { title: 'My post title', format: 'essay' });
  assert.deepEqual(parseArgs(['My post title', '--note']), { title: 'My post title', format: 'note' });
  assert.deepEqual(parseArgs(['--tutorial', 'My post title']), { title: 'My post title', format: 'tutorial' });
});

test('rejects missing, unquoted multiple, unsupported, and conflicting arguments', () => {
  for (const args of [[], ['My', 'post'], ['Title', '--draft'], ['Title', '--note', '--tutorial'], ['Title', '--note', '--note']]) {
    assert.throws(() => parseArgs(args), /title|option|format/i, JSON.stringify(args));
  }
});

test('normalizes punctuation and Unicode into a usable filename', () => {
  assert.equal(slugTitle('  Café: a new idea?!  '), 'cafe-a-new-idea');
  assert.equal(slugTitle('你好，世界'), '你好-世界');
  assert.equal(slugTitle('Why “AI” — and art?'), 'why-ai-and-art');
});

test('only retains Unicode characters that Astro can preserve in the post ID', () => {
  assert.equal(slugTitle('AI ፩'), 'ai');
  assert.equal(slugTitle('AI ፪'), 'ai');
  assert.throws(() => slugTitle('፩'), /title|letter|number/i);
});

test('different Unicode titles cannot create the same Astro post ID', async (t) => {
  const contentDir = await temporaryDirectory(t);
  const destination = await createPost({ title: 'AI ፩' }, { contentDir, now: today });
  assert.equal(githubSlug(path.basename(destination, '.md')), 'ai');
  await assert.rejects(createPost({ title: 'AI ፪' }, { contentDir, now: today }), /exists|collision/i);
  assert.deepEqual(await readdir(contentDir), ['ai.md']);
});

test('rejects empty or unsafe title content and overlong filenames', () => {
  for (const title of ['', '   ', '…! / ../', '🧠', 'New\n---\ndraft: false', 'Nul\u0000title', 'a'.repeat(250)]) {
    assert.throws(() => slugTitle(title), /title|long/i, JSON.stringify(title));
  }
});

test('generated frontmatter round-trips through the installed Astro parser', async (t) => {
  const contentDir = await temporaryDirectory(t);
  const title = 'Gio’s "quoted" title: #yes \\ paths & [ideas]';
  const destination = await createPost({ title }, { contentDir, now: today });
  const markdown = await readFile(destination, 'utf8');
  const { frontmatter, content } = parseFrontmatter(markdown);
  assert.deepEqual(frontmatter, {
    title,
    date: new Date('2026-10-24T00:00:00.000Z'),
    description: '',
    draft: true,
    featured: false,
    tags: [],
    format: 'essay',
    archived: false,
  });
  assert.equal(path.dirname(destination), contentDir);
  assert.match(content, /Write/);
});

for (const format of ['note', 'tutorial']) {
  test(`creates the ${format} format as a draft`, async (t) => {
    const contentDir = await temporaryDirectory(t);
    const destination = await createPost({ title: `A ${format}`, format }, { contentDir, now: today });
    const { frontmatter } = parseFrontmatter(await readFile(destination, 'utf8'));
    assert.equal(frontmatter.format, format);
    assert.equal(frontmatter.draft, true);
  });
}

test('path-like titles cannot write outside the requested content directory', async (t) => {
  const root = await temporaryDirectory(t);
  const contentDir = path.join(root, 'essays');
  const destination = await createPost({ title: '../../outside/secret' }, { contentDir, now: today });
  assert.equal(destination, path.join(contentDir, 'outside-secret.md'));
  assert.deepEqual(await readdir(root), ['essays']);
});

test('refuses to overwrite an existing post', async (t) => {
  const contentDir = await temporaryDirectory(t);
  await writeFile(path.join(contentDir, 'my-post.md'), 'Existing body\n');
  await assert.rejects(createPost({ title: 'My Post' }, { contentDir, now: today }), /exists|collision/i);
  assert.equal(await readFile(path.join(contentDir, 'my-post.md'), 'utf8'), 'Existing body\n');
});

test('refuses equivalent Markdown filenames, including differently cased extensions', async (t) => {
  const contentDir = await temporaryDirectory(t);
  await writeFile(path.join(contentDir, 'My Post!.MD'), 'Original');
  await assert.rejects(createPost({ title: 'My Post' }, { contentDir, now: today }), /exists|collision/i);
  assert.deepEqual(await readdir(contentDir), ['My Post!.MD']);
});

test('refuses punctuation-only differences that Astro removes from post IDs', async (t) => {
  const contentDir = await temporaryDirectory(t);
  await writeFile(path.join(contentDir, 'My.Post!.md'), 'Original');
  await assert.rejects(createPost({ title: 'MyPost' }, { contentDir, now: today }), /exists|collision/i);
});

test('refuses an existing filename whose Unicode characters Astro removes', async (t) => {
  const contentDir = await temporaryDirectory(t);
  await writeFile(path.join(contentDir, 'a፩b.md'), 'Original');
  await assert.rejects(createPost({ title: 'ab' }, { contentDir, now: today }), /exists|collision/i);
  assert.deepEqual(await readdir(contentDir), ['a፩b.md']);
});

test('refuses an existing post with an explicit matching frontmatter slug', async (t) => {
  const contentDir = await temporaryDirectory(t);
  await writeFile(path.join(contentDir, 'older-title.md'), '---\ntitle: Older title\nslug: my-post\n---\nOriginal');
  await assert.rejects(createPost({ title: 'My Post' }, { contentDir, now: today }), /exists|collision/i);
  assert.deepEqual(await readdir(contentDir), ['older-title.md']);
});

test('refuses an existing nested index with the same post ID', async (t) => {
  const contentDir = await temporaryDirectory(t);
  await mkdir(path.join(contentDir, 'my-post'));
  await writeFile(path.join(contentDir, 'my-post', 'index.md'), 'Original');
  await assert.rejects(createPost({ title: 'My Post' }, { contentDir, now: today }), /exists|collision/i);
});

test('refuses a destination symlink without touching its target', async (t) => {
  const root = await temporaryDirectory(t);
  const contentDir = path.join(root, 'essays');
  await mkdir(contentDir);
  const existing = path.join(root, 'existing.md');
  await writeFile(existing, 'Keep this');
  await symlink(existing, path.join(contentDir, 'my-post.md'));
  await assert.rejects(createPost({ title: 'My Post' }, { contentDir, now: today }), /exists|collision/i);
  assert.equal(await readFile(existing, 'utf8'), 'Keep this');
});

test('concurrent creation of the same post allows exactly one writer', async (t) => {
  const contentDir = await temporaryDirectory(t);
  const results = await Promise.allSettled([
    createPost({ title: 'One post' }, { contentDir, now: today }),
    createPost({ title: 'One post' }, { contentDir, now: today }),
  ]);
  assert.equal(results.filter((result) => result.status === 'fulfilled').length, 1);
  assert.deepEqual(await readdir(contentDir), ['one-post.md']);
});

test('invalid options fail before creating the target directory', async (t) => {
  const root = await temporaryDirectory(t);
  const contentDir = path.join(root, 'essays');
  await assert.rejects(createPost({ title: 'A title', format: 'video' }, { contentDir, now: today }), /format/i);
  await assert.rejects(createPost({ title: 'A title' }, { contentDir, now: new Date('invalid') }), /date/i);
  assert.deepEqual(await readdir(root), []);
});

test('CLI reports invalid input with a nonzero exit and useful usage', () => {
  const script = fileURLToPath(new URL('../scripts/new-post.mjs', import.meta.url));
  const result = spawnSync(process.execPath, [script, 'Title', '--unknown'], { encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /unknown|unsupported/i);
  assert.match(result.stderr, /npm run new/);
});

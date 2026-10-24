import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFile, readdir, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkBuild } from './check-build.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const created = [];
const prefix = `publication-check-${randomUUID().slice(0, 8)}`;
const titleMarker = `Publication check ${prefix}`;
const bodyMarker = `Unpublished fixture body ${prefix}`;
const now = new Date();
const past = new Date(now.valueOf() - 86_400_000).toISOString();
const future = new Date(now.valueOf() + 31_536_000_000).toISOString();
const fixtures = [
  { name: 'draft', date: past, draft: true, archived: false },
  { name: 'archived-draft', date: past, draft: true, archived: true },
  { name: 'future', date: future, draft: false, archived: false },
  { name: 'archived-future', date: future, draft: false, archived: true },
];

function build() {
  const result = spawnSync(process.execPath, [path.join(root, 'node_modules/astro/bin/astro.mjs'), 'build'], {
    cwd: root,
    stdio: 'inherit',
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `Astro build failed${result.signal ? ` (${result.signal})` : ''}`);
}

async function assertNoFixtureContent(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await assertNoFixtureContent(filename);
    } else {
      // Byte scanning also covers text bundled into scripts, JSON, and source maps.
      const contents = await readFile(filename);
      for (const marker of [prefix, titleMarker, bodyMarker]) {
        assert.ok(!contents.includes(marker), `Fixture content leaked into ${path.relative(root, filename)}`);
      }
    }
  }
}

let failure;
try {
  for (const fixture of fixtures) {
    const filename = path.join(root, 'src/content/essays', `${prefix}-${fixture.name}.md`);
    await writeFile(filename, `---
title: "${titleMarker}: ${fixture.name}"
date: ${fixture.date}
description: "Temporary production visibility fixture."
draft: ${fixture.draft}
archived: ${fixture.archived}
featured: true
format: note
tags: ["publication-check"]
---

${bodyMarker}. This must never appear in a production build.
`, { encoding: 'utf8', flag: 'wx' });
    created.push(filename);
  }
  build();
  await checkBuild({ root, now });
  await assertNoFixtureContent(path.join(root, 'dist'));
  console.log('Publication fixtures passed: featured drafts and future posts, with and without archive status, are absent from production.');
  // Also prove a valid Unicode filename can be published and checked end to end.
  const control = path.join(root, 'src/content/essays', `${prefix}-你好-世界.md`);
  await writeFile(control, `---\ntitle: "Public Unicode control"\ndate: ${past}\ndescription: "Temporary published control."\ndraft: false\n---\n\nA public control for URL encoding.\n`, { flag: 'wx' });
  created.push(control);
  build();
  await checkBuild({ root, now });
  console.log('Published Unicode control is discoverable and its route resolves.');
} catch (error) {
  failure = error;
} finally {
  // Remove only the files this run created, even when the first build/check fails.
  try {
    await Promise.all(created.map((filename) => unlink(filename)));
    build();
    await checkBuild({ root });
  } catch (error) {
    failure = failure ? new AggregateError([failure, error], 'Fixture check and cleanup verification failed') : error;
  }
}

if (failure) {
  console.error(failure);
  process.exitCode = 1;
} else {
  console.log('Temporary fixtures removed; normal production build restored and checked.');
}

import test from 'node:test';
import assert from 'node:assert/strict';
import { markdownToHtml } from 'satteri';
import { headingHierarchy } from '../src/lib/heading-hierarchy.mjs';

test('renders a coherent heading hierarchy without changing words or code', () => {
  const markdown = '### First\n\n### Second\n\n## Third\n\n### Fourth\n\n##### Fifth\n\n## Sixth\n\n# Seventh\n\n## Eighth\n\n```python\n# code comment\n```';
  const { html } = markdownToHtml(markdown, { mdastPlugins: [headingHierarchy] });
  assert.deepEqual([...html.matchAll(/<h(\d)[^>]*>(.*?)<\/h\d>/g)].map(match => [Number(match[1]), match[2]]), [
    [2, 'First'], [2, 'Second'], [2, 'Third'], [3, 'Fourth'], [4, 'Fifth'], [2, 'Sixth'], [2, 'Seventh'], [3, 'Eighth'],
  ]);
  assert.match(html, /# code comment/);
});

test('heading hierarchy state resets for each document', () => {
  markdownToHtml('## Parent\n\n### Child', { mdastPlugins: [headingHierarchy] });
  const { html } = markdownToHtml('### Separate article', { mdastPlugins: [headingHierarchy] });
  assert.match(html, /<h2>Separate article<\/h2>/);
});

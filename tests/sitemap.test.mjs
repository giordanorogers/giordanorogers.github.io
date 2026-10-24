import test from 'node:test';
import assert from 'node:assert/strict';
import { createSitemap, escapeXml } from '../src/lib/sitemap.mjs';

test('XML values escape all five reserved characters', () => {
  assert.equal(escapeXml(`A & B < C > D "quoted" 'single'`), 'A &amp; B &lt; C &gt; D &quot;quoted&quot; &apos;single&apos;');
});

test('sitemap lists only canonical discoverable pages and uses updated dates', () => {
  const entries = [
    { id: 'visible&searchable', data: { date: new Date('2026-01-02'), updated: new Date('2026-10-01') } },
    { id: 'archived', data: { date: new Date('2026-01-02'), archived: true } },
    { id: 'draft', data: { date: new Date('2026-01-02'), draft: true } },
    { id: 'future', data: { date: new Date('2027-01-02') } },
  ];
  const xml = createSitemap(entries, new URL('https://example.com'), new Date('2026-10-24'));
  assert.match(xml, /<loc>https:\/\/example\.com\/<\/loc>/);
  assert.match(xml, /<loc>https:\/\/example\.com\/about\/<\/loc>/);
  assert.match(xml, /<loc>https:\/\/example\.com\/essays\/visible&amp;searchable\/<\/loc>/);
  assert.match(xml, /<lastmod>2026-10-01T00:00:00\.000Z<\/lastmod>/);
  assert.equal((xml.match(/<url>/g) ?? []).length, 3);
  assert.doesNotMatch(xml, /\/archived\/|\/draft\/|\/future\/|\/posts\/|\/404\//);
});

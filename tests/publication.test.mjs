import test from 'node:test';
import assert from 'node:assert/strict';
import { getPublicPosts, isDiscoverable, isPublished } from '../src/lib/publication.mjs';

const now = new Date('2026-10-24T12:00:00.000Z');

function post(id, data = {}) {
  return { id, data: { date: new Date('2026-10-20T00:00:00.000Z'), ...data } };
}

// Draft status always wins. Archive status affects discovery, not direct access.
const cases = [
  ['past public', '2026-10-24T11:59:59.999Z', false, false, true, true],
  ['past archived', '2026-10-24T11:59:59.999Z', false, true, true, false],
  ['past draft', '2026-10-24T11:59:59.999Z', true, false, false, false],
  ['past archived draft', '2026-10-24T11:59:59.999Z', true, true, false, false],
  ['exactly now public', '2026-10-24T12:00:00.000Z', false, false, true, true],
  ['exactly now archived', '2026-10-24T12:00:00.000Z', false, true, true, false],
  ['exactly now draft', '2026-10-24T12:00:00.000Z', true, false, false, false],
  ['exactly now archived draft', '2026-10-24T12:00:00.000Z', true, true, false, false],
  ['future public', '2026-10-24T12:00:00.001Z', false, false, false, false],
  ['future archived', '2026-10-24T12:00:00.001Z', false, true, false, false],
  ['future draft', '2026-10-24T12:00:00.001Z', true, false, false, false],
  ['future archived draft', '2026-10-24T12:00:00.001Z', true, true, false, false],
];

for (const [name, date, draft, archived, published, discoverable] of cases) {
  test(name, () => {
    const entry = post(name, { date: new Date(date), draft, archived });
    assert.equal(isPublished(entry, now), published);
    assert.equal(isDiscoverable(entry, now), discoverable);
  });
}

test('existing entries without new flags remain public', () => {
  assert.equal(isDiscoverable(post('existing'), now), true);
});

test('invalid publication dates fail closed', () => {
  assert.equal(isPublished(post('invalid', { date: new Date('invalid') }), now), false);
});

test('discovery filters before sorting newest first without changing the collection', () => {
  const entries = [
    post('older'),
    post('archived', { archived: true }),
    post('future', { date: new Date('2027-01-01') }),
    post('newer', { date: new Date('2026-10-23') }),
    post('draft', { draft: true }),
  ];
  assert.deepEqual(getPublicPosts(entries, now).map((entry) => entry.id), ['newer', 'older']);
  assert.deepEqual(entries.map((entry) => entry.id), ['older', 'archived', 'future', 'newer', 'draft']);
});

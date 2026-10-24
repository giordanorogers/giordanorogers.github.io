import { getPublicPosts } from './publication.mjs';

/** @param {string} value */
export function escapeXml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}

/**
 * @param {Array<{ id: string, data: { date: Date, updated?: Date, draft?: boolean, archived?: boolean } }>} posts
 * @param {URL | string} site
 */
export function createSitemap(posts, site, now = new Date()) {
  const pages = [
    { path: '/', updated: undefined },
    { path: '/about/', updated: undefined },
    ...getPublicPosts(posts, now).map((post) => ({
      path: `/essays/${post.id}/`,
      updated: post.data.updated ?? post.data.date,
    })),
  ];
  const urls = pages.map(({ path, updated }) => {
    const location = escapeXml(new URL(path, site).href);
    const lastModified = updated ? `<lastmod>${updated.toISOString()}</lastmod>` : '';
    return `  <url><loc>${location}</loc>${lastModified}</url>`;
  });

  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.join('\n')}\n</urlset>\n`;
}

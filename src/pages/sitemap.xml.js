import { getCollection } from 'astro:content';
import { createSitemap } from '../lib/sitemap.mjs';

export async function GET({ site }) {
  const posts = await getCollection('essays');
  return new Response(createSitemap(posts, site), {
    headers: { 'Content-Type': 'application/xml; charset=utf-8' },
  });
}

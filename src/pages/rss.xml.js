import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { getPublicPosts } from '../lib/publication.mjs';

export async function GET(context) {
  const essays = getPublicPosts(await getCollection('essays'));
  return rss({
    title: 'Gio Rogers',
    description: 'Posts on AI safety, interpretability, research, and art.',
    site: context.site,
    items: essays.map((essay) => ({
      title: essay.data.title,
      pubDate: essay.data.date,
      description: essay.data.description,
      link: `/essays/${essay.id}/`,
      categories: [...new Set([essay.data.format, ...essay.data.tags])],
    })),
  });
}

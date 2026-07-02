import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const essays = (await getCollection('essays')).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );
  return rss({
    title: 'Gio Rogers',
    description: 'Essays on AI safety, interpretability, and research.',
    site: context.site,
    items: essays.map((essay) => ({
      title: essay.data.title,
      pubDate: essay.data.date,
      description: essay.data.description,
      link: `/essays/${essay.id}/`,
    })),
  });
}

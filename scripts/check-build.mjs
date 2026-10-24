import assert from 'node:assert/strict';
import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
// These are already installed with Astro; this checker adds no production dependency.
import { parseFrontmatter } from '@astrojs/internal-helpers/frontmatter';
import { slug } from 'github-slugger';

const repository = fileURLToPath(new URL('../', import.meta.url));

function decode(value) {
  const named = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };
  return value.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos);/gi, (_, entity) => {
    if (entity.startsWith('#')) return String.fromCodePoint(parseInt(entity.slice(entity[1].toLowerCase() === 'x' ? 2 : 1), entity[1].toLowerCase() === 'x' ? 16 : 10));
    return named[entity.toLowerCase()];
  });
}

function attributes(tag) {
  return Object.fromEntries([...tag.matchAll(/([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)]
    .map((match) => [match[1].toLowerCase(), decode(match[2] ?? match[3])]));
}

function tags(html, name) {
  return [...html.matchAll(new RegExp(`<${name}\\b[^>]*>`, 'gi'))].map((match) => attributes(match[0]));
}

function metadata(html) {
  return Object.fromEntries(tags(html, 'meta').map((tag) => [tag.name ?? tag.property, tag.content]));
}

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  return (await Promise.all(entries.map((entry) => entry.isDirectory()
    ? files(path.join(directory, entry.name)) : [path.join(directory, entry.name)]))).flat();
}

async function exists(filename) {
  try { return (await stat(filename)).isFile(); } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return false;
    throw error;
  }
}

function routeFor(filename, dist) {
  const relative = path.relative(dist, filename).split(path.sep).join('/');
  return `/${relative.replace(/index\.html$/, '')}`;
}

/** Validate the actual generated files, independently of the publication helper. */
export async function checkBuild({ root = repository, now = new Date() } = {}) {
  const dist = path.join(root, 'dist');
  const config = (await import(pathToFileURL(path.join(root, 'astro.config.mjs')).href)).default;
  const site = new URL(config.site);
  assert.equal(site.origin, 'https://giorogers.com', 'Keep the existing production origin');
  assert.equal((await readFile(path.join(dist, 'CNAME'), 'utf8')).trim(), site.hostname, 'Built CNAME must match the site');
  assert.equal((await readFile(path.join(root, 'public/CNAME'), 'utf8')).trim(), site.hostname, 'Source CNAME must match the site');
  const builtFiles = await files(dist);
  const htmlFiles = builtFiles.filter((filename) => filename.endsWith('.html'));
  const htmlByFile = new Map(await Promise.all(htmlFiles.map(async (filename) => [filename, await readFile(filename, 'utf8')])));
  let localReferences = 0;

  for (const [filename, html] of htmlByFile) {
    const route = routeFor(filename, dist);
    const page = new URL(route, site);
    const meta = metadata(html);
    const redirect = config.redirects?.[route];
    const canonical = tags(html, 'link').filter((tag) => tag.rel === 'canonical');
    assert.equal(canonical.length, 1, `${route}: expected one canonical link`);
    assert.equal(canonical[0].href, new URL(typeof redirect === 'string' ? redirect : route, site).href, `${route}: incorrect canonical`);
    assert.match(html, /<title>[^<]+<\/title>/, `${route}: missing title`);
    if (redirect) {
      assert.match(meta.robots ?? '', /\bnoindex\b/, `${route}: redirects need noindex`);
    } else {
      for (const key of ['description', 'og:title', 'og:description', 'og:image', 'twitter:card', 'twitter:title', 'twitter:description', 'twitter:image']) {
        assert.ok(meta[key], `${route}: missing ${key} metadata`);
      }
      assert.equal(meta['og:url'], canonical[0].href, `${route}: Open Graph URL must match canonical`);
      assert.equal(meta['og:title'], meta['twitter:title'], `${route}: social titles differ`);
      assert.equal(meta.description, meta['og:description'], `${route}: descriptions differ`);
      if (route === '/404.html') assert.match(meta.robots ?? '', /\bnoindex\b/, '404 must be noindex');
    }

    // This checks emitted attributes, not script text or source-code strings.
    const markup = html.replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, (block) => block.slice(0, block.indexOf('>') + 1));
    for (const match of markup.matchAll(/<[a-z][^>]*>/gi)) {
      const attrs = attributes(match[0]);
      if (attrs.rel === 'canonical') continue; // Canonicals are checked above, including redirect destinations.
      for (const value of [attrs.href, attrs.src].filter(Boolean)) {
        const target = new URL(value, page);
        if (!['https:', 'http:'].includes(target.protocol) || target.origin !== site.origin) continue;
        const requested = path.resolve(dist, `.${decodeURIComponent(target.pathname)}`);
        assert.ok(requested.startsWith(`${dist}${path.sep}`) || requested === dist, `${route}: path escapes dist`);
        const resolved = await exists(requested) ? requested : path.join(requested, 'index.html');
        assert.ok(await exists(resolved), `${route}: broken local reference ${value}`);
        localReferences++;
        if (target.hash && /\.(html|svg)$/.test(resolved)) {
          const targetHtml = htmlByFile.get(resolved) ?? await readFile(resolved, 'utf8');
          const fragment = decodeURIComponent(target.hash.slice(1));
          const identifiers = [...targetHtml.matchAll(/<[a-z][^>]*>/gi)].map((tag) => attributes(tag[0]));
          assert.ok(identifiers.some((attrs) => attrs.id === fragment || attrs.name === fragment), `${route}: missing fragment ${value}`);
        }
      }
    }
  }

  const contentDirectory = path.join(root, 'src/content/essays');
  const posts = await Promise.all((await files(contentDirectory)).filter((filename) => filename.endsWith('.md')).map(async (filename) => {
    const { frontmatter: data } = parseFrontmatter(await readFile(filename, 'utf8'));
    const relative = path.relative(contentDirectory, filename).replace(/\.md$/, '');
    const id = data.slug ?? relative.split(path.sep).map((segment) => slug(segment)).join('/').replace(/\/index$/, '');
    const date = new Date(data.date);
    const published = !data.draft && date.valueOf() <= now.valueOf();
    return { id, data, date, published, discoverable: published && !data.archived };
  }));
  const homepage = await readFile(path.join(dist, 'index.html'), 'utf8');
  const rss = await readFile(path.join(dist, 'rss.xml'), 'utf8');
  const sitemap = await readFile(path.join(dist, 'sitemap.xml'), 'utf8');
  const publicUrls = posts.filter((post) => post.discoverable).map((post) => new URL(`/essays/${post.id}/`, site).href).sort();
  const feedUrls = [...rss.matchAll(/<item>\s*([\s\S]*?)<\/item>/g)].map((item) => decode(item[1].match(/<link>([^<]+)<\/link>/)?.[1] ?? '')).sort();
  const sitemapUrls = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((match) => decode(match[1])).sort();
  assert.deepEqual(feedUrls, publicUrls, 'RSS must list exactly the discoverable posts');
  assert.deepEqual(sitemapUrls, [new URL('/', site).href, new URL('/about/', site).href, ...publicUrls].sort(), 'Sitemap must contain home, about, and discoverable posts only');

  for (const post of posts) {
    const route = `/essays/${post.id}/`;
    const pathname = new URL(route, site).pathname;
    const filename = path.join(dist, 'essays', post.id, 'index.html');
    assert.equal(await exists(filename), post.published, `${route}: incorrect production route visibility`);
    const homepageLinks = tags(homepage, 'a').map((tag) => new URL(tag.href ?? '', site).pathname);
    assert.equal(homepageLinks.includes(pathname), post.discoverable, `${route}: incorrect homepage visibility`);
    if (post.published) {
      const meta = metadata(htmlByFile.get(filename));
      assert.equal(/\bnoindex\b/.test(meta.robots ?? ''), Boolean(post.data.archived), `${route}: incorrect noindex policy`);
      assert.equal(meta['article:published_time'], post.date.toISOString(), `${route}: publication date changed`);
      if (post.data.updated) assert.equal(meta['article:modified_time'], new Date(post.data.updated).toISOString(), `${route}: updated date missing`);
    }
    if (!post.discoverable) {
      for (const [source, html] of htmlByFile) {
        for (const section of html.matchAll(/<section\b[^>]*class="[^"]*\bkeep-reading\b[^"]*"[^>]*>([\s\S]*?)<\/section>/g)) {
          assert.ok(!tags(section[1], 'a').some((tag) => new URL(tag.href ?? '', site).pathname === pathname), `${path.relative(dist, source)}: recommends hidden post ${post.id}`);
        }
      }
    }
  }
  const robots = await readFile(path.join(dist, 'robots.txt'), 'utf8');
  assert.match(robots, new RegExp(`Sitemap: ${new URL('/sitemap.xml', site).href.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`));
  assert.ok(!/^Disallow:\s*\S/im.test(robots), 'Keep archived pages crawlable so noindex can be read');
  return { htmlPages: htmlFiles.length, localReferences, posts: posts.length };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    const result = await checkBuild();
    console.log(`Build checks passed: ${result.htmlPages} HTML pages, ${result.localReferences} local references, ${result.posts} posts.`);
  } catch (error) {
    console.error(`Build check failed: ${error.message}`);
    process.exitCode = 1;
  }
}

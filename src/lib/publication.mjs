/**
 * @typedef {{ data: { date: Date, draft?: boolean, archived?: boolean } }} PublicationEntry
 */

/** Published entries may have direct routes, including the archive. @param {PublicationEntry} post */
export function isPublished(post, now = new Date()) {
  return !post.data.draft && post.data.date.valueOf() <= now.valueOf();
}

/** Discovery surfaces omit archived entries as well as unpublished ones. @param {PublicationEntry} post */
export function isDiscoverable(post, now = new Date()) {
  return isPublished(post, now) && !post.data.archived;
}

/**
 * @template {PublicationEntry} T
 * @param {readonly T[]} posts
 * @returns {T[]}
 */
export function getPublicPosts(posts, now = new Date()) {
  return posts
    .filter((post) => isDiscoverable(post, now))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

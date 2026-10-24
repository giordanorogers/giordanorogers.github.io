// Render headings below the layout's h1, closing skipped levels in older posts.
// State belongs to the document, so parallel renders cannot affect each other.
export const headingHierarchy = {
  name: 'article-heading-hierarchy',
  heading(node, context) {
    const ancestors = context.data.articleHeadingAncestors ??= [];
    const sourceDepth = node.depth;
    while (ancestors.length && ancestors.at(-1).sourceDepth >= sourceDepth) ancestors.pop();
    const depth = Math.min(6, (ancestors.at(-1)?.depth ?? 1) + 1);
    context.setProperty(node, 'depth', depth);
    ancestors.push({ sourceDepth, depth });
  },
};

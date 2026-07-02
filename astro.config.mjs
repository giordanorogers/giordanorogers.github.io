import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://giorogers.com',
  markdown: {
    shikiConfig: {
      themes: { light: 'github-light', dark: 'vesper' },
      defaultColor: false,
    },
  },
  redirects: {
    '/posts/2025/07/activation_patching_residual/': '/essays/activation-patching-the-residual-stream/',
    '/posts/2025/08/what_is_creativity/': '/essays/on-creativity/',
    '/posts/2999/11/research_taste/': '/essays/research-taste/',
    '/posts/2025/12/ai_mediated_mediocrity/': '/essays/ai-mediated-mediocrity/',
  },
});

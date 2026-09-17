// @ts-check
import { defineConfig } from 'astro/config';

// User site (yugumishra.github.io) — served from the domain root, so no `base`.
// The Yope3D blog is a separate project site at /Yope3D with its own deploy.
export default defineConfig({
  site: 'https://yugumishra.github.io',
  vite: {
    optimizeDeps: {
      // The drawer imports TensorFlow only when opened. Prebundle it at startup
      // so Vite does not hand an already-open page an outdated dependency URL.
      include: ['@tensorflow/tfjs'],
    },
  },
});

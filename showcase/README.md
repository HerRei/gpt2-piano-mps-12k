# Piano showcase

The source for the GitHub Pages site is in this directory. The production build is committed in `docs/`, which Pages serves from `main`.

From the repository root:

```sh
npm ci
npm run dev
npm run build
```

Vite builds the static page. GSAP handles the entrance and scroll animations. Three.js renders the illustrated piano roll in a lazy-loaded module. Content remains readable without JavaScript, and a CSS piano illustration is used if WebGL is unavailable. The page respects reduced-motion preferences and includes an explicit motion toggle.

The audio is the selected v1 continuation rendered on Salamander Grand Piano by Alexander Holm (CC BY 3.0), with long rests shortened. It is never played automatically. The intensity sketch is a visual demonstration, not inference.

`public/social-preview.png` is a 1200 × 630 browser render of `public/social-preview.svg`. Keep the two in sync when changing the image. All fonts, scripts, and media are served locally.

Rebuild after every source or public asset edit. Do not edit generated `docs/assets/` files directly.

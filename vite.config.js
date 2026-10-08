import { defineConfig } from "vite";
export default defineConfig({
  root: "showcase",
  base: "./",
  build: { outDir: "../docs", emptyOutDir: true, chunkSizeWarningLimit: 650 },
});

import markdownIt from "markdown-it";
import texmath from "markdown-it-texmath";
import katex from "katex";
import fs from "node:fs";
import path from "node:path";

export default function (eleventyConfig) {
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  for (const entry of fs.readdirSync("src/projects", { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;

    const staticDirectory = path.join("src/projects", entry.name, "static");
    if (!fs.existsSync(staticDirectory)) continue;

    eleventyConfig.addPassthroughCopy({
      [staticDirectory]: path.join("projects", entry.name, "static"),
    });
  }
  eleventyConfig.addPassthroughCopy({
    "node_modules/katex/dist/katex.min.css": "assets/vendor/katex.min.css",
    "node_modules/katex/dist/fonts": "assets/vendor/fonts",
    "node_modules/markdown-it-texmath/css/texmath.css": "assets/vendor/texmath.css",
  });
  eleventyConfig.addPassthroughCopy("CNAME");
  // eleventyConfig.addPassthroughCopy("favicon.ico");
  // eleventyConfig.addPassthroughCopy("favicon-16x16.png");
  // eleventyConfig.addPassthroughCopy("favicon-32x32.png");
  // eleventyConfig.addPassthroughCopy("apple-touch-icon.png");
  // eleventyConfig.addPassthroughCopy("android-chrome-192x192.png");
  // eleventyConfig.addPassthroughCopy("android-chrome-512x512.png");

  eleventyConfig.addFilter("readableDate", (date) =>
    new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
      timeZone: "UTC",
    }).format(date),
  );
  eleventyConfig.addFilter("isoDate", (date) =>
    new Date(date).toISOString().slice(0, 10),
  );
  eleventyConfig.addFilter("monthYear", (date) =>
    new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      timeZone: "UTC",
    }).format(date),
  );
  eleventyConfig.addFilter("head", (items, count) => items.slice(0, count));
  eleventyConfig.addGlobalData("currentYear", () => new Date().getFullYear());

  eleventyConfig.addCollection("writing", (collectionApi) =>
    collectionApi
      .getFilteredByTag("writing")
      .sort((a, b) => b.date - a.date),
  );

  const markdownLibrary = markdownIt({
    html: true,
    linkify: true,
    typographer: false,
  }).use(texmath, {
    engine: katex,
    delimiters: "dollars",
    katexOptions: {
      throwOnError: false,
      strict: false,
    },
  });

  eleventyConfig.setLibrary("md", markdownLibrary);

  return {
    dir: {
      input: "src",
      includes: "_includes",
      output: "_site",
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
  };
}

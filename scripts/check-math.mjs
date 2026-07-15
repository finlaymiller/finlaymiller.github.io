import fs from "node:fs";

const pages = [
  "_site/cs285/supervised-learning/index.html",
  "_site/cs285/intro-to-deep-rl/index.html",
  "_site/cs285/policy-gradients/index.html",
  "_site/cs285/actor-critic-algorithms/index.html",
];

const failures = [];

for (const page of pages) {
  const html = fs.readFileSync(page, "utf8");

  if (!html.includes('class="katex')) {
    failures.push(`${page}: no build-time KaTeX output found`);
  }

  if (html.includes("$$")) {
    failures.push(`${page}: raw $$ delimiters remain in generated HTML`);
  }

  if (html.includes("katex-error")) {
    failures.push(`${page}: KaTeX reported an invalid expression`);
  }

  if (/\\[A-Za-z]+[^<\n]*<em>/.test(html)) {
    failures.push(`${page}: Markdown emphasis split a TeX expression`);
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(`Checked ${pages.length} math-heavy pages; all equations were rendered by KaTeX.`);

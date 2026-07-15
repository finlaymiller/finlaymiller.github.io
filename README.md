# finlaymiller.com

A deliberately small personal site built with [Eleventy](https://www.11ty.dev/).
The main site uses shared templates; academic project pages are independent
microsites with their own HTML, CSS, JavaScript, and media.

## Local development

```sh
npm install
npm run dev
```

Eleventy prints the local preview URL. A production build is generated with:

```sh
npm run build
```

The generated site is written to `_site/`.

## Add writing

Create a Markdown file under `src/writing/` with front matter like:

```yaml
---
layout: layouts/post.njk
title: "Post title"
date: 2026-07-15
excerpt: "One-sentence summary"
tags:
  - writing
permalink: /writing/post-slug/index.html
math: false
---
```

Set `math: true` when a post uses TeX notation.

Math is rendered to HTML by KaTeX during the Eleventy build. Use single-dollar
delimiters for inline expressions (`$E = mc^2$`) and double-dollar delimiters on
their own lines for display equations:

```text
$$
E = mc^2
$$
```

Run `npm test` to build the site and verify that math-heavy pages contain no
unparsed delimiters or KaTeX errors.

## Add an academic project microsite

Duplicate `src/projects/example-project/`, rename the copied directory to the
project slug, and edit its `index.html` and `static/` assets. The page is kept
outside the main site layout so a project can use its own visual identity and
dependencies without affecting the personal site.

Eleventy automatically copies the `static/` directory for every project. If you
create a new project while `npm run dev` is already running, restart the server
once so Eleventy discovers the new directory.

Update both `src/projects/index.njk` and `src/index.njk` to link to the project.
Remove `noindex` from the project page when its placeholder content has been
replaced.

## Deployment

Pushing `master` runs `.github/workflows/deploy-pages.yml`, builds Eleventy, and
deploys `_site/` through GitHub Pages. In the repository's Pages settings, select
**GitHub Actions** as the publishing source and configure `finlaymiller.com` as
the custom domain.

import fs from "node:fs";
import path from "node:path";

const projectsRoot = "src/projects";
const failures = [];

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(fullPath) : [fullPath];
  });
}

for (const entry of fs.readdirSync(projectsRoot, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;

  const project = entry.name;
  const sourceIndex = path.join(projectsRoot, project, "index.html");
  const sourceStatic = path.join(projectsRoot, project, "static");
  if (!fs.existsSync(sourceIndex)) continue;

  const outputIndex = path.join("_site/projects", project, "index.html");
  if (!fs.existsSync(outputIndex)) {
    failures.push(`${project}: generated index.html is missing`);
  }

  if (!fs.existsSync(sourceStatic)) continue;

  for (const sourceFile of walk(sourceStatic)) {
    const relativeFile = path.relative(sourceStatic, sourceFile);
    const outputFile = path.join("_site/projects", project, "static", relativeFile);
    if (!fs.existsSync(outputFile)) {
      failures.push(`${project}: static/${relativeFile} was not copied`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("All project microsite HTML and static assets are present in the build.");

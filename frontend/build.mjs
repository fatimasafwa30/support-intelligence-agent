import {mkdir, copyFile} from 'node:fs/promises';
const files = ['index.html', 'styles.css', 'view.js', 'app.js'];
await mkdir(new URL('./dist/', import.meta.url), {recursive: true});
for (const file of files) {
  await copyFile(new URL(file, import.meta.url), new URL(`dist/${file}`, import.meta.url));
}
console.log(`Production build complete: ${files.length} assets, no runtime dependencies.`);

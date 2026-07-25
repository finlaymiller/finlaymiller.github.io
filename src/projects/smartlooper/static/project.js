const copyButton = document.querySelector("[data-copy-bibtex]");
const bibtex = document.querySelector("#bibtex");

if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      const previousLabel = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = previousLabel;
      }, 1800);
    } catch {
      copyButton.textContent = "Select and copy manually";
    }
  });
}

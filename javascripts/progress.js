// Thin scroll-progress bar pinned to the top of the viewport.
// Directly visualizes reading progress through the single-page report.
(function () {
    function ensureBar() {
      var bar = document.getElementById("scroll-progress");
      if (!bar) {
        bar = document.createElement("div");
        bar.id = "scroll-progress";
        document.body.appendChild(bar);
      }
      return bar;
    }
    function update() {
      var bar = ensureBar();
      var doc = document.documentElement;
      var scrolled = doc.scrollTop || document.body.scrollTop;
      var height = doc.scrollHeight - doc.clientHeight;
      bar.style.width = (height > 0 ? (scrolled / height) * 100 : 0) + "%";
    }
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    document.addEventListener("DOMContentLoaded", update);
    update();
  })();
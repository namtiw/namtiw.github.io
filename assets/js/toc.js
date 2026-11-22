// Side contents menu for posts.
//
// _layouts/post.html renders an empty <nav id="post-toc" hidden> unless the
// post sets `toc: false` in its front matter. This script fills it with the
// post's h2/h3 headings, pins it to the left edge of the screen, bolds the
// heading currently being read, and opens only the current section's h3s
// (accordion). CSS in assets/css/toc.css hides it below 1080px wide.
(function () {
  "use strict";

  var NAV_GAP = 24;        // px between the navbar and the menu
  var EDGE = 16;           // px from the left edge of the screen
  var GAP_TO_TEXT = 24;    // px between the menu and the post text
  var MAX_WIDTH = 260;
  var ACTIVE_LINE = 0.25;  // a heading is "current" once above 25% of the screen

  function init() {
    var nav = document.getElementById("post-toc");
    var post = document.querySelector(".blog-post");
    if (!nav || !post) return;

    // Headings inside collapsed blocks are hidden on load, so leave them out.
    var heads = Array.prototype.filter.call(
      post.querySelectorAll("h2[id], h3[id]"),
      function (h) { return !h.closest(".collapsiblecontent"); }
    );
    if (heads.length < 2) return;

    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var navbar = document.querySelector(".navbar");

    // ---- build the list ----
    var label = document.createElement("div");
    label.className = "post-toc-label";
    label.textContent = "Contents";
    var root = document.createElement("ul");
    nav.appendChild(label);
    nav.appendChild(root);

    var items = [];      // { head, link, group } in document order
    var group = null;    // current h2 entry: { li, sub }

    heads.forEach(function (h) {
      var link = document.createElement("a");
      link.href = "#" + h.id;
      link.textContent = headingText(h);
      var li = document.createElement("li");
      li.appendChild(link);

      if (h.tagName === "H2" || !group) {
        // An h3 before any h2 is listed at the top level.
        li.className = "post-toc-h2";
        root.appendChild(li);
        group = { li: li, sub: null };
      } else {
        if (!group.sub) {
          var wrap = document.createElement("div");
          wrap.className = "post-toc-sub";
          group.sub = document.createElement("ul");
          wrap.appendChild(group.sub);
          group.li.appendChild(wrap);
        }
        group.sub.appendChild(li);
      }
      items.push({ head: h, link: link, group: group, top: h.tagName === "H2" || li.className === "post-toc-h2" });
    });

    nav.hidden = false;

    // ---- click to jump ----
    nav.addEventListener("click", function (e) {
      var a = e.target.closest("a");
      if (!a) return;
      var target = document.getElementById(decodeURIComponent(a.hash.slice(1)));
      if (!target) return;
      e.preventDefault();
      var y = target.getBoundingClientRect().top + window.pageYOffset - navbarBottom() - 16;
      window.scrollTo({ top: y, behavior: reduceMotion ? "auto" : "smooth" });
      history.replaceState(null, "", a.hash);
    });

    // ---- position and current heading ----
    var current = -1;

    function navbarBottom() {
      return navbar ? Math.max(0, navbar.getBoundingClientRect().bottom) : 0;
    }

    function place() {
      var r = post.getBoundingClientRect();
      var top = Math.max(navbarBottom() + NAV_GAP, r.top);
      var width = Math.min(MAX_WIDTH, r.left - EDGE - GAP_TO_TEXT);
      nav.style.setProperty("--toc-top", top + "px");
      nav.style.setProperty("--toc-left", EDGE + "px");
      nav.style.setProperty("--toc-width", Math.max(0, width) + "px");
      // Fade out once the post has scrolled away (tags, footer).
      nav.classList.toggle("post-toc-gone", r.bottom < top + 120);
    }

    function pickCurrent() {
      var line = window.innerHeight * ACTIVE_LINE;
      var idx = -1;
      for (var i = 0; i < items.length; i++) {
        if (items[i].head.getBoundingClientRect().top <= line) idx = i;
        else break;
      }
      // Short last sections never reach the line: at the bottom, take the last one.
      var atBottom = window.innerHeight + window.pageYOffset >= document.documentElement.scrollHeight - 2;
      if (atBottom) idx = items.length - 1;
      return idx;
    }

    function mark(idx) {
      if (idx === current) return;
      current = idx;
      var openGroup = idx >= 0 ? items[idx].group : null;
      items.forEach(function (it, i) {
        var on = i === idx;
        it.link.classList.toggle("active", on);
        if (on) it.link.setAttribute("aria-current", "location");
        else it.link.removeAttribute("aria-current");
        if (it.top) {
          it.group.li.classList.toggle("open", it.group === openGroup);
          // Bold-ish parent while reading one of its h3s.
          it.link.classList.toggle("ancestor", it.group === openGroup && !on);
        }
      });
    }

    var queued = false;
    function update() {
      queued = false;
      place();
      mark(pickCurrent());
    }
    function queue() {
      if (!queued) { queued = true; window.requestAnimationFrame(update); }
    }

    window.addEventListener("scroll", queue, { passive: true });
    window.addEventListener("resize", queue);
    window.addEventListener("load", queue);   // images and diagrams shift headings
    update();
  }

  // Heading text without footnote markers or rendered math.
  function headingText(h) {
    var c = h.cloneNode(true);
    Array.prototype.forEach.call(
      c.querySelectorAll("sup, .MathJax, mjx-container, script"),
      function (n) { n.remove(); }
    );
    return c.textContent.trim();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();

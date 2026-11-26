// Theme controller — shared by the app, admin page and generated team pages.
// The initial theme is set by an inline script in <head> so there is no flash.
(function () {
  var root = document.documentElement;
  var KEY = 'nfl_theme';
  var meta = document.querySelector('meta[name="theme-color"]');

  function apply(theme) {
    root.dataset.theme = theme;
    try {
      localStorage.setItem(KEY, theme);
    } catch (e) {
      /* private mode */
    }
    if (meta) meta.content = theme === 'dark' ? '#070b14' : '#f3f6fb';
    var button = document.getElementById('theme-toggle');
    if (button) {
      var next = theme === 'dark' ? 'light' : 'dark';
      button.setAttribute('aria-label', 'Switch to ' + next + ' theme');
      button.title = 'Switch to ' + next + ' theme';
    }
  }

  function current() {
    return root.dataset.theme === 'dark' ? 'dark' : 'light';
  }

  document.addEventListener('DOMContentLoaded', function () {
    apply(root.dataset.theme ? current() : 'light');
    var button = document.getElementById('theme-toggle');
    if (button) {
      button.addEventListener('click', function () {
        var next = current() === 'dark' ? 'light' : 'dark';
        apply(next);
        if (window.nflTrack) window.nflTrack('theme_toggle', next);
      });
    }
  });
})();
